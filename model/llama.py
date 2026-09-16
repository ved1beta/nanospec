import torch
from torch import nn

@dataclass
class ModelArgs:
    dim: int = 512              # embedding dimension
    n_layers: int = 8           # number of model decoder blocks
    n_heads: int = 8            # number of heads for queries embedding
    n_kv_heads: int = 4         # number of heads for keys and values embedding
    vocab_size: int = len(vocab) # Length of vocabulary
    multiple_of: int = 256        # Require to calculate dim of feedfoward network
    ffn_dim_multiplier: Optional[float] = None  # Require to calculate dim of feedfoward network
    norm_eps: float = 1e-5                       # Default Epsilon value set for the RMSNorm calculation
    rope_theta: float = 10000.0   # Default theta value for the RePE calculation

    max_batch_size: int = 10     # Max batch size
    max_seq_len: int = 256         # Max sequence length

    epochs: int = 2500             # Total number of training iteration
    log_interval: int = 10        # Number of interval to print the logs and loss values   
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'   # Assign device to cuda or cpu based on availability 


class RMSNorm(nn.Module):
  def __init__(self, dim: int, eps: float = 1e-6):
    super().__init__()
    device = ModelArgs.device
    self.eps = eps
    # Scaling parameter gamma, initialized with one and the no of parameters is equal to the size of dim
    self.weight = nn.Parameter(torch.ones(dim).to(device))

  def _norm(self, x):
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps).to(device)

  def forward(self, x):
    #Shape: x[bs,seq,dim]
    output = self._norm(x.float()).type_as(x)

    #Shape: x[bs,seq,dim] -> x_norm[bs,seq,dim]
    return output * self.weight

def precompute_freqs_cis(dim:int, seq_len: int, theta: float=10000.0):
  # Computing Theta value for each dim pair which is dim/2
  device = ModelArgs.device
  freqs = 1.0 / (theta ** (torch.arange(0, dim, 2,device=device)[:(dim//2)].float()/dim))

  # Computing range of positions(m) in the sequence
  t = torch.arange(seq_len, dtype=torch.float32, device=device)

  # freqs gives all the Theta value range for all the position of tokens in the sequence
  freqs = torch.outer(t, freqs).to(device)

  # This is the rotation matrix which needs to be converted to Polar form in order to perform rotation to the embedding
  freqs_cis = torch.polar(torch.ones_like(freqs).to(device), freqs).to(device)
  return freqs_cis

def reshape_for_broadcast(freqs_cis, x):
  ndim = x.ndim
  assert 0<=1<ndim
  assert freqs_cis.shape == (x.shape[1],x.shape[-1]), "the last two dimension of freqs_cis, x must match"
  shape = [d if i==1 or i==ndim-1 else 1 for i,d in enumerate(x.shape)]
  return freqs_cis.view(*shape)

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor)->Tuple[torch.Tensor, torch.Tensor]:
  device = ModelArgs.device
  # Applying rotary positional encoding to both query and key embedding together
  # First: The last dimension of xq and xk embedding needs to be reshaped to make it a pair. As rotation matrix is applied to each pair of dim.
  # Next: convert both xq and xk to complex number as the rotation matrix is only applicable to complex number
  xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2)).to(device)    #xq_:[bsz, seq_len, n_heads, head_dim/2]
  xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2)).to(device)    #xk_:[bsz, seq_len, n_heads, head_dim/2]

  # The rotation matrix(freqs_cis) dimensions across seq_len(dim=1) and head_dim(dim=3) should match with the embedding
  # Also, the shape freqs_cis should be the same with xq and xk, hence change the shape of freqs_cis:[seq_len,head_dim] -> freqs_cis:[1,seq_len,1,head_dim]
  freqs_cis = reshape_for_broadcast(freqs_cis, xq_)

  #Finally, perform rotation operation by multiplying with freqs_cis.
  #After the rotation is completed, convert both xq_out and xk_out back to real number and return
  xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3).to(device) #xq_out:[bsz, seq_len, n_heads, head_dim]
  xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3).to(device) #xk_out:[bsz, seq_len, n_heads, head_dim]
  return xq_out.type_as(xq), xk_out.type_as(xk)

class Attention(nn.Module):
  def __init__(self, args: ModelArgs):
    super().__init__()
    self.args = args
    # Embedding dimension
    self.dim = args.dim
    # Number of heads assigned to Query
    self.n_heads = args.n_heads
    # Number of heads assigned to Key and values. If "None", the number will be same as Query.
    self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
    # Dimension of each head relative to model dimension
    self.head_dim = args.dim // args.n_heads
    # Number of repetition in order to make time Key, Value heads to match Query heads number
    self.n_rep = args.n_heads // args.n_kv_heads

    # Weight initialize for Keys, Querys, Values and Oupt. Notice that the out_feature value of weight for q and kv are based on it's heads
    self.wq = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=False, device=device)
    self.wk = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False, device=device)
    self.wv = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False, device=device)
    self.wo = nn.Linear(self.n_heads * self.head_dim, self.dim, bias=False, device=device)

    # Initialize caches to store Key, Values at start. (KV Cache Implementation)
    self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim), device=args.device)
    self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim), device=args.device)

  def forward(self, x: torch.Tensor, start_pos, inference):
    # Shape of the input embedding: [bsz,seq_len,dim]
    bsz, seq_len, _ = x.shape
    # Mask will be used during 'Training' and is not required for 'inference' due to the use of KV cache.
    mask = None

    xq = self.wq(x)  #x[bsz,seq_len,dim]*wq[dim,n_heads * head_dim] -> q[bsz,seq_len,n_heads * head_dim]
    xk = self.wk(x)  #x[bsz,seq_len,dim]*wq[dim,n_kv_heads * head_dim] -> k[bsz,seq_len,n_kv_heads * head_dim]
    xv = self.wv(x)  #x[bsz,seq_len,dim]*wq[dim,n_kv_heads * head_dim] -> v[bsz,seq_len,n_kv_heads * head_dim]

    # Reshaping Querys, Keys and Values by their number of heads. (Group Query Attention Implementation)
    xq = xq.view(bsz, seq_len, self.n_heads, self.head_dim)      #xq[bsz,seq_len,n_heads, head_dim]
    xk = xk.view(bsz, seq_len, self.n_kv_heads, self.head_dim)   #xk[bsz,seq_len,n_kv_heads, head_dim]
    xv = xv.view(bsz, seq_len, self.n_kv_heads, self.head_dim)   #xv[bsz,seq_len,n_kv_heads, head_dim]

    # Model - Inference Mode: kv-cache is enabled at inference mode only.
    if inference:
      # Compute rotation matrix for each position in the sequence
      freqs_cis = precompute_freqs_cis(dim=self.head_dim, seq_len=self.args.max_seq_len * 2)
      # During inferencing, we should only take the rotation matrix range from the current position of the tokens.
      freqs_cis = freqs_cis[start_pos : start_pos + seq_len]
      # Apply RoPE to Queries and Keys embeddings
      xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

      self.cache_k = self.cache_k.to(xq)
      self.cache_v = self.cache_v.to(xq)
      # Store Keys and Values token embedding into their respective cache [KV Cache Implementation]
      self.cache_k[:bsz, start_pos:start_pos + seq_len] = xk
      self.cache_v[:bsz, start_pos:start_pos + seq_len] = xv

      # Assign all the previous tokens embeddings upto current tokens position to Keys and Values variable for Attention Calculation
      keys = self.cache_k[:bsz, :start_pos + seq_len]
      values = self.cache_v[:bsz, :start_pos + seq_len]

      # At this point, they Keys and Values shape aren't same with Queries Embedding which has to be in order to computer attention score
      # Use repeat_kv function to make Keys,Values shape same as queries shape
      keys = repeat_kv(keys, self.n_rep)      #keys[bsz,seq_len,n_heads,head_dim]
      values = repeat_kv(values, self.n_rep)  #values[bsz,seq_len,n_heads,head_dim]

    # Mode - Training mode: KV-Cache not implemented
    else:
      # Compute rotation matrix and apply RoPE to queries and keys for for training.
      freqs_cis = precompute_freqs_cis(dim=self.head_dim, seq_len=self.args.max_seq_len)

      #xq[bsz,seq_len,n_heads, head_dim], xk[bsz,seq_len,n_heads, head_dim]
      xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

      # Use repeat_kv function to make Keys,Values shape same as the queries shape
      #keys[bsz,seq_len,n_heads,head_dim], #values[bsz,seq_len,n_heads,head_dim]
      keys = repeat_kv(xk, self.n_rep)
      values = repeat_kv(xv, self.n_rep)

      # For training mode, we'll compute mask and apply to the attention score later
      mask = torch.full((seq_len, seq_len),float("-inf"),device=self.args.device)
      mask = torch.triu(mask, diagonal=1).to(self.args.device)

    # To compute attention, we'll need to perform a transpose operation to reshape all queries, keys and values bring heads at dim 1 and seq at dim 2
    xq = xq.transpose(1,2)                  #xq[bsz,n_heads,seq_len,head_dim]
    keys = keys.transpose(1,2)              #keys[bsz,n_heads,seq_len,head_dim]
    values = values.transpose(1,2)          #values[bsz,n_heads,seq_len,head_dim]

    # Computing attention score
    scores = torch.matmul(xq, keys.transpose(2,3)).to(self.args.device)/math.sqrt(self.head_dim)
    if mask is not None:
      scores = scores + mask

    # Apply softmax to the attention score
    scores = F.softmax(scores.float(), dim=-1).type_as(xq)
    # Matrix multiplication of attention score with the values
    output = torch.matmul(scores, values).to(self.args.device)

    # We get the contextual embedding for each head
    # All heads need to be reshaped back and combined to give a single single contextual attention output
    # Shape change: output[bsz,n_heads,seq_len,head_dim] -> output[bsz,seq_len, n_heads,head_dim] -> output[bsz,seq_len, n_heads * head_dim]
    output = output.transpose(1,2).contiguous().view(bsz, seq_len, -1)

    # shape: output [bsz,seq_len,dim]
    return self.wo(output)

# If the number of keys/values heads is less than query heads, this function expands the key/values embeddings with the required number of repetition
def repeat_kv(x:torch.Tensor, n_rep: int)-> torch.Tensor:
  bsz, seq_len, n_kv_heads, head_dim = x.shape
  if n_rep == 1:
    return x
  return (
      x[:,:,:,None,:]
      .expand(bsz,seq_len,n_kv_heads,n_rep, head_dim)
      .reshape(bsz,seq_len,n_kv_heads * n_rep, head_dim)
  )


class FeedForward(nn.Module):
  def __init__(self, dim:int, hidden_dim:int, multiple_of:int, ffn_dim_multiplier: Optional[float]):
    super().__init__()
    # Models embedding dimension
    self.dim = dim

    # We must use the hidden dimensions calculation shared by Meta which is the ideal one for this model
    # Hidden dimension are calculated such that it is a multiple of 256.
    hidden_dim = int(2 * hidden_dim/3)
    if ffn_dim_multiplier is not None:
      hidden_dim = int(ffn_dim_multiplier * hidden_dim)
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

    # define hiddne layers weights
    self.w1 = nn.Linear(self.dim, hidden_dim, bias=False, device=device)
    self.w2 = nn.Linear(hidden_dim, self.dim, bias=False, device=device)
    self.w3 = nn.Linear(self.dim, hidden_dim, bias=False, device=device)

  def forward(self, x):
    # Shape: [bsz,seq_len,dim]
    return self.w2(F.silu(self.w1(x)) * self.w3(x))



class TransformerBlock(nn.Module):
  def __init__(self, args: ModelArgs):
    super().__init__()
    self.args = args
    # Initilizate RMSNorm for attention
    self.attention_norm = RMSNorm(dim=args.dim, eps = args.norm_eps)
    # Initilizate Attention class
    self.attention = Attention(args)
    # Initilizate RMSNorm for feedfoward class
    self.ff_norm = RMSNorm(dim=args.dim, eps = args.norm_eps)
    # Initilizate feedfoward class
    self.feedforward = FeedForward(args.dim, 4 * args.dim, args.multiple_of, args.ffn_dim_multiplier)

  def forward(self, x, start_pos, inference):
    # start_pos = token position for inference mode, inference = True for inference and False for training mode
    # i) pass input embedding to attention_norm and then pass to attention block.
    # ii) the output of attention is then added to embedding(before norm)
    h = x + self.attention(self.attention_norm(x), start_pos, inference)

    # i) pass attention output to ff_norm and then pass to the feedforward network.
    # ii) the output of feedforward network is then added to the attention output(before ff_norm)
    out = h + self.feedforward(self.ff_norm(h))
    # Shape: [bsz,seq_len,dim]
    return out


class Transformer(nn.Module):
  def __init__(self, params: ModelArgs):
    super().__init__()
    # set all the ModelArgs in params variable
    self.params = params
    # Initilizate embedding class from the input block
    self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)

    # Initialize the decoder block and store it inside the ModuleList. 
    # This is because we've 4 decoder blocks in our Llama 3 model. (Official Llama 3 has 32 blocks)
    self.layers = nn.ModuleList()
    for layer_id in range(params.n_layers):
      self.layers.append(TransformerBlock(args=params))

    # Initilizate RMSNorm for the output block
    self.norm = RMSNorm(params.dim, eps = params.norm_eps)
    
    # Initilizate linear layer at the output block.
    self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

  def forward(self, x, start_pos=0, targets=None):
    
    # start_pos = token position for inference mode, inference = True for inference and False for training mode
    # x is the batch of token_ids generated from the texts or prompts using tokenizers.
    # x[bsz, seq_len] -> h[bsz, seq_len, dim]
    h = self.tok_embeddings(x)

    # If the target is none, Inference mode is activated and set to "True" and "False" if Training mode is activated.
    if targets is None:
      inference = True
    else:
      inference = False

    # The embeddings (h) will then pass though all the decoder blocks.
    for layer in self.layers:
      h = layer(h, start_pos, inference)

    # The output from the final decoder block will feed into the RMSNorm
    h = self.norm(h)

    # After normalized, the embedding h will then feed into the Linear layer. 
    # The main task of the Linear layer is to generate logits that maps the embeddings with the vocabulary size.
    # h[bsz, seq_len, dim] -> logits[bsz, seq_len, vocab_size]
    logits = self.output(h).float()
    loss = None

    # Inference mode is activated if the targets is not available
    if targets is None:
      loss = None
    # Training mode is activated if the targets are available. And Loss will be calculated for further model training. 
    else:
      loss = F.cross_entropy(logits.view(-1, self.params.vocab_size), targets.view(-1))

    return logits, loss
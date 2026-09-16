"""32 prompts for the correctness gates. Mixed length and register on purpose:
short factual, instructions, code, a little multilingual, one long-ish one."""

G1_PROMPTS = [
    "The capital of France is",
    "Write a haiku about a GPU that is always busy.",
    "Explain what a KV cache is in two sentences.",
    "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
    "Once upon a time, in a village at the edge of the forest,",
    "List three differences between TCP and UDP.",
    "Translate to French: The weather is nice today and I would like to go for a walk.",
    "Q: What is 17 * 23?\nA:",
    "Speculative decoding speeds up inference because",
    "Here is a recipe for a simple tomato soup:\n1.",
    "The following is a conversation between a user and a helpful assistant.\nUser: How do I reverse a linked list?\nAssistant:",
    "import torch\n\nclass RMSNorm(torch.nn.Module):\n",
    "In 1969, Neil Armstrong",
    "Summarize the plot of Hamlet in one paragraph.",
    "SELECT name, count(*) FROM users",
    "Why is the sky blue? Answer for a five-year-old.",
    "El rápido zorro marrón",
    "Dear hiring manager,\n\nI am writing to apply for",
    "The three laws of thermodynamics are",
    "Rewrite this sentence in passive voice: The cat chased the mouse.",
    "1, 1, 2, 3, 5, 8, 13,",
    "A limerick about a compiler:",
    "What are the main causes of the fall of the Roman Empire?",
    "// Compute the dot product of two float arrays of length n\nfloat dot(const float* a, const float* b, int n) {\n",
    "Give me five names for a coffee shop run by cats.",
    "The difference between a process and a thread is",
    "Tell me a joke about statisticians.",
    "Photosynthesis is the process by which",
    "How would you explain recursion to someone who has never programmed?",
    "Continue the story: The last human on Earth sat alone in a room. There was a knock at the door.",
    "Convert 100 degrees Fahrenheit to Celsius and show the formula.",
    "Explain, step by step, how paged attention keeps memory fragmentation low in a serving engine, "
    "and then contrast it with a naive contiguous per-request allocation, mentioning what happens "
    "when requests of different lengths arrive and finish out of order.",
]
assert len(G1_PROMPTS) == 32

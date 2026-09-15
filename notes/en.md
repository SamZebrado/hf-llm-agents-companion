# Hugging Face LLM & Agents Learner Companion

> Unofficial companion notes. The goal is to fill in prerequisites, learner questions, misconceptions, and practical pitfalls that the official course may only mention briefly. This is not a line-by-line translation.

## 1. Study strategy

- Treat text, diagrams, code/notebooks, and quizzes as the main path.
- Watch ordinary concept videos only when helpful; demo videos are often more valuable.
- When the course says *what* something is but not *why*, fill in the mechanism and math.
- Do not memorize model-specific conventions when a Model Card can answer them faster.

The LLM Course assumes some Python and introductory deep-learning background, so terms such as `batch`, `logits`, `cross-entropy`, `hidden state`, `head`, and `optimizer` may appear without a full introduction.

---

## 2. Core vocabulary

### Batch
A group of samples processed in parallel. `batch_size = 32` means 32 samples are handled together. Variable-length sequences are often padded to a common length.

### Token ID, embedding, hidden state

```text
text
→ tokenizer
→ token ID (discrete vocabulary index)
→ embedding lookup
→ token embedding (continuous vector)
→ Transformer layers
→ hidden state (contextualized vector)
```

### Logit
A **raw score** for a candidate class or token. It is not yet a probability. Softmax converts a set of logits into a probability distribution.

```text
cat  logit = 4.0
dog  logit = 3.0
car  logit = 1.0
```

The relative values matter more than the absolute scale.

### Label
The training target. In next-token prediction, the label at one position is the ID of the next true token.

### Head
A task-specific output module attached to the model body.

```text
hidden state
→ LM head / classification head / QA head
→ logits
```

---

## 3. Text vs. token classification

- **Text Classification**: one label for an entire sequence, such as sentiment or topic.
- **Token Classification**: one label per token position, such as NER or POS tagging.

Token-classification datasets are often annotated at the word level while tokenizers may split words into subwords. A common strategy is to supervise the first subtoken and assign `-100` to extra subtokens so those positions are ignored by the loss.

Those subtokens are still present in the model: they have embeddings, hidden states, and participate in attention.

---

## 4. BERT and MLM

**MLM = Masked Language Modeling**.

```text
The [MASK] sat on the mat.
→ BERT encoder
→ hidden state at [MASK]
→ MLM head
→ vocabulary logits
→ target label = cat
```

BERT is encoder-only and can use context on both sides of a token.

---

## 5. BART and T5

### BART
BART is an encoder-decoder Transformer pretrained by corrupting input text and reconstructing the full original sequence.

```text
original:  The cat sat on the mat.
corrupted: The <mask> on the mat.
target:    The cat sat on the mat.
```

Text infilling can replace an entire span with one `<mask>`, forcing the model to learn both content and missing-span length.

### T5
T5 (Text-to-Text Transfer Transformer) also uses an encoder-decoder architecture, but frames many NLP tasks uniformly as text-to-text.

```text
input:  The <extra_id_0> on the mat because <extra_id_1>.
target: <extra_id_0> cat sat <extra_id_1> it was tired <extra_id_2>
```

The target often contains only the missing spans, so it can be shorter than BART's full reconstruction target. That does not imply T5 is always globally more efficient.

---

## 6. Encoder-only / decoder-only / encoder-decoder

- **Encoder-only**: builds representations; natural for classification and token-level understanding tasks.
- **Decoder-only**: autoregressively predicts what comes next; common for GPT, Llama, DeepSeek, and general LLMs.
- **Encoder-decoder**: maps input sequence A to output sequence B; natural for translation and summarization.

`seq2seq` describes a task shape. Encoder-decoder is the classic implementation, but decoder-only models can also perform many seq2seq tasks.

---

## 7. Self-attention: Q / K / V

Use a small **3 tokens × 2 dimensions** example to keep sequence length separate from hidden dimension.

```text
X = [
  [1, 0],  # token A
  [0, 1],  # token B
  [1, 1]   # token C
]
```

A layer contains learnable parameter matrices:

```text
W_Q, W_K, W_V
```

The current activations are:

```text
Q = X @ W_Q
K = X @ W_K
V = X @ W_V
```

Distinguish:

- `W_Q / W_K / W_V`: learned model parameters.
- `Q / K / V`: activations computed from the current input.

Core equation:

```text
Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V
```

Interpretation:

1. `QK^T`: compare every query to every key.
2. `/ sqrt(d_k)`: keep scores from becoming too large.
3. `softmax`: turn each row into non-negative weights that sum to 1.
4. `×V`: mix value vectors using those weights.

If token A has attention weights `[0.6, 0.3, 0.1]`:

```text
O_A = 0.6 V_A + 0.3 V_B + 0.1 V_C
```

`O_A` is the attention output/context vector, not yet the final hidden state of the full Transformer block.

---

## 8. Softmax

```text
softmax(x_i) = exp(x_i) / sum_j exp(x_j)
```

Example:

```text
[0.707, 0]
→ [exp(0.707), exp(0)]
≈ [2.028, 1]
→ divide by 3.028
≈ [0.67, 0.33]
```

Softmax converts arbitrary real-valued scores into differentiable relative weights.

---

## 9. Multi-head attention and W_O

In full multi-head attention, the outputs from multiple heads are concatenated and passed through a learnable output projection:

```text
head_1, head_2, ...
→ concat
→ W_O
→ attention sublayer output
```

So `W_O` can also change the direction and scale of the attention branch before it is added back to the residual stream.

---

## 10. Residual connections, Pre-Norm, and MLP

Classic residual connection:

```text
y = x + F(x)
```

The skip path itself has no parameter.

A common Pre-Norm Transformer pattern:

```text
x = x + Attention(Norm(x))
x = x + MLP(Norm(x))
```

Pre-Norm does not erase the relative contribution of the residual path and the new branch. Normalization is applied before the branch; learned projections afterward can still alter its scale.

### MLP

**MLP = Multilayer Perceptron**. In a Transformer it can be viewed as:

```text
Linear → GELU/SiLU → Linear
```

A useful intuition:

- Attention: exchange information across tokens.
- MLP: further process each token's representation independently.

---

## 11. Is a Transformer recurrent?

A standard Transformer is not a recurrent architecture.

RNN:

```text
h_t = f(h_{t-1}, x_t)
```

Self-attention can process all positions of a known sequence in parallel. Decoder-only generation is still autoregressive and sequential at inference time, but that does not make the architecture an RNN.

---

## 12. Positional information

The sequence dimension of Q/K/V preserves token order: row 1 corresponds to token 1, row 2 to token 2, and so on.

But QKV alone does not encode what *position 1 vs. position 3* means, so models use position embeddings, RoPE, or related methods.

### Axial positional encoding

Think of it like splitting a position into high and low digits. With `l1 = 10` and `j = 37`:

```text
j % 10 = 7
j // 10 = 3
```

Two smaller embedding tables are queried and concatenated, reducing parameter and memory cost for very long sequences.

---

## 13. ViT

Vision Transformer turns image patches into token-like embeddings.

```text
224×224 RGB image
→ Conv2D(kernel=16, stride=16)
→ 14×14 = 196 patches
→ patch embeddings
→ + learnable [CLS]
→ + position embeddings
→ Transformer encoder
→ final CLS hidden state
→ MLP classification head
→ class logits
```

`[CLS]` is a learnable special token, not a Transformer layer. After multiple attention layers, its final hidden state can summarize information from the whole image/sequence.

---

## 14. Fine-tuning and weight updates

Hugging Face `Trainer` still wraps a standard training loop:

```python
outputs = model(**batch)
loss = outputs.loss
loss.backward()
optimizer.step()
optimizer.zero_grad()
```

`optimizer.step()` changes parameter values.

After training:

```python
weights = model.state_dict()
model.save_pretrained("./my_finetuned_model")
```

Ordinary chat inference normally does not run `backward()` or `optimizer.step()`, so a conversation does not immediately rewrite `W_Q/W_K/W_V/...`.

---

## 15. KV cache

During autoregressive generation, K/V for previous tokens have already been computed and can be cached:

```text
past tokens
→ cached K / V
new token
→ compute new Q/K/V only
→ new Q attends to past K/V
```

KV cache is related to service-level prompt caching/cache hits, but the terms are not identical.

---

## 16. Long-context attention

### Local attention
Attend only to a nearby window. Stacking layers gradually increases the receptive field.

### LSH attention
**LSH = Locality-Sensitive Hashing**.

Similar vectors are likely to hash into the same bucket, so attention only needs to compare likely neighbors rather than all pairs.

```text
Local attention: nearby in sequence position
LSH attention: nearby in representation space
```

---

## 17. Model routing vs. MoE vs. MTP

- **Model routing / cascade**: system-level choice of which whole model handles a request.
- **MoE routing**: token-level routing to a subset of expert subnetworks inside one model.
- **MTP (Multi-Token Prediction)**: predicts additional future tokens for training/decoding efficiency; it is not model routing.

---

## 18. Hugging Face practical intuition

### Hub
Choose the task first, then the model. Read the Model Card for task, inputs/outputs, license, size, tokenizer, and limitations.

### Inference API
Useful for quick model trials and checking I/O without downloading or deploying the full model.

### Task → model intuition

```text
whole text → one label          → Text Classification
one label per token             → Token Classification
find answer span in context     → Extractive QA
input sequence → output sequence→ seq2seq / encoder-decoder
continue from previous context  → decoder-only causal LM
```

---

## 19. Practical course pitfalls

### smolagents requires Python 3.10+

```bash
conda create -n hf-agents python=3.11 -y
conda activate hf-agents
python -m pip install --upgrade pip
python -m pip install 'smolagents[litellm]'
```

### Ollama is an optional local fallback
It manages model files itself; running `ollama pull` inside a project directory does not place the model inside that project.

### Old `question-answering` pipeline examples may not match current Transformers
When old course code breaks, check the migration guide before assuming your environment is wrong.

---

## 20. Notes on Chapter 1 depth

Chapter 1 functions more like a conceptual tour of Transformers and inference than a complete deep-learning course. Section depth varies considerably. `Bias and Limitations` is especially short and does not systematically cover hallucination, factuality, calibration, or prompt sensitivity.

There is no strong evidence that all missing depth is hidden in the videos. A practical approach is: official text as the backbone, then fill mechanism-level gaps when they matter.

---

## 21. Hallucination

Hallucination is not simply “random output.” A language model directly optimizes likely next tokens under context; it does not automatically perform an external truth check. A response can therefore be fluent and locally high-probability while still being factually wrong.

---

## 22. Temperature

Temperature rescales logits before softmax:

```text
p_i = softmax(logit_i / T)
```

- `T < 1`: sharper distribution, more deterministic.
- `T > 1`: flatter distribution, more diverse.
- `T = 1`: no extra rescaling.

---

## 23. Why these notes may help other learners and agents

This repository is increasingly a **FAQ + misconception dataset**. It records questions such as:

- logits vs. token IDs
- `W_Q/K/V` vs. `Q/K/V`
- whether `AV` is already the final hidden state
- where `W_O` comes from
- whether Pre-Norm erases residual magnitude differences
- why decoder-only models can still use an entire prompt

That makes the material useful both to human learners and as RAG/context for an agent asked to generate a beginner-friendly tutorial.

---

## References

- [Hugging Face LLM Course](https://huggingface.co/learn/llm-course/)
- [Hugging Face Agents Course](https://huggingface.co/learn/agents-course/)
- [Transformers documentation](https://huggingface.co/docs/transformers/)
- [smolagents documentation](https://huggingface.co/docs/smolagents/)

> This repository is unofficial, does not represent Hugging Face, and is not an exam answer bank.

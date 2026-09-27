# Hugging Face LLM & Agents 学习者补充笔记

> 非官方 companion notes。目标是补足课程中默认省略的前置知识、真实学习疑问、常见误解和实践踩坑，而不是逐段翻译原课。

## 1. 学习策略

- 正文、图、代码 / notebook、quiz 是主线。
- 普通概念视频按需看；演示型视频（browser / vision agent 等）更值得保留。
- 遇到“知道是什么，但不知道为什么”的地方，再补机制和数学。
- 不死记模型特定事实；不确定时查对应 Model Card / 官方文档。

官方课程默认读者已有一定 Python 和 introductory deep learning 基础，因此 `batch`、`logits`、`cross-entropy`、`hidden state`、`head`、`optimizer` 等词可能不会被重新解释。

---

## 2. 常见基础词

### Batch
一次并行送进模型的一组样本。`batch_size = 32` 表示每次处理 32 个样本。不同长度的序列常通过 padding 整成相同长度。

### Token ID、Embedding、Hidden State

```text
文字
→ tokenizer
→ token ID（离散索引）
→ embedding lookup
→ token embedding（连续向量）
→ Transformer layers
→ hidden state（结合上下文后的连续向量）
```

### Logit
模型对每个候选类别 / token 给出的**原始分数**，还不是概率。Softmax 之后才得到概率分布。

例如：

```text
cat  logit = 4.0
dog  logit = 3.0
car  logit = 1.0
```

logit 的绝对值本身通常没有直接概率含义，主要看相对大小。

### Label
训练时的正确答案。做 next-token prediction 时，当前位置的 label 是下一个真实 token 的 token ID。

### Head
接在模型主体后、面向具体任务的输出模块。

```text
hidden state
→ LM head / classification head / QA head
→ logits
```

---

## 3. Text Classification vs Token Classification

- **Text Classification**：整段文本输出一个 label，如情感、主题、垃圾邮件、意图分类。
- **Token Classification**：每个 token 位置输出一个 label，如 NER、POS tagging。

Token Classification 常遇到 word-level 标注和 subword tokenizer 不一致的问题。一个词被拆成多个 subtoken 后，可以只让第一个 subtoken 参与 supervision，其余位置设为 `-100`，从 loss 中忽略。

这些 subtoken 并没有被删除：它们仍参与 embedding、attention 和 hidden-state 计算，只是不计算对应位置的训练误差。

---

## 4. BERT / MLM

**MLM = Masked Language Modeling**。

```text
The [MASK] sat on the mat.
→ BERT encoder
→ [MASK] 位置的 hidden state
→ MLM head
→ vocabulary logits
→ label = cat
```

BERT 是典型 encoder-only 模型；每个 token 可以利用左右两侧上下文。

---

## 5. BART 与 T5

### BART
BART 是 encoder-decoder Transformer。预训练时先破坏输入，再要求 decoder 重建完整原文。

```text
原句：The cat sat on the mat.
破坏：The <mask> on the mat.
目标：The cat sat on the mat.
```

BART 的 text infilling 可以让一个 `<mask>` 代表整段缺失 span，因此模型还要学会缺失内容有多长。

### T5
T5（Text-to-Text Transfer Transformer）也用 encoder-decoder，但把各种 NLP 任务统一成 text-to-text。

```text
input:  The <extra_id_0> on the mat because <extra_id_1>.
target: <extra_id_0> cat sat <extra_id_1> it was tired <extra_id_2>
```

T5 的 span corruption 只让 decoder 生成缺失部分，target 通常比 BART 的“重建整句”更短；这不等于 T5 在所有训练场景都一定更高效。

---

## 6. Encoder-only / Decoder-only / Encoder–Decoder

- **Encoder-only**：擅长把输入变成表示，适合理解、分类、token-level 任务。
- **Decoder-only**：根据已有上下文自回归生成，GPT / Llama / DeepSeek 等通用 LLM 多采用这条路线。
- **Encoder-decoder**：输入序列 A → 输出序列 B，适合 translation、summarization 等 seq2seq 任务。

`seq2seq` 是任务形式；encoder-decoder 是最经典、最自然的实现方式，但 decoder-only 也能做很多 seq2seq 任务。

---

## 7. Self-Attention：Q / K / V

为了避免 token 数和 hidden dimension 混淆，统一用 **3 tokens × 2 dimensions** 的例子。

设：

```text
X = [
  [1, 0],  # token A
  [0, 1],  # token B
  [1, 1]   # token C
]
```

这一层内部有三个可学习矩阵：

```text
W_Q, W_K, W_V
```

当前输入经过它们后得到：

```text
Q = X @ W_Q
K = X @ W_K
V = X @ W_V
```

要区分：

- `W_Q / W_K / W_V`：模型参数，会被训练并保存。
- `Q / K / V`：当前输入临时计算得到的 activations。

核心公式：

```text
Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V
```

含义：

1. `QK^T`：每个 token 的 Query 与所有 token 的 Key 做点积。
2. `/ sqrt(d_k)`：控制 score 尺度。
3. `softmax`：把每一行转成非负、和为 1 的 attention weights。
4. `×V`：按这些权重混合 Value。

如果 token A 的 attention weights 是 `[0.6, 0.3, 0.1]`：

```text
O_A = 0.6 V_A + 0.3 V_B + 0.1 V_C
```

这里 `O_A` 是 attention output / context vector，还不是整个 Transformer block 最终的 hidden state。

---

## 8. Softmax

Softmax：

```text
softmax(x_i) = exp(x_i) / sum_j exp(x_j)
```

例如：

```text
[0.707, 0]
→ [exp(0.707), exp(0)]
≈ [2.028, 1]
→ 除以 3.028
≈ [0.67, 0.33]
```

Softmax 把任意实数 scores 转成非负、和为 1 的权重，而且保持可微。

---

## 9. Multi-Head Attention 与 W_O

完整 Multi-Head Attention 在多个 head 输出 concat 后，还会经过一个可学习的 output projection：

```text
head_1, head_2, ...
→ concat
→ W_O
→ attention sublayer output
```

因此 attention 分支的最终尺度不仅由 `V` 决定，`W_O` 也能改变输出方向和大小。

---

## 10. Residual Connection / Pre-Norm / MLP

经典 residual：

```text
y = x + F(x)
```

直连分支本身没有参数。

现代 Pre-Norm Transformer 常写成：

```text
x = x + Attention(Norm(x))
x = x + MLP(Norm(x))
```

Pre-Norm 不会把 residual 和新信息分支的相对贡献“抹掉”。Normalization 发生在新分支输入端；Attention / MLP 后面仍有大量可学习权重。

### MLP

**MLP = Multilayer Perceptron**，在 Transformer 中可先理解为：

```text
Linear → GELU/SiLU → Linear
```

一个好记的分工：

- Attention：token 之间交换信息。
- MLP：每个 token 对已经拿到的信息继续加工。

---

## 11. Transformer 是否 recurrent？

标准 Transformer 不是 recurrent architecture。

RNN：

```text
h_t = f(h_{t-1}, x_t)
```

标准 self-attention 可以同时处理整段已知序列。Decoder-only 模型生成文本时仍然是 autoregressive / sequential，但这是生成过程的顺序性，不等于内部架构是 RNN。

---

## 12. Positional Information

Q / K / V 的 sequence dimension 保留 token 顺序：第 1 行对应第 1 个 token，第 2 行对应第 2 个 token。

但单靠 QKV 并不知道“第几个位置”意味着什么，因此还需要 position embedding、RoPE 等 positional information。

### Axial positional encoding

可以像“高位 + 低位”来理解。例如 `l1 = 10`、`j = 37`：

```text
j % 10 = 7
j // 10 = 3
```

分别查两个较小 embedding table，再 concat，从而减少超长序列 positional embedding 的参数与显存占用。

---

## 13. ViT

ViT 把图片切成 patch，再把 patch 当作 token。

```text
224×224 RGB image
→ Conv2D(kernel=16, stride=16)
→ 14×14 = 196 patches
→ patch embeddings
→ + learnable [CLS]
→ + position embeddings
→ Transformer encoder
→ 取 CLS final hidden state
→ MLP classification head
→ class logits
```

`[CLS]` 是特殊 learnable token，不是 Transformer 的一层。它经过多层 self-attention 后可以聚合整张图 / 整段文本的信息。

`Conv2D` 就是 CNN 里的二维卷积层；在 ViT patch embedding 中常用来同时完成 patch 切分和线性 projection。

---

## 14. Fine-tuning 与权重更新

Hugging Face `Trainer` 背后仍是标准训练循环：

```python
outputs = model(**batch)
loss = outputs.loss
loss.backward()
optimizer.step()
optimizer.zero_grad()
```

真正改变参数数值的是 `optimizer.step()`。

训练后：

```python
weights = model.state_dict()
model.save_pretrained("./my_finetuned_model")
```

普通聊天 inference 一般不会每轮执行 `backward()` / `optimizer.step()`，所以不会因为一轮对话即时修改 `W_Q/W_K/W_V/...`。

---

## 15. KV Cache

Decoder-only 模型生成新 token 时，历史 token 的 K / V 已经算过，可以缓存下来：

```text
历史 token
→ cached K / V
新 token
→ 只计算新的 Q/K/V
→ 新 Q attention 到历史 K/V
```

这就是 KV cache。它和服务层面的 prompt cache / cache hit 有关系，但不是完全同义。

---

## 16. 长上下文：Local / LSH Attention

### Local Attention
只看附近窗口；多层叠加后 receptive field 会逐渐扩大。

### LSH Attention
**LSH = Locality-Sensitive Hashing（局部敏感哈希）**。

相似向量有较高概率被 hash 到同一桶，只对候选邻居做 attention，减少 full attention 的两两比较。

可以记：

```text
Local attention：按位置近
LSH attention：按表示空间里近
```

---

## 17. Model Routing vs MoE vs MTP

- **Model routing / cascade**：系统层，根据任务难度、成本、领域，把整条请求交给不同模型。
- **MoE routing**：模型内部，通常按 token 选择少数 expert 子网络。
- **MTP (Multi-Token Prediction)**：让模型辅助预测更远的未来 token，与“请求分给哪个模型”不是一回事。

---

## 18. Hugging Face 实践速记

### Hub
先明确 task，再选 model。看 Model Card：任务、输入输出、license、模型大小、tokenizer、限制。

### Inference API
适合快速试模型、比较候选、确认输入输出格式；不等于本地部署或大规模训练。

### Task → model intuition

```text
整段文本 → 一个 label        → Text Classification
每个 token → 一个 label      → Token Classification
context 中找答案 span        → Extractive QA
输入序列 → 输出序列          → seq2seq / encoder-decoder
根据前文继续生成             → decoder-only causal LM
```

---

## 19. 课程中的几个实际坑

### smolagents 需要 Python 3.10+

```bash
conda create -n hf-agents python=3.11 -y
conda activate hf-agents
python -m pip install --upgrade pip
python -m pip install 'smolagents[litellm]'
```

### Ollama 是可选本地 fallback
它是本地模型服务，不会因为你在哪个项目目录执行 `ollama pull` 就把模型下载到那个目录。

### 旧 `question-answering` pipeline 可能和新版 Transformers 不兼容
遇到 API 错误先查 migration guide，不要立刻怀疑自己的 Python 环境。

---

## 20. Chapter 1 阅读体验

Chapter 1 的定位更像 Transformer / inference 概念导览，各小节深浅不均。部分内容只说“是什么”，没有继续解释“为什么”。`Bias and Limitations` 也非常简短，并没有系统覆盖 hallucination、factuality、calibration、prompt sensitivity 等现代 LLM limitation。

目前没有证据显示这些深度内容被系统地“藏进视频”里。比较合适的学习方式仍是：正文主线 + 按需补机制。

---

## 21. 为什么 hallucination 不等于“随机乱说”

语言模型直接优化的是在当前上下文下预测高概率 token，并没有一个内置步骤强制检查“这句话是否符合外部现实”。因此回答可以语言流畅、token probability 很高，同时事实错误。

---

## 22. Temperature

Temperature 调整 logits 的尺度：

```text
p_i = softmax(logit_i / T)
```

- `T < 1`：分布更尖，更确定。
- `T > 1`：分布更平，更多样。
- `T = 1`：不额外改变相对尺度。

---

## 23. 公开笔记定位

这份材料适合看成一份 **FAQ + misconception dataset**：记录真实学习者会卡住的地方，例如：

- `logits` 与 token ID 的区别
- `W_Q/K/V` 与 `Q/K/V` 的区别
- `AV` 是否已经是 hidden state
- `W_O` 从哪里来
- Pre-Norm 是否会抹掉 residual 幅度差异
- decoder-only 为什么仍能利用完整 prompt

它既适合学习者直接阅读，也适合给 Agent / RAG 当补充上下文，用来生成更少“默认你已经懂了”的教程。

---


## 24. Chapter 2：Tensor、矩阵乘法与 ModelOutput

对应课程：[LLM Course Chapter 2 / 2](https://huggingface.co/learn/llm-course/chapter2/2)

### ML 语境里的 tensor

在 PyTorch / Transformers 里，可以先把 **tensor** 理解成任意维度的数值数组：

- 0D：scalar
- 1D：vector
- 2D：matrix
- 3D 及以上：higher-dimensional array

Transformer 中常见：

```text
[batch, sequence_length, hidden_size]
```

PyTorch tensor 和 NumPy array 很像，但还带有 ML 常用能力，例如 device、dtype 和 autograd。

### `return_tensors="pt"`

```python
inputs = tokenizer(
    raw_inputs,
    padding=True,
    truncation=True,
    return_tensors="pt",
)
```

`pt` 表示 **PyTorch**。tokenizer 会把 `input_ids`、`attention_mask` 等结果直接包装成 PyTorch tensors，随后可以直接送进 PyTorch model：

```python
outputs = model(**inputs)
```

### 数学上的 tensor 为什么更严格

严格数学语境里，tensor 不只是“多维数组”。它代表一个与坐标系无关的对象；换 basis 后，components 必须按照确定的 transformation law 变化。

vector 是最简单的类比：`[2, 3]` 是某根几何箭头在一个 basis 下的 coordinates。换 basis 后数字会变，但 vector 本身没有变。

因此，数组是 tensor 的一种 coordinate representation，不等于 tensor 本身。

### 矩阵乘法：可以换括号，不能换顺序

矩阵乘法满足：

```text
(AB)C = A(BC)
```

所以 `x^T T y` 可以先算 `Ty`，也可以先算 `x^T T`。

但一般：

```text
AB != BA
```

例如：

```text
x^T : 1×2
T   : 2×3
y   : 3×1
```

右边先算：

```text
Ty:        (2×3)(3×1) -> 2×1
x^T(Ty):  (1×2)(2×1) -> 1×1
```

左边先算：

```text
x^T T:    (1×2)(2×3) -> 1×3
(x^T T)y: (1×3)(3×1) -> 1×1
```

### dot product 为什么是 bilinear

定义：

```text
F(x, y) = x · y
```

它有两个输入槽。固定其中一个后，另一个都满足线性规则：

```text
F(a x1 + b x2, y) = a F(x1, y) + b F(x2, y)
F(x, a y1 + b y2) = a F(x, y1) + b F(x, y2)
```

因此叫 **bi-linear**：两个输入槽分别都是 linear。

### 为什么两个输入槽会带来两个坐标变换

若：

```text
x = A x'
y = B y'
F(x,y) = x^T T y
```

代入：

```text
F = (A x')^T T (B y')
  = x'^T A^T T B y'
```

所以新的 component matrix 是：

```text
T' = A^T T B
```

两个变换分别来自两个输入槽。只有当两边恰好使用同一个 basis change 时，才写成 `A^T T A`。

### vector 和 covector

一个实用直觉：

- **vector**：方向和大小组成的“箭头”；
- **covector**：输入一个 vector、输出一个 scalar 的线性测量器。

例如：

```text
v = [3, 4]^T
ω(v) = 2 v1 - v2
ω = [2, -1]
ω(v) = 2
```

换坐标时，vector 和 covector 的 components 使用不同 transformation rules，并互相补偿，使 `ω(v)` 这个实际测量结果不变。上标 / 下标用于标记这种不同的变换行为。

### `namedtuple` 与 Hugging Face model outputs

普通 tuple 只能按位置访问：

```python
x = ("cat", 0.97)
x[0]
x[1]
```

Python 的 `namedtuple` 给每个位置加上字段名：

```python
from collections import namedtuple

Result = namedtuple("Result", ["label", "score"])
r = Result("cat", 0.97)

r[0]       # "cat"
r.label    # "cat"
r.score    # 0.97
```

Hugging Face model outputs 提供类似 namedtuple / dictionary 的便利接口，因此常见三种访问方式：

```python
outputs.last_hidden_state
outputs["last_hidden_state"]
outputs[0]
```

实际代码里优先使用 attribute 或 key，通常比 `outputs[0]` 更清楚，也不依赖字段顺序。

---


## 25. Softmax、logit 与 0–1 参数重参数化

Softmax 把一组任意实数分数变成非负、总和为 1 的权重：

```text
logits
→ softmax
→ probability distribution
```

例如：

```text
[2, 1, 0]
→ softmax
≈ [0.67, 0.24, 0.09]
```

深度学习里的 `logits` 通常指 softmax / sigmoid 前的原始预测分数。统计学里的严格 logit 是：

```text
logit(p) = log(p / (1-p))
```

其中 `p` 是概率，`p/(1-p)` 是 odds；inverse-logit 就是 sigmoid：

```text
p = 1 / (1 + exp(-z))
```

若模型参数必须满足 `0 < p < 1`，可以优化无界参数 `z`，再令 `p = sigmoid(z)`。这样优化器不必直接处理 0–1 边界。Bayesian fitting 需要额外注意 induced prior / Jacobian：给 `z` 设 prior 一般不等于直接给 `p` 设同样意义的 prior。

---

## 26. Padding、truncation 与 batch

### Padding

`padding=True` 默认把同一个 batch 里的独立样本 pad 到当前 batch 中最长的 sequence：

```text
A: 5 tokens  -> 8
B: 8 tokens  -> 8
C: 6 tokens  -> 8
```

于是可以形成矩形 tensor：

```text
input_ids.shape = [3, 8]
```

padding positions 的 attention mask 通常为 0。

### Truncation

`truncation=True` 会删除超过 `max_length` / model maximum 的 token。被删除内容不会变成特殊“截断 token”。

对于 sequence pair，默认 `True` / `"longest_first"` 会逐 token 从较长的一侧删除；也可指定：

```python
truncation="only_first"
truncation="only_second"
```

若需要保留超长内容，可用 chunking / sliding window，并结合 `return_overflowing_tokens=True` 与 `stride`。

### Batch item vs. sentence pair

```python
tokenizer(["sentence A", "sentence B"], padding=True)
```

表示两个独立 batch items。它们分别受 sequence-length 限制；长度不会相加，self-attention 也不会跨 batch item。

```python
tokenizer(sentence_a, sentence_b)
```

则是一个 sequence pair。A、B 与 special tokens 共同占用一个 sequence-length budget，在允许跨段 self-attention 的 encoder 中可以互相影响。

可记：

```text
sequence dimension 内：token 可互相 attention
batch dimension 之间：独立样本不互相 attention
```

参考：[Hugging Face Padding and truncation](https://huggingface.co/docs/transformers/main/pad_truncation)

---

## 27. 中文在 GPT 类 tokenizer 中怎么切

OpenAI 的开源 `tiktoken` 是 BPE tokenizer。当前公开 model mapping 中，GPT-4o、o-series 和 GPT-5 family 前缀使用 `o200k_base`。

参考：[OpenAI tiktoken](https://github.com/openai/tiktoken) · [model mapping](https://github.com/openai/tiktoken/blob/main/tiktoken/model.py)

中文通常不会按语言学意义上的偏旁、部首或拼音拆分。Byte-level BPE 更接近下面这个逻辑：

- 高频单字可能成为一个 token；
- 高频双字 / 多字字符串也可能被 merge 成一个 token；
- 生僻字符或罕见组合可能拆成更细的 byte pieces；
- 切法由 vocabulary 和 BPE merge table 决定，而不是由“词根 / 偏旁规则”直接决定。

因此：

```text
清 / 情 / 晴 / 请 / 睛
```

不会因为都包含“青”就必然共享一个 `青` subtoken；

```text
在 / 再
```

也不会因为同音就自动共享 token。

偏旁、读音、同音字和语义之间的关系主要在后续 embedding + Transformer training 中从上下文、字词共现、拼音/语音描述等数据里学习。若模型同时训练 audio modality，还可以从声音信号获得更直接的 phonetic information。

更大的、多语言优化的 vocabulary 往往能让常见中文字符串用更少 token 表示，提高 context / cost efficiency；这不代表 tokenizer 已经显式理解汉字构形或语音学。

---

## 28. 一个模型为什么能同时处理多种语言

现代多语言 LLM 可以粗略理解为：

```text
shared tokenizer
+ shared embedding / Transformer parameters
+ multilingual training data
```

不同语言先映射到同一个 token vocabulary。语义相近的词不需要共享 token；例如 `cat` 和“猫”可以是完全不同的 token IDs。

所有 token 随后进入同一个 embedding table 和同一套 Transformer layers。模型不会为每种语言准备一套完全独立的网络。

在多语言训练中，共享参数同时处理不同语言。平行语料、翻译样本、跨语言 QA / instructions 会提供直接的跨语言对应信号；大量普通多语言文本也能让 contextual representations 形成一定程度的语义对齐和 cross-lingual transfer。

目标语言可以由 prompt / context 指定。较早的 multilingual NMT 还常用显式 target-language token，例如在输入前告诉模型“输出法语”。

最终生成机制仍然相同：

```text
hidden state
→ shared vocabulary logits
→ next token
```

需要注意：OpenAI 公开了 `tiktoken` 等 tokenizer 工具，但 GPT 系列具体训练语料比例、跨语言 sampling 和专门的对齐目标并未完整公开，因此不应把这些未公开细节当作事实。

---

## 29. 机器翻译：从规则到统计、神经网络和 Transformer

### Rule-Based Machine Translation

早期机器翻译大量依赖双语词典、词法/句法分析和人工编写的语法、词序与形态变化规则：

```text
source
→ dictionary + grammar rules
→ structural transfer
→ target
```

优点是可解释；缺点是规则维护成本高，对歧义、习语和新表达非常脆弱。

### Statistical / Phrase-Based Machine Translation

Google 官方回顾显示，2006 年上线的 Google Translate 使用 statistical machine translation，并长期以 Phrase-Based Machine Translation 为核心。

系统从双语平行语料中估计短语翻译概率，再结合：

- translation model
- target-language model
- reordering model
- decoder / search

在大量候选中选择综合得分更高的翻译。它并不是逐字查字典，而是可以翻译短语、调整语序并用统计语言模型判断目标句是否自然。

参考：[Google Research — GNMT at production scale](https://research.google/blog/a-neural-network-for-machine-translation-at-production-scale/)

### Neural Machine Translation

2014 年前后，seq2seq NMT 开始成熟：

```text
source sentence
→ encoder RNN / LSTM
→ representation
→ decoder RNN / LSTM
→ target sentence
```

attention 进一步允许 decoder 在生成不同目标词时回看 source sequence 的不同位置。

Google 在 2016 年开始把 Google Translate 大规模切换到 GNMT。Google 当时明确对比：旧 PBMT 更像处理词和短语，NMT 则把整句作为整体做端到端翻译。

参考：[Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215) · [Google GNMT](https://research.google/blog/a-neural-network-for-machine-translation-at-production-scale/)

### Multilingual NMT 和 zero-shot translation

Google 2016 年展示了一个模型服务多个语言方向，并通过 target-language token 指定输出语言。共享参数带来 cross-lingual transfer，还出现了 zero-shot translation：某些语言对没有直接训练样本，模型仍能进行一定程度的直接翻译。

参考：[Google Research — Zero-Shot Translation](https://research.google/blog/zero-shot-translation-with-googles-multilingual-neural-machine-translation-system/)

### Transformer

2017 年的 `Attention Is All You Need` 首先在机器翻译任务上证明 Transformer 的优势：去掉 RNN recurrence，用 attention-based encoder-decoder 提高建模能力和训练并行性。

历史链条可以简记为：

```text
Rule-based MT
→ Statistical / Phrase-Based MT
→ RNN/LSTM seq2seq + attention
→ Transformer NMT
→ multilingual pretrained models / LLMs
```

参考：[Attention Is All You Need](https://arxiv.org/abs/1706.03762)

---

## 参考

- [Hugging Face LLM Course](https://huggingface.co/learn/llm-course/)
- [Hugging Face Agents Course](https://huggingface.co/learn/agents-course/)
- [Transformers documentation](https://huggingface.co/docs/transformers/)
- [smolagents documentation](https://huggingface.co/docs/smolagents/)

> 本仓库是非官方学习者笔记，不代表 Hugging Face 官方立场，也不提供考试答案库。

# HF LLM & Agents Companion

Independent bilingual companion notes for the Hugging Face LLM Course and Agents Course.

Hugging Face LLM / Agents 课程的中英双语学习伴读。

The notes collect the questions that tend to appear between the lines of a course: missing prerequisites, mechanisms behind short definitions, common misconceptions, small numerical examples, model-selection intuition, and practical troubleshooting.

这份笔记整理课程文字之间容易漏掉的一层：默认前置知识、简短定义背后的机制、真实学习疑问、常见误解、小型数值例子、模型选择直觉和实践踩坑。

## Read / 阅读

**[Bilingual reading site / 中英双语完整阅读页](https://samzebrado.github.io/hf-llm-agents-companion/)**

The Pages site renders the full notes directly and switches Chinese / English in place. GitHub Markdown remains available as the source format:

网页会直接排版完整笔记，并在同一页面切换中文 / English。Markdown 文件保留为源码和便于检索的版本：

- [中文 Markdown](notes/zh-CN.md)
- [English Markdown](notes/en.md)

## Why this exists / 为什么做这份笔记

It is useful in two ways: read it yourself, or give the page/repository to your own AI or agent as prepared context. The explanations, misconceptions, and worked examples are already organized, so a smaller or cheaper model can retrieve the relevant background before answering a follow-up question instead of reconstructing everything from scratch.

它既可以直接给人读，也可以交给自己的 AI / Agent 作为整理好的背景资料。遇到课程概念时，模型可以先从这里检索已有解释、误区和数值例子，再回答追问；这样较小、较便宜的模型也更容易给出靠谱的伴读回答。

## Scope / 内容范围

- Transformer basics: hidden states, logits, heads, batch, labels
- Attention: Q/K/V, softmax, causal mask, KV cache, `W_O`
- Residual connections, Pre-Norm, MLP / FFN
- Encoder-only, decoder-only, encoder-decoder, seq2seq
- BERT, BART, T5, ViT and task-specific heads
- Token vs. text classification and subtoken label alignment
- Long-context ideas: local attention, LSH attention, axial positional encoding
- Fine-tuning, weight updates, inference vs. learning
- Model routing vs. MoE vs. MTP
- Hugging Face course/API/version pitfalls and study strategy

## Editorial scope / 编辑范围

The official Hugging Face materials remain the primary course reference. This repository concentrates on learner-side explanations and links back to the relevant course or documentation where useful. Quiz answers are intentionally left out so the notes stay useful as learning material rather than an answer sheet.

Hugging Face 官方课程仍是主线资料；这里集中补学习者视角的解释，并在适合的位置回链课程和官方文档。公开内容保留学习过程和概念解释，不整理成 quiz 答案表。

## Privacy / 隐私

The public version is edited from private study notes before publishing. Personal project details, schedules, account information, local paths, and identifying context are removed.

公开版从私人学习记录中整理后再发布；个人项目细节、日程、账号信息、本地路径和可识别上下文会在同步前移除。

## Sources / 来源

Primary materials:

- [Hugging Face LLM Course](https://huggingface.co/learn/llm-course/)
- [Hugging Face Agents Course](https://huggingface.co/learn/agents-course/)
- [Transformers documentation](https://huggingface.co/docs/transformers/)
- [smolagents documentation](https://huggingface.co/docs/smolagents/)

Course links and third-party references remain with their respective authors. The companion notes are independent learning commentary.

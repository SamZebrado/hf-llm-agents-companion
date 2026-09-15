# HF LLM & Agents Companion

Unofficial bilingual companion notes for the Hugging Face LLM Course and Agents Course.

Hugging Face LLM / Agents 课程的非官方中英双语学习者补充笔记。

These notes focus on the places where a learner can understand every word in the official course and still wonder, “but why?” — missing prerequisites, common misconceptions, small numerical examples, model-selection intuition, and practical troubleshooting.

这份笔记重点补官方课程里容易“一句话带过”的部分：默认前置知识、真实学习疑问、常见误解、小型数值例子、模型选择直觉，以及实践踩坑。

## Read / 阅读

- [Bilingual GitHub Pages / 中英双语网页](https://samzebrado.github.io/hf-llm-agents-companion/)
- [中文完整笔记](notes/zh-CN.md)
- [English notes](notes/en.md)
- GitHub Pages source: [`docs/`](docs/)

> GitHub Pages uses the `main` branch `/docs` directory. If the site is not live yet, enable it in **Settings → Pages → Deploy from a branch → main → /docs**.
>
> GitHub Pages 使用 `main` 分支的 `/docs` 目录；如果网页尚未上线，可在 **Settings → Pages → Deploy from a branch → main → /docs** 中启用。

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

## Positioning / 定位

This repository is **not** an official Hugging Face resource, a translation of the course, or an answer bank. It is a learner-oriented companion that links back to the official materials and explains gaps that commonly cause confusion.

本仓库**不是** Hugging Face 官方资料、逐段翻译或考试答案库。它以学习者视角补充官方材料，并尽量回链到原课程与官方文档。

## Privacy / 隐私

The public version intentionally excludes private project details, personal schedules, account information, local paths, and other identifying context from the original study notes.

公开版已主动移除私人项目细节、个人日程、账号信息、本地路径及其他不适合公开的上下文。

## Sources / 来源

Primary materials:

- [Hugging Face LLM Course](https://huggingface.co/learn/llm-course/)
- [Hugging Face Agents Course](https://huggingface.co/learn/agents-course/)
- [Transformers documentation](https://huggingface.co/docs/transformers/)
- [smolagents documentation](https://huggingface.co/docs/smolagents/)

Course links and third-party references remain the property of their respective authors. These notes are independent commentary and learning material.

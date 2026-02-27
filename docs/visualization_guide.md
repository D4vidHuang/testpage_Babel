# EMSE-Babel Visualization Portal — Guide / 可视化门户指南

[English](#english) | [中文](#chinese)

---

## ⚡ English Guide {#english}

The EMSE-Babel Visualization Portal is the central interface for analyzing our study's results.

### 1. How to Open

Run this in your terminal from the project root:

```bash
open docs/index.html
```

### 2. Dashboard Sections

- **LLM-as-a-Judge**: Evaluates models acting as judges.
  - **Accuracy (κ)**: Correlation with human experts.
  - **Confusion Matrices**: Detailed 3-class (Correct/Partial/Incorrect) breakdown.
  - **Hallucinations**: Models generating non-standard error codes.
- **Neural Metrics**: Performance of metrics like BARTScore and BERTScore.
  - **Human Alignment**: Correlation between neural scores and expert judgment.
  - **Noise Detection**: Comparison between real and synthetic (noisy) comments.

---

## ⚡ 中文指南 {#chinese}

EMSE-Babel 可视化门户是查看研究结果的中央入口。

### 1. 如何打开

在项目根目录下，于终端运行：

```bash
open docs/index.html
```

### 2. 看板主要板块

- **LLM-as-a-Judge (裁判模型评测)**: 衡量 LLM 扮演专家角色时的表现。
  - **准确率 (κ)**: 模型评分与人类专家的一致性。
  - **混淆矩阵**: 三类标签（正确/部分正确/错误）的详细分布。
  - **幻觉率**: 模型生成非官方分类系统中错误代码的频率。
- **Neural Metrics (神经评价指标)**: 评估 BARTScore 和 BERTScore 的表现。
  - **专家对齐**: 神经指标评分与人类判断的相关度。
  - **噪声检测**: 衡量模型区分真实注释与随机/目标噪声注释的能力。

---
*Note: This portal is powered by Plotly & Bokeh. / 备注：本门户由 Plotly 和 Bokeh 提供可视化支持。*

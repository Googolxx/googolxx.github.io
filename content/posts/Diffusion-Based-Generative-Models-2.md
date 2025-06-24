+++
date = '2025-06-24T20:08:36+08:00'
draft = true
title = 'Diffusion-Based Generative Models <2>: DDIM'
tags = ["diffusion-models", "deep-learning", "generative-AI"]
categories = ["Generative Models"]
mermaid = true
+++

## 一. 引言

上一篇内容讲了DDPM的算法框架，看起来一切都很完美，但采样速度还是太慢了，如果设置 $ T=1000$, 那采样的代价还是太大了。因此迎来了DDIM (Denoising Diffusion Implicit Models)。对于DDIM，我觉得还是可以从 DDPM和 SDE/ODE 两个角度去分析的。

### 1.1 DDPM视角下的DDIM

#### 核心思想
- **DDPM** 是一个基于马尔可夫链的扩散模型，通过逐步加噪（前向过程）和逐步去噪（反向过程）学习数据分布。
- **DDIM** 是 DDPM 的 **非马尔可夫推广**，它重新参数化了反向过程，允许 **跳过中间步骤**，从而加速采样。

#### 非马尔可夫性
- **DDPM**：前向和反向过程都是马尔可夫的（下一步仅依赖当前步）。
- **DDIM**：通过设计非马尔可夫的逆过程，打破了这一限制，允许更灵活的生成路径（如跳步采样）。

#### 确定性生成
- **DDPM**：反向过程是随机的（每一步注入高斯噪声）。
- **DDIM**：可以通过设定噪声方差为0，实现 **确定性生成**（类似ODE），从而生成结果可重复。

#### 采样加速
- DDIM 通过重新参数化，将 DDPM 的 $T$ 步采样压缩到 $S$ 步（$S \ll T$），而保持相似的生成质量。

#### 数学形式
DDIM 的逆过程改写为：
$$
x_{t-1} = \sqrt{\alpha_{t-1}} \left( \frac{x_t - \sqrt{1-\alpha_t} \epsilon_\theta(x_t, t)}{\sqrt{\alpha_t}} \right) + \sqrt{1-\alpha_{t-1}} \epsilon_\theta(x_t, t)
$$
其中 $\alpha_t$ 是噪声调度，$\epsilon_\theta$ 是去噪网络。当噪声项系数为0时，生成过程变为确定性。

---

### 1.2. SDE/ODE视角下的DDIM

#### 核心思想
扩散模型可以统一描述为 **随机微分方程（SDE）** 或 **常微分方程（ODE）** 的离散化：
- **SDE**：前向过程是加噪的随机过程，反向过程对应一个逆时间的SDE。
- **ODE**：通过Fokker-Planck方程可证明，任何逆向SDE均存在一个确定性ODE，其解与SDE共享相同的边缘概率分布 $ p_t(x) $。 忽略SDE中的随机噪声项，即可导出确定性生成路径（概率流ODE），适合快速采样和确定性生成。

概率流ODE的连续形式：

$$
dx = \left[ f(x, t) - \frac{1}{2}g(t)^2 \nabla_x \log p_t(x) \right] dt
$$

**DDIM** 是 **ODE离散化的一种特例**，其生成路径对应概率流ODE的数值解法。DDPM和DDIM原文中的采样方法，本质上分别对应 SDE和 ODE的一阶离散化，比如欧拉-丸山法和欧拉法。本篇就不展开讲SDE/ODE视角下的DDIM了，等后面站在 SDE/ODE大一统的视角下去看，一切就都明朗了。


## 二. DDIM算法框架

DDIM算法框架分成两部分讲：
- DDPM的非马尔可夫推广
- 加速采样

### DDPM的非马尔可夫推广

<!-- 插句题外话 -->
回顾DDPM的优化目标 $\mathcal{L}_{VLB}$ :

$$
\mathcal{L}_{VLB} = - \sum_{t=1}^T \frac{1}{2\sigma_{q}^2(t)} \cdot \frac{(1 - \alpha_t)^2 }{\alpha_t(1 - \bar{\alpha}_t)} \mathbb{E}_{q(\mathbf{x_{t}}|\mathbf{x_0})} \left[ \| \epsilon_t - \epsilon_\theta(\mathbf{x_t}, t)\|^2 \right]
$$

DDIM的核心动机，就来源于DDPM的目标函数只依赖于边际分布 $q(\mathbf{x_t}|\mathbf{x_0})$，
而不是联合分布 $q(\mathbf{x_{1:T}}|\mathbf{x_0})$。

再回顾DDPM优化目标推导过程的一个中间表达形式：

$$
\frac{1}{2\sigma_{q}^2(t)} \|  \mu_q(\mathbf{x}_t, \mathbf{x}_0) -  \mu_{\theta}(\mathbf{x}_t, t) \|^2
$$

仔细想想不难发现，其实我们关心的只有 $q(\mathbf{x_t}|\mathbf{x_0})$ 和 $q(\mathbf{x_{t-1}}|\mathbf{x_0}, \mathbf{x_t})$ 的均值 $\mu_q(\mathbf{x}_t, \mathbf{x}_0)$

通过对联合分布进行合适的分解，我们可以在保持DDPM优化目标不变的前提下，完成非马尔可夫链的推广形式：

$$
\begin{aligned}
q(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0) &= \frac{q(\mathbf{x}_t | \mathbf{x}_{t-1}, \mathbf{x}_0) \cdot q(\mathbf{x}_{t-1} | \mathbf{x}_0)}{q(\mathbf{x_{t}} | \mathbf{x}_0)} \\
&= \frac{q(\mathbf{x}_t | \mathbf{x}_{t-1}) \cdot q(\mathbf{x}_{t-1} | \mathbf{x}_0)}{q(\mathbf{x_{t}} | \mathbf{x}_0)}
\end{aligned}
$$


{{< figure src="/pic_diff_2/non-markov-forward.png" title="">}}

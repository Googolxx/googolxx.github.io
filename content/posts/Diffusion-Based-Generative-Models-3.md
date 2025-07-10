+++
date = '2025-07-10T19:28:35+08:00'
draft = true
title = 'Diffusion-Based Generative Models <3>: SMLD'
tags = ["diffusion-models", "deep-learning", "generative-AI"]
categories = ["Generative Models"]
mermaid = true
+++

## 一. 引言

在生成模型的研究中，基于得分（Score-based）的生成方法提供了一种从目标分布生成数据的新颖而强大的框架。与传统的生成方法不同，这类模型通过估计数据分布的梯度信息——即所谓的得分函数（score function），来指导样本生成过程。其核心思想在于利用统计物理学中的朗之万方程（Langevin equation）作为采样工具，将随机噪声逐步演化为符合目标分布的样本。尽管在一维情况下，我们可以通过反累积分布函数（CDF）的方法轻松实现采样，但在高维空间中，这种直接方法不再适用。
此时，基于得分的方法则展现出其独特优势：它通过对概率密度的局部变化进行建模，使得生成过程能够在计算上变得可行，并赋予模型更强的表达能力和灵活性。
本部分将介绍(Denosing) Score Matching，并通过 Langevin Dynamics进行采样的算法思想。

## 二. 背景

### 2.1 一维变量的采样
考虑一种简单情况，$ x \in \mathbb{R}^1 $ 为一维变量，$ x \sim p(x)$，可以通过逆变换采样（Inverse Transform Sampling），从 $ p(x)$进行随机采样，求其累计密度函数 CDF：

$$
F(x) = p(X \leq x)
$$

然后求CDF的逆函数，并从 $[0,1]$ 均匀分布中采样, 代入到CDF的逆函数中，就能实现从分布 $ p(x)$ 中采样：

$$
\begin{aligned}
u &\sim Uniform[0,1] \\
x &= F^{-1}(u)
\end{aligned}
$$

（再回想下VAE中的Reparameterization技巧，从高斯分布 $\mathcal{N}(\mu, \sigma^2)$ 中采样时，是先从标准正态分布 $\mathcal{N}(0, 1)$ 中采样得到 $\mathbf{\epsilon}$，然后通过变换 $x = \mu + \sigma \epsilon$ 得到样本。VAE采用这种技巧的主要目的是解决反向传播时梯度无法通过随机采样节点的问题）

但对于高维变量来说，这种方法是行不通的：
- 在高维空间中，计算累积分布函数（CDF）变得极其困难：CDF变成 $F(x_1, x_2, ..., x_d) = \int_{-\infty}^{x_1} \int_{-\infty}^{x_2} ... \int_{-\infty}^{x_d} p(t_1, t_2, ..., t_d) dt_1 dt_2 ... dt_d$，这个多重积分的计算复杂度随维度呈指数增长。
- 即使能够计算出高维CDF，其逆函数 $F^{-1}(u_1, u_2, ..., u_d)$ 的求解也极其困难，通常是没有解析解的。
- 高维空间中数据分布的稀疏性问题，这是所有生成式模型面临的共同挑战。

### 2.2 Langevin Dynamics

如何对高维空间的变量进行采样呢，有效地采样意味着我们希望从概率密度更高的地方进行采样，如果有 $ p(\mathbf{x})$ 的梯度，那就可以根据梯度前往高密度区域采样。等价地假设有 $ \log p(\mathbf{x})$ 的梯度 $ \nabla_{\mathbf{x}} \log p(\mathbf{x})$，这里使用 $ \nabla_{\mathbf{x}} \log p(\mathbf{x})$ 的原因是$\log p(\mathbf{x})$ 的梯度在数值上更稳定，梯度方向相同，在采样过程中，我们关心的是梯度方向，而不是梯度大小。

为了简化表示，我们将 $\nabla_{\mathbf{x_t}} \log p(\mathbf{x_{t}})$ 记为 $\nabla_{\mathbf{x}} \log p_t(\mathbf{x})$。通过梯度上升方法，可以迭代更新样本位置：

$$
\mathbf{x_{t+1}} = \mathbf{x_{t}} + \tau \nabla_{\mathbf{x}} \log p_t(\mathbf{x})
$$

其中 $\tau > 0$ 是步长参数，控制每次更新的幅度。然而，这种纯梯度上升方法存在一个根本性问题：样本最终会收敛到概率密度的局部最大值点，无法实现对整个目标分布的采样。

为了实现真正的从分布 $p(\mathbf{x})$ 中采样，我们需要引入朗之万动力学（Langevin Dynamics），这是一种基于马尔可夫链蒙特卡罗（MCMC）的采样方法。该方法在梯度上升的基础上添加随机噪声项：

$$
\mathbf{x_{t+1}} = \mathbf{x_{t}} + \tau \nabla_{\mathbf{x}} \log p_t(\mathbf{x}) + \sqrt{2\tau} \mathbf{z}_t, \quad t = 0, 1, \dots, T-1
$$

其中：
- $\tau$ 是步长参数，控制更新步长的大小
- $\mathbf{z}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ 是独立同分布的高斯噪声
- $T$ 是总迭代次数，决定了采样过程的长度
- $\sqrt{2\tau}$ 是噪声的标准差，这个特定选择确保了在 $\tau \to 0$ 时，离散时间过程收敛到连续时间的朗之万扩散过程

**$\tau$ 和 $T$ 的关系**：较小的 $\tau$ 值需要更大的 $T$ 值来确保采样过程充分探索目标分布。通常选择 $\tau$ 使得 $T \cdot \tau$ 保持在一个合理的范围内，以保证采样效率和收敛性。

对比可以发现

$$
Langevin Dynamics = gradient descent/ascent + noise
$$

梯度上升是告诉我们如何找到局部最大值点($p(\mathbf{x}的峰值)$)，而朗之万动力学告诉我们如何从 $p(\mathbf{x})$ 采样


## 三. SMLD算法框架
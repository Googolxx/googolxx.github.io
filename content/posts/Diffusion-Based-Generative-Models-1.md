+++
date = '2025-06-22T13:43:59+08:00'
draft = false
title = 'Diffusion-Based Generative Models <1>: DDPM'
tags = ["diffusion-models", "deep-learning", "generative-AI"]
categories = ["Generative Models"]
mermaid = true
+++

<!-- summary = "深入解读去噪扩散概率模型 (DDPM) 的核心算法与数学推导，揭示其如何通过前向加噪与反向去噪过程实现高质量生成。" -->



## 一. 引言

扩散模型（**Diffusion Models**）作为当前生成式 AI 的核心范式，受热力学启发[^1]，主要思想是迭代地加噪/去噪数据，模拟粒子扩散过程。在图像、视频生成等领域实现了非常好的效果。下文介绍核心代表作之一 **DDPM** [^2] (Denoising Diffusion Probabilistic Models)。

随着 Diffusion-Based Generative Models 理论的逐渐完善，可以从多种视角（分数匹配、 微分方程等）推导出 DDPM 的前向/逆向扩散过程、优化目标和采样过程。这里，我们将遵循 DDPM 原文的思路进行推导。


---

## 二. DDPM 算法框架

### 1. 前向扩散过程（Forward Diffusion Process）

前向扩散过程是***无参***的扩散过程，服从一个马尔可夫链 (Markov Chain)：马尔科夫链为状态空间中经过从一个状态到另一个状态的转换的随机过程，该过程要求具备"无记忆性"，即下一状态的概率分布只能由当前状态决定，在时间序列中它前面的事件均与之无关。

具体来说，从一个真实数据分布采样 $\mathbf{x_0} \sim q(\mathbf{x})$，通过逐步对数据 $\mathbf{x_0}$ 添加高斯噪声（Gaussian Noise），得到被扰动的样本 $\mathbf{x_1},...\mathbf{x_t},...\mathbf{x_T}$，在 $T$ 步后接近纯噪声。得益于高斯分布的特殊数学性质，其线性组合仍然是高斯分布，因此可以将加噪过程中互相独立的高斯噪声进行合并:

$$
\begin{aligned}
\mathbf{x_t} &= \sqrt{\alpha_t} \mathbf{x_{t-1}} + \sqrt{1 - \alpha_t} \epsilon_{t-1} \\
    &= \sqrt{\alpha_t} \left( \sqrt{\alpha_{t-1}} \mathbf{x_{t-2}} + \sqrt{1 - \alpha_{t-1}} \epsilon_{t-2} \right) + \sqrt{1 - \alpha_t} \epsilon_{t-1} \\
    &= \sqrt{\alpha_{t-1} \alpha_t} \mathbf{x_{t-2}} + \underbrace{{\sqrt{\alpha_t} \sqrt{1 - \alpha_{t-1}} \epsilon_{t-2} + \sqrt{1 - \alpha_t} \epsilon_{t-1}}}_{\text{Combine noise using linear Gaussian}} \\
    &= \sqrt{\alpha_{t-1} \alpha_t} \mathbf{x_{t-2}} + \sqrt{1 - \alpha_{t-1} \alpha_t} \bar{\epsilon}_{t-2} \\
    &=  ... \\
    &= \sqrt{\bar{\alpha}_t} \mathbf{x_0} +  \sqrt{1 - \bar{\alpha}_t} \bar{\epsilon}_{t}
\end{aligned}
$$

其中 $ \{ \alpha_0, \dots, \alpha_T \}$ 是一组人为设置的超参数，用于控制扩散过程中噪声的强度。

定义 $$\bar{\alpha}_t = \prod_{i=1}^t \alpha_i$$ 

那么我们可以得到:

$$
\begin{aligned}
q(\mathbf{x_t}|\mathbf{x_{t-1}}) = \mathcal{N}(\mathbf{x_t} | \sqrt{{\alpha_t}} \mathbf{x_{t-1}}, (1 - \alpha_t) \mathbf{I}) \\
q(\mathbf{x_t}|\mathbf{x_0}) = \mathcal{N}(\mathbf{x_t} | \sqrt{\bar{\alpha}_t} \mathbf{x_0}, (1 - \bar{\alpha}_t) \mathbf{I})
\end{aligned}
$$

当扩散过程足够长，可以得到预先假设的先验分布 $ q(\mathbf{x_T}) = \mathcal{N}(\mathbf{x_T} |\mathbf{0}, \mathbf{I})$

Note：
- 加噪过程中设置系数为 $ \sqrt{\alpha_t}$ 和 $(1 - \sqrt{\alpha_t})$ 是使其平方和为 $1$，从而保持扩散过程中方差的稳定。在SMLD中，因为系数设置的不同，为方差膨胀的形式。
- 逆向分布 $q(\mathbf{x_{t-1}}|\mathbf{x_{t}})$ 没有显式解析解，因为 $\mathbf{x_{t-1}}$ 和 $\epsilon_{t}$ 的依赖性使得无法直接利用前向过程的线性高斯性质。但是，当 $1-\alpha_t$ 足够小（即扩散步长极短或总步数 $T$ 足够大）时，$q(\mathbf{x_{t-1}}|\mathbf{x_{t}})$ 可近似为高斯分布，这一近似在扩散过程的连续极限下（如随机微分方程SDE的视角下）有理论支持。

{{< figure src="/pic_diff_1/pipeline.png" title="">}}

### 2. 逆向扩散过程（Reverse Diffusion Process）
前向过程在手动设计下，均有明确的解析解。在假设逆向过程也为马尔可夫链的情况下，如果我们能够得到逆向过程 $q(\mathbf{x_{t-1}}|\mathbf{x_{t}})$ 的形式，那就能够根据联合分布 $q(\mathbf{x_0}, \mathbf{x_1},..., \mathbf{x_T})$，从先验分布 $q(\mathbf{x_T}) = \mathcal{N}(\mathbf{x_T} |\mathbf{0}, \mathbf{I})$ 开始，逐步采样得到 $\mathbf{x_0}$。上文提到 $q(\mathbf{x_{t-1}}|\mathbf{x_{t}})$ 虽然是未知的，但可近似为高斯分布，所以这里用参数化的神经网络学习 $p_{\theta}(\mathbf{x_{t-1}}|\mathbf{x_t})$ 来逼近 $q(\mathbf{x_{t-1}}|\mathbf{x_{t}})$:

$$
p_\theta(\mathbf{x_{t-1}}|\mathbf{x_t}) = \mathcal{N}(\mathbf{x_{t-1}}; \mu_\theta(\mathbf{x_t},t), \Sigma_\theta(\mathbf{x_t},t))
$$

完成训练后得到近似真实逆向分布的 $p_{\theta}(\mathbf{x_{t-1}}|\mathbf{x_t})$，即可通过联合分布来进行采样：

$$
p_{\theta}(\mathbf{x_0},\mathbf{x_1},...,\mathbf{x_T}) = p(\mathbf{x_T}) \prod_{t=1}^{T} p_{\theta}(\mathbf{x_{t-1}}|\mathbf{x_t})
$$

定义了前向和逆向的扩散过程，下一步就是确定优化目标，也就是损失函数。

### 3. 优化目标推导
老规矩，直接最大化似然:

$$
\begin{aligned}
\log p_\theta(\mathbf{x_0}) &= \log \int p_\theta(\mathbf{x_{0:T}}) d\mathbf{x_{1:T}} \quad  \\
&= \log \int p_\theta(\mathbf{x_{0:T}}) \frac{q(\mathbf{x_{1:T}}|\mathbf{x_0})}{q(\mathbf{x_{1:T}}|\mathbf{x_0})} d\mathbf{x_{1:T}}  \\
&= \log \int q(\mathbf{x_{1:T}}|\mathbf{x_0}) \left( \frac{p_\theta(\mathbf{x_{0:T}})}{q(\mathbf{x_{1:T}}|\mathbf{x_0})} \right) d\mathbf{x_{1:T}}  \\
&= \log \mathbb{E}_{q(\mathbf{x_{1:T}}|\mathbf{x_0})} \left[ \frac{p_\theta(\mathbf{x_{0:T}})}{q(\mathbf{x_{1:T}}|\mathbf{x_0})} \right] \\
&\geq \mathbb{E}_{q(\mathbf{x_{1:T}}|\mathbf{x_0})} \left[ \log \frac{p_\theta(\mathbf{x_{0:T}})}{q(\mathbf{x_{1:T}}|\mathbf{x_0})} \right]
\end{aligned}
$$

这里用到了Jensen’s inequality: 对于任意随机变量 $X$ 和任意凹函数 $f$ ，$ f(\mathbb{E}[X]) ≥ \mathbb{E}[f(X)] $ 都成立。这也变分推断（Variational Inference）中的重要概念，从优化最大化似然 ——> 优化变分下界 VLB（Variational Lower Bound）/ ELBO 。

继续往下推导之前，可以想一个问题，如何分解上式中的联合分布 $q(\mathbf{x_{1:T}}|\mathbf{x_0})$ 和 $p_\theta(\mathbf{x_{0:T}})$ ：因为我们假设前向和逆向扩散过程都是一个马尔可夫链，这里肯定是要用到马尔可夫链的性质，对 $p_\theta(\mathbf{x_{0:T}})$ 来说很简单，我们继续把他拆成 $p(\mathbf{x_T}) \prod_{t=1}^{T} p_{\theta}(\mathbf{x_{t-1}}|\mathbf{x_t})$ 即可，但对于 $q(\mathbf{x_{1:T}}|\mathbf{x_0})$ 来说就有点难办了，因为 $q(\mathbf{x_{t-1}}|\mathbf{x_{t}})$ 没有解析解，所以需要想个办法来规避 $q(\mathbf{x_{t-1}}|\mathbf{x_{t}})$。 然后我们就可以发现， 如果给 $q(\mathbf{x_{t-1}}|\mathbf{x_{t}})$ 加上条件 $\mathbf{x_0}$, 得到的 $q(\mathbf{x_{t-1}}|\mathbf{x_{t}}, \mathbf{x_0})$ 是有解析解的：

$$
\begin{aligned}
q(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0) &= \frac{q(\mathbf{x}_t | \mathbf{x}_{t-1}, \mathbf{x}_0) \cdot q(\mathbf{x}_{t-1} | \mathbf{x}_0)}{q(\mathbf{x_{t}} | \mathbf{x}_0)} \\
&= \frac{q(\mathbf{x}_t | \mathbf{x}_{t-1}) \cdot q(\mathbf{x}_{t-1} | \mathbf{x}_0)}{q(\mathbf{x_{t}} | \mathbf{x}_0)}
\end{aligned}
$$

上式中的 $q(\mathbf{x_{t}}|\mathbf{x_{t-1}}) $ , $q(\mathbf{x_{t}}|\mathbf{x_0}) $ 和 $q(\mathbf{x_{t-1}}|\mathbf{x_0}) $ 都是有解析解的高斯分布，由于高斯分布的特性，$q(\mathbf{x_{t-1}}|\mathbf{x_{t}}, \mathbf{x_0})$ 也是高斯分布，对其求解析解：

$$
\begin{equation}
\begin{aligned}
q(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0) &= \frac{q(\mathbf{x}_t | \mathbf{x}_{t-1}) \cdot q(\mathbf{x}_{t-1} | \mathbf{x}_0)}{q(\mathbf{x}_t | \mathbf{x}_0)} \\
&= \frac{
    \mathcal{N}\left(\mathbf{x}_t \mid \sqrt{\alpha_t}\mathbf{x}_{t-1}, (1-\alpha_t)\mathbf{I}\right)
    \cdot \mathcal{N}\left(\mathbf{x}_{t-1} \mid \sqrt{\bar{\alpha}_{t-1}}\mathbf{x}_0, (1-\bar{\alpha}_{t-1})\mathbf{I}\right)
}{
    \mathcal{N}\left(\mathbf{x}_t \mid \sqrt{\bar{\alpha}_t}\mathbf{x}_0, (1-\bar{\alpha}_t)\mathbf{I}\right)
} \\
% &\propto \exp -\frac{1}{2}
&\propto \exp - \frac{1}{2} \left\{\frac{\left(\mathbf{x}_t - \sqrt{\alpha_t} \mathbf{x} \right)^2}{1 - \alpha_t} + \frac{(\mathbf{x}_{t-1} - \sqrt{\bar{\alpha}_{t-1}}\mathbf{x}_0)^2}{1-\bar{\alpha}_{t-1}} - \frac{(\mathbf{x}_t - \sqrt{\bar{\alpha}_t}\mathbf{x}_0)^2}{1-\bar{\alpha}_t}
\right\}
% - \frac{\left(\mathbf{x}_t - \sqrt{\alpha_t} \mathbf{x} \right)}{2(1 - \alpha_t)}
\end{aligned}
\end{equation}
$$

然后可以通过配方法凑出高斯分布的形式，得到 $q(\mathbf{x_{t-1}}|\mathbf{x_{t}}, \mathbf{x_0})$ 的均值和方差：

$$
\begin{aligned}
q(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0) &\propto \exp - \frac{1}{2} \left\{\frac{\left(\mathbf{x}_t - \sqrt{\alpha_t} \mathbf{x} \right)^2}{1 - \alpha_t} + \frac{(\mathbf{x}_{t-1} - \sqrt{\bar{\alpha}_{t-1}}\mathbf{x}_0)^2}{1-\bar{\alpha}_{t-1}} - \frac{(\mathbf{x}_t - \sqrt{\bar{\alpha}_t}\mathbf{x}_0)^2}{1-\bar{\alpha}_t}
\right\}\\
&= \exp - \frac{1}{2} \left\{ ( \frac{\alpha_t}{1-\alpha_t} + \frac{1}{1-\bar{\alpha}_{t-1}}) \mathbf{x}_{t-1}^2 - (\frac{2\sqrt{\alpha_t}}{1-\alpha_t}\mathbf{x}_{t} + \frac{2\sqrt{\bar{\alpha}_{t-1}}}{1-\bar{\alpha}_{t-1}}\mathbf{x}_{0} ) \mathbf{x}_{t-1} + \mathbf{C}(\mathbf{x}_{t}, \mathbf{x}_{0}) \right\}
\end{aligned}
$$

也就能得到：

$$
\begin{aligned}
\mu_q(\mathbf{x}_t, \mathbf{x}_0) &= \frac{(1 - \bar{\alpha}_{t-1})\sqrt{\alpha_t}}{1 - \bar{\alpha}_t} \mathbf{x}_t + \frac{(1 - \alpha_t)\sqrt{\bar{\alpha}_{t-1}}}{1 - \bar{\alpha}_t} \mathbf{x}_0 \\
\Sigma_q(t) &= \frac{(1 - \alpha_t)(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} \mathbf{I}
\end{aligned}
$$

或者对  $ \mathbf{x_{t-1}}$求导得到均值和方差，一阶导数置为 $0$ 得到均值，求二阶导数再取倒数得到方差。得到 $ q(\mathbf{x_{t-1}}|\mathbf{x_{t}}, \mathbf{x_0})$ 服从的高斯分布:

$$
\begin{aligned}
q(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0) = \mathcal{N}\left(\mathbf{x}_{t-1}; \frac{(1 - \bar{\alpha}_{t-1})\sqrt{\alpha_t}}{1 - \bar{\alpha}_t} \mathbf{x}_t + \frac{(1 - \alpha_t)\sqrt{\bar{\alpha}_{t-1}}}{1 - \bar{\alpha}_t} \mathbf{x}_0, \frac{(1 - \alpha_t)(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} \mathbf{I} \right)
\end{aligned}
$$

得到了 $ q(\mathbf{x_{t-1}}|\mathbf{x_{t}}, \mathbf{x_0})$ 的解析解，可以发现均值项其实就是 $\mathbf{x_0} 和 \mathbf{x_t}$ 的线性组合 。接下来就回到优化目标(变分下界VLB)上，将前向和逆向的联合分布，按之前的计划进行分解：

$$
\begin{aligned}
\log p_\theta(\mathbf{x_0}) &\geq \mathbb{E}_{q(\mathbf{x_{1:T}}|\mathbf{x_0})} \left[ \log \frac{p_\theta(\mathbf{x_{0:T}})}{q(\mathbf{x_{1:T}}|\mathbf{x_0})} \right] \\
&= \mathbb{E}_{q(\mathbf{x_{1:T}}|\mathbf{x_0})} \left[ \log \frac{p_\theta(\mathbf{x_T}) p_\theta(\mathbf{x_0}|\mathbf{x_1}) \prod_{t=2}^T p_\theta(\mathbf{x_{t-1}}|\mathbf{x_t})}{q(\mathbf{x_1}|\mathbf{x_0}) \prod_{t=2}^T q(\mathbf{x_t}|\mathbf{x_{t-1}}, \mathbf{x_0})} \right] \\
&= \mathbb{E}_{q(\mathbf{x_{1:T}}|\mathbf{x_0})} \left[ \log \frac{p_\theta(\mathbf{x_T}) p_\theta(\mathbf{x_0}|\mathbf{x_1})}{q(\mathbf{x_1}|\mathbf{x_0})} \right] + \mathbb{E}_{q(\mathbf{x_{1:T}}|\mathbf{x_0})} \left[ \log \prod_{t=2}^T \frac{p_\theta(\mathbf{x_{t-1}}|\mathbf{x_t})}{q(\mathbf{x_t}|\mathbf{x_{t-1}}, \mathbf{x_0})} \right] \\
&= \mathbb{E}_{q(\mathbf{x_{1:T}}|\mathbf{x_0})} \left[ \log \frac{p_\theta(\mathbf{x_T}) p_\theta(\mathbf{x_0}|\mathbf{x_1})}{q(\mathbf{x_1}|\mathbf{x_0})} \right] + \mathbb{E}_{q(\mathbf{x_{1:T}}|\mathbf{x_0})} \left[ \log \prod_{t=2}^T \frac{p_\theta(\mathbf{x_{t-1}}|\mathbf{x_t})}{q(\mathbf{x_{t-1}}|\mathbf{x_t}, \mathbf{x_0})} \cdot \frac{q(\mathbf{x_{t-1}}| \mathbf{x_0})}{q(\mathbf{x_t}| \mathbf{x_0})} \right] \\
&= \mathbb{E}_{q(\mathbf{x_{1:T}}|\mathbf{x_0})} \left[ \log \frac{p_\theta(\mathbf{x_T}) p_\theta(\mathbf{x_0}|\mathbf{x_1})}{q(\mathbf{x_1}|\mathbf{x_0})} \right] + \mathbb{E}_{q(\mathbf{x_{1:T}}|\mathbf{x_0})} \left[\frac{q(\mathbf{x_1}| \mathbf{x_0})}{q(\mathbf{x_T}| \mathbf{x_0})} \right] + \sum_{t=2}^T \mathbb{E}_{q(\mathbf{x_{1:T}}|\mathbf{x_0})} \left[ \log  \frac{p_\theta(\mathbf{x_{t-1}}|\mathbf{x_t})}{q(\mathbf{x_{t-1}}|\mathbf{x_{t}}, \mathbf{x_0})}  \right] \\
&= \underbrace{\mathbb{E}_{q(\mathbf{x_{1}}|\mathbf{x_0})} \left[\log p_\theta(\mathbf{x_0}|\mathbf{x_1})\right]}_{\text{reconstruction}} + \underbrace{\mathbb{E}_{q(\mathbf{x_{T}}|\mathbf{x_0})} \left[\log \frac{p_\theta(\mathbf{x_T})}{q(\mathbf{x_{T}}|\mathbf{x_0})}\right]}_{\text{prior matching}} + \underbrace{\sum_{t=2}^T \mathbb{E}_{q(\mathbf{x_{1:T}}|\mathbf{x_0})}  \left[\log  \frac{p_\theta(\mathbf{x_{t-1}}|\mathbf{x_t})}{q(\mathbf{x_{t-1}}|\mathbf{x_{t}}, \mathbf{x_0})}\right]}_{\text{consistency term}} \\
% &= \mathbb{E}_{q(\mathbf{x_{1}}|\mathbf{x_0})} \log p_\theta(\mathbf{x_0}|\mathbf{x_1}) + \mathbb{E}_{q(\mathbf{x_{T}}|\mathbf{x_0})} \log \frac{p_\theta(\mathbf{x_T})}{q(\mathbf{x_{T}}|\mathbf{x_0})} + \sum_{t=2}^T \mathbb{E}_{q(\mathbf{x_{t}}|\mathbf{x_0})}  \log  \frac{p_\theta(\mathbf{x_{t-1}}|\mathbf{x_t})}{q(\mathbf{x_{t-1}}|\mathbf{x_{t}}, \mathbf{x_0})} \\
\end{aligned} 
$$

第一项为重建项reconstruction, 第二项为先验项prior matching，第三项为consistency term。其中，第一项重建项和VAE中的重建项类似，因为 $p_\theta(\mathbf{x_0}|\mathbf{x_1})$ 为参数化的高斯分布，即优化重建图和原图的MSE；第二项先验项，因为我们假设步数足够的情况下， $ q(\mathbf{x_T} | \mathbf{x_0}) = \mathcal{N}(\mathbf{x_T} |0, I)$ 且  $p_\theta(\mathbf{x_T}) = \mathcal{N}(0, I)$，推理采样阶段 $p_\theta(\mathbf{x_T})$ 从正态分布中采样，与模型参数无关，所以先验项可以忽略； 重点是第三项consistency term，其实训练阶段，重建项往往也合并到了consistency term中，继续对consistency term进行推导：

$$
\begin{aligned} 
&\sum_{t=2}^T \mathbb{E}_{q(\mathbf{x_{1:T}}|\mathbf{x_0})}  \left[ \log  \frac{p_\theta(\mathbf{x_{t-1}}|\mathbf{x_t})}{q(\mathbf{x_{t-1}}|\mathbf{x_{t}}, \mathbf{x_0})}\right] \\
=&\sum_{t=2}^T \int q(\mathbf{x_{t-1}}, \mathbf{x_t} | \mathbf{x_0}) \log  \frac{p_\theta(\mathbf{x_{t-1}}|\mathbf{x_t})}{q(\mathbf{x_{t-1}}|\mathbf{x_{t}}, \mathbf{x_0})} d\mathbf{x_{t-1:t}} \\
=&\sum_{t=2}^T \int q(\mathbf{x_{t-1}}| \mathbf{x_0}, \mathbf{x_t} ) q(\mathbf{x_t}| \mathbf{x_0})  \log  \frac{p_\theta(\mathbf{x_{t-1}}|\mathbf{x_t})}{q(\mathbf{x_{t-1}}|\mathbf{x_{t}}, \mathbf{x_0})} d\mathbf{x_{t-1:t}} \\
=& - \sum_{t=2}^T \int q(\mathbf{x_t}| \mathbf{x_0}) D_{\text{KL}} (q(\mathbf{x_{t-1}}|\mathbf{x_{t}}, \mathbf{x_0}) \, || \, p_\theta(\mathbf{x_{t-1}}|\mathbf{x_t})) d\mathbf{x_{t}} \\
=& - \sum_{t=2}^T \mathbb{E}_{q(\mathbf{x_{t}}|\mathbf{x_0})} \left[ D_{\text{KL}} (q(\mathbf{x_{t-1}}|\mathbf{x_{t}}, \mathbf{x_0}) \, || \, p_\theta(\mathbf{x_{t-1}}|\mathbf{x_t})) \right] \\ 
\end{aligned} 
$$

根据我们之前的推导，不难发现对consistency term来说，仍然是求两个高斯分布的KL散度。在VAE中就提到过，俩个高斯分布的KL散度是可以直接算解析解的：

$$
\begin{aligned}
D_{\text{KL}}\left(\mathcal{N}(\mu_0, \Sigma_0) \parallel \mathcal{N}(\mu_1, \Sigma_1)\right) = \frac{1}{2} \left[
\operatorname{tr}(\Sigma_1^{-1}\Sigma_0) + (\mu_1 - \mu_0)^\top \Sigma_1^{-1}(\mu_1 - \mu_0) - k + \log \frac{\det \Sigma_1}{\det \Sigma_{0}} \right]
\end{aligned}
$$

代入 $q(\mathbf{x_{t-1}}|\mathbf{x_{t}}, \mathbf{x_0})$ 和  $p_\theta(\mathbf{x_{t-1}}|\mathbf{x_t})$ ,可以得到：

$$
\begin{aligned}
& D_{\text{KL}} (q(\mathbf{x_{t-1}}|\mathbf{x_{t}}, \mathbf{x_0}) \, || \, p_\theta(\mathbf{x_{t-1}}|\mathbf{x_t})) \\[1em]
=& D_{\text{KL}} (\mathcal{N}(\mathbf{x_{t-1}}| \mu_q(\mathbf{x}_t, \mathbf{x}_0), \Sigma_q(t)) \, || \, \mathcal{N}(\mathbf{x_{t-1}}| \mu_{\theta}(\mathbf{x}_t, t), \Sigma_q(t)))
\end{aligned}
$$

因为 $\Sigma_q(t) = \frac{(1 - \alpha_t)(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} \mathbf{I} $ 是有明确表达形式的。

所以 $p_\theta(\mathbf{x_{t-1}}|\mathbf{x_t})$ 的方差不需要神经网络去拟合，只需要学习 $\mu_{\theta}(\mathbf{x}_t, t)$ 即可。

令 $ \sigma_{q}^2(t) = \frac{(1 - \alpha_t)(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t}$， 那么可以得到：

$$
\begin{aligned}
& D_{\text{KL}} (q(\mathbf{x_{t-1}}|\mathbf{x_{t}}, \mathbf{x_0}) \, || \, p_\theta(\mathbf{x_{t-1}}|\mathbf{x_t})) \\[1em]
=& D_{\text{KL}} (\mathcal{N}(\mathbf{x_{t-1}}| \mu_q(\mathbf{x}_t, \mathbf{x}_0), \Sigma_q(t)) \, || \, \mathcal{N}(\mathbf{x_{t-1}}| \mu_{\theta}(\mathbf{x}_t, t), \Sigma_q(t))) \\
=& \frac{1}{2\sigma_{q}^2(t)} \|  \mu_q(\mathbf{x}_t, \mathbf{x}_0) -  \mu_{\theta}(\mathbf{x}_t, t) \|^2
\end{aligned}
$$

以上就是推导consistency term得到的最终优化目标了：两个均值向量之间的欧氏距离平方。实际上经过简单的变换，上述目标可以变为更简洁的形式，但在此之前，让我们看看重建项的解析形式：

$$
\begin{aligned}
\log p_\theta(\mathbf{x_0}|\mathbf{x_1}) &= \log \mathcal{N} (\mathbf{x_0} | \mu_\theta(\mathbf{x_1}, 1), \sigma_{q}^2(1)\mathbf{I}) \\
&= - \frac{\| \mathbf{x_0} - \mu_\theta(\mathbf{x_1},1) \|^2}{2\sigma_{q}^2(1)} - \frac{d}{2} \log 2\pi\sigma_{q}^2(1)
\end{aligned}
$$

去掉与模型训练无关的参数得到最后的优化目标 $\mathcal{L}_{VLB}$ 为：

$$
\begin{aligned}
\mathcal{L}_{VLB} &= \mathbb{E}_{q(\mathbf{x_{1}}|\mathbf{x_0})} \left[ \log p_\theta(\mathbf{x_0}|\mathbf{x_1}) \right]+ \sum_{t=2}^T \mathbb{E}_{q(\mathbf{x_{1:T}}|\mathbf{x_0})}  \left[ \log  \frac{p_\theta(\mathbf{x_{t-1}}|\mathbf{x_t})}{q(\mathbf{x_{t-1}}|\mathbf{x_{t}}, \mathbf{x_0})} \right] \\
&= - \frac{\| \mathbf{x_0} - \mu_\theta(\mathbf{x_1},1) \|^2}{2\sigma_{q}^2(1)} -  \sum_{t=2}^T \mathbb{E}_{q(\mathbf{x_{t}}|\mathbf{x_0})} \left[\frac{1}{2\sigma_{q}^2(t)} \|  \mu_q(\mathbf{x}_t, \mathbf{x}_0) -  \mu_{\theta}(\mathbf{x}_t, t) \|^2 \right]
\end{aligned}
$$

### 4. $\mathcal{L}_{VLB}$ 求解/分析

#### 4.1 优化形式一

观察 $\mathcal{L}_{VLB}$，核心是优化：

$$
\frac{1}{2\sigma_{q}^2(t)} \|  \mu_q(\mathbf{x}_t, \mathbf{x}_0) -  \mu_{\theta}(\mathbf{x}_t, t) \|^2
$$

已知 $\mu_q(\mathbf{x}_t, \mathbf{x}_0)$ 的表达式：

$$
\mu_q(\mathbf{x}_t, \mathbf{x}_0) = \frac{(1 - \bar{\alpha}_{t-1})\sqrt{\alpha_t}}{1 - \bar{\alpha}_t} \mathbf{x}_t + \frac{(1 - \alpha_t)\sqrt{\bar{\alpha}_{t-1}}}{1 - \bar{\alpha}_t} \mathbf{x}_0 
$$

考虑到已知的变量和超参，可以将 $\mu_{\theta}(\mathbf{x}_t, t)$ 定义为：

$$
\mu_{\theta}(\mathbf{x}_t, t) := \frac{(1 - \bar{\alpha}_{t-1})\sqrt{\alpha_t}}{1 - \bar{\alpha}_t} \mathbf{x}_t + \frac{(1 - \alpha_t)\sqrt{\bar{\alpha}_{t-1}}}{1 - \bar{\alpha}_t} \mathbf{x}_\theta (\mathbf{x_t}, t) 
$$

容易发现，模型输出的 $ \mathbf{x}_\theta (\mathbf{x_t}, t) $ 现在学习目标为 $\mathbf{x_0}$， 那么优化目标可以改写为：

$$
\frac{1}{2\sigma_{q}^2(t)} \|  \mu_q(\mathbf{x}_t, \mathbf{x}_0) -  \mu_{\theta}(\mathbf{x}_t, t) \|^2 = \frac{1}{2\sigma_{q}^2(t)} \cdot \frac{(1 - \alpha_t)^2 \bar{\alpha}_{t-1}}{(1 - \bar{\alpha}_t)^2} \| \mathbf{x_0} - \mathbf{x}_\theta (\mathbf{x_t}, t)\|^2
$$

有意思的是，设置 $ \alpha_0 =1$, 可以发现与重建项一致，进行合并，可以得到：

$$
\mathcal{L}_{VLB} = - \sum_{t=1}^T \frac{1}{2\sigma_{q}^2(t)} \cdot \frac{(1 - \alpha_t)^2 \bar{\alpha}_{t-1}}{(1 - \bar{\alpha}_t)^2} \mathbb{E}_{q(\mathbf{x_{t}}|\mathbf{x_0})} \left[ \| \mathbf{x_0} - \mathbf{x}_\theta (\mathbf{x_t}, t)\|^2 \right]
$$

#### 4.2 优化形式二

回到 $\mu_q(\mathbf{x}_t, \mathbf{x}_0)$ 的表达式：

$$
\mu_q(\mathbf{x}_t, \mathbf{x}_0) = \frac{(1 - \bar{\alpha}_{t-1})\sqrt{\alpha_t}}{1 - \bar{\alpha}_t} \mathbf{x}_t + \frac{(1 - \alpha_t)\sqrt{\bar{\alpha}_{t-1}}}{1 - \bar{\alpha}_t} \mathbf{x}_0 
$$

利用上我们已知的其他信息，也就是 $\mathbf{x}_t 和 \mathbf{x}_0$ 间的关系：

$$
\begin{aligned}
\mathbf{x}_t &= \sqrt{\bar{\alpha}_t} \mathbf{x_0} +  \sqrt{1 - \bar{\alpha}_t} \epsilon_{t} \\
\mathbf{x_0} &= \frac{\mathbf{x}_t - \sqrt{1 - \bar{\alpha}_t} \epsilon_{t}}{\sqrt{\bar{\alpha}_t}}
\end{aligned}
$$

代入 $\mu_q(\mathbf{x}_t, \mathbf{x}_0)$ 的表达式中：

$$
\begin{aligned}
\mu_q(\mathbf{x}_t, \mathbf{x}_0) &= \frac{(1 - \bar{\alpha}_{t-1})\sqrt{\alpha_t}}{1 - \bar{\alpha}_t} \mathbf{x}_t + \frac{(1 - \alpha_t)\sqrt{\bar{\alpha}_{t-1}}}{1 - \bar{\alpha}_t} \mathbf{x}_0 \\
&= \frac{(1 - \bar{\alpha}_{t-1})\sqrt{\alpha_t}}{1 - \bar{\alpha}_t} \mathbf{x}_t + \frac{(1 - \alpha_t)\sqrt{\bar{\alpha}_{t-1}}}{1 - \bar{\alpha}_t} (\frac{\mathbf{x}_t - \sqrt{1 - \bar{\alpha}_t} \epsilon_{t}}{\sqrt{\bar{\alpha}_t}}) \\
&= \frac{(1 - \bar{\alpha}_{t-1})\alpha_t + 1 - \alpha_t}{(1 - \bar{\alpha}_t)\sqrt{\alpha_t}} \mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{(1 - \bar{\alpha}_t)}\sqrt{\alpha_t}} \epsilon_{t} \\
&= \frac{1}{\sqrt{\alpha_t}}  \mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{(1 - \bar{\alpha}_t)}\sqrt{\alpha_t}} \epsilon_{t}
\end{aligned}
$$

类似地，我们可以将 $\mu_{\theta}(\mathbf{x}_t, t)$ 定义为：

$$
\mu_{\theta}(\mathbf{x}_t, t) = \frac{1}{\sqrt{\alpha_t}}  \mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{(1 - \bar{\alpha}_t)}\sqrt{\alpha_t}} \epsilon_{\theta} (\mathbf{x_t}, t)
$$

同样地，优化目标可以改写为：

$$
\frac{1}{2\sigma_{q}^2(t)} \|  \mu_q(\mathbf{x}_t, \mathbf{x}_0) -  \mu_{\theta}(\mathbf{x}_t, t) \|^2 = \frac{1}{2\sigma_{q}^2(t)} \cdot \frac{(1 - \alpha_t)^2 }{\alpha_t(1 - \bar{\alpha}_t)}  \| \epsilon_t - \epsilon_\theta(\mathbf{x_t}, t)\|^2 
$$

根据 $\mathbf{x}_1 和 \mathbf{x}_0$ 间的关系，将重建项也改写一下：

$$
\begin{aligned}
\mathbf{x}_1 &= \sqrt{\alpha_1} \mathbf{x}_0 + \sqrt{1 - \alpha_1} \epsilon_1 \\
\mathbf{x}_0 &= \frac{1}{sqrt{\alpha_1}} \mathbf{x}_1 - \frac{\sqrt{1 - \alpha_1}}{sqrt{\alpha_1}} \epsilon_1 \\
\frac{1}{2\sigma_{q}^2(1)} \| \mathbf{x_0} - \mu_\theta(\mathbf{x_1},1) \|^2 &=  \frac{1}{2\sigma_{q}^2(1)} \cdot\frac{1 - \alpha_1 }{\alpha_1} \| \epsilon_1 - \epsilon_\theta(\mathbf{x_1}, 1)\|^2 
\end{aligned}
$$

与优化形式一中一样，合并重建项，得到最终的 $\mathcal{L}_{VLB}$ 为：

$$
\mathcal{L}_{VLB} = - \sum_{t=1}^T \frac{1}{2\sigma_{q}^2(t)} \cdot \frac{(1 - \alpha_t)^2 }{\alpha_t(1 - \bar{\alpha}_t)} \mathbb{E}_{q(\mathbf{x_{t}}|\mathbf{x_0})} \left[ \| \epsilon_t - \epsilon_\theta(\mathbf{x_t}, t)\|^2 \right]
$$

容易发现，模型本质上是在学习一个噪声预测网络 $\epsilon_\theta$，其目标是最小化预测噪声 $\epsilon_\theta(\mathbf{x_t}, t)$与真实噪声 $\epsilon_t$ 之间的加权均方误差。

---

## 三. Training 和 Inference流程

Training 流程，损失函数就是 $ \epsilon_\theta(\mathbf{x_t}, t) $ 与 $\epsilon_t$ 的欧氏距离平方

Inference流程就是分解联合分布 $p_\theta(\mathbf{x_{0:T}})$，从 $p_\theta(\mathbf{x_{t-1}} | \mathbf{x_t})$ 中逐步完成采样

----
### Algorithm 1 Training

1: **repeat**  
2: &nbsp;&nbsp;&nbsp;&nbsp; $x_0 \sim q(x_0)$  
3: &nbsp;&nbsp;&nbsp;&nbsp; $t \sim \text{Uniform}(\{1, \ldots, T\})$  
4: &nbsp;&nbsp;&nbsp;&nbsp; $\epsilon \sim \mathcal{N}(0, \mathbf{I})$  
5: &nbsp;&nbsp;&nbsp;&nbsp; Take gradient descent step on  
&nbsp;&nbsp;&nbsp;&nbsp; $\nabla_\theta \left\| \epsilon - \epsilon_\theta\left(\sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon, t \right) \right\|^2$  
6: **until** converged  

----
### Algorithm 2 Sampling (Inference)

1: $\mathbf{x_T} \sim \mathcal{N}(0, \mathbf{I})$  
2: **for** $t = T, \ldots, 1$ **do**  
3: &nbsp;&nbsp;&nbsp;&nbsp; $\mathbf{z} \sim \mathcal{N}(0, \mathbf{I})$ **if** $t > 1$, **else** $\mathbf{z} = 0$  
4: &nbsp;&nbsp;&nbsp;&nbsp; $\mathbf{x_{t-1}} = \frac{1}{\sqrt{\alpha}} \left( \mathbf{x_t} - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha_t}}} \epsilon_\theta(\mathbf{x_t}, t) \right) + \sigma_t \mathbf{z}$  
5: **end for**  
6: **return** $\mathbf{x}_0$  

<!-- \left( x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right) + \sigma_t z$   -->


<!-- $$
\begin{aligned}
q(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0) &= \frac{q(\mathbf{x}_t | \mathbf{x}_{t-1}) \cdot q(\mathbf{x}_{t-1} | \mathbf{x}_0)}{q(\mathbf{x_{t}} | \mathbf{x}_0)} \\
&= \frac{
    \mathcal{N}\left(\mathbf{x}_t \mid \sqrt{\alpha_t}\mathbf{x}_{t-1}, (1-\alpha_t)\mathbf{I}\right)
    \cdot \mathcal{N}\left(\mathbf{x}_{t-1} \mid \sqrt{\bar{\alpha}_{t-1}}\mathbf{x}_0, (1-\bar{\alpha}_{t-1})\mathbf{I}\right)
}{
    \mathcal{N}\left(\mathbf{x}_t \mid \sqrt{\bar{\alpha}_t}\mathbf{x}_0, (1-\bar{\alpha})\mathbf{I}\right)
} \\
&= \propto \exp\left\{
-\frac{1}{2}\left[
\frac{(\mathbf{x}_t - \sqrt{\alpha_t}\mathbf{x}_{t-1})^2}{1-\alpha_t}
+ \frac{(\mathbf{x}_{t-1} - \sqrt{\alpha_{t-1}}\mathbf{x}_0)^2}{1-\alpha_{t-1}}
- \frac{(\mathbf{x}_t - \sqrt{\alpha_t}\mathbf{x}_0)^2}{1-\alpha_t}
\right]
\right\}
\end{aligned}
$$ -->





<!-- % q(\mathbf{x}_t | \mathbf{x}_{t-1}, \mathbf{x}_0) &= \frac{q(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0) \cdot q(\mathbf{x}_t | \mathbf{x}_0)}{q(\mathbf{x}_{t-1} | \mathbf{x}_0)}
% &= \frac{q(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0) \cdot q(\mathbf{x}_t | \mathbf{x}_0)}{q(\mathbf{x}_{t-1} | \mathbf{x}_0)} -->


---


<!-- ## Code Snippet (Optional)

```python
# Simplified DDPM training step (PyTorch-style)
def train_step(model, x0, t, noise):
    xt = sqrt_alphas_cumprod[t] * x0 + sqrt_one_minus_alphas_cumprod[t] * noise
    predicted_noise = model(xt, t)
    loss = F.mse_loss(noise, predicted_noise)
    return loss
```  -->

## Reference
[^2]: Ho, J., Jain, A., & Abbeel, P. (2020). Denoising Diffusion Probabilistic Models. *NeurIPS*.
[^1]: Sohl-Dickstein, J., et al. (2015). Deep Unsupervised Learning using Nonequilibrium Thermodynamics. *ICML*.  
[^3]: Song, Y., & Ermon, S. (2019). Generative Modeling by Estimating Gradients of the Data Distribution. *NeurIPS*.
+++
date = '2025-06-24T20:08:36+08:00'
draft = false
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

<!-- #### 数学形式
DDIM 的逆过程改写为：
$$
x_{t-1} = \sqrt{\alpha_{t-1}} \left( \frac{x_t - \sqrt{1-\alpha_t} \epsilon_\theta(x_t, t)}{\sqrt{\alpha_t}} \right) + \sqrt{1-\alpha_{t-1}} \epsilon_\theta(x_t, t)
$$
其中 $\alpha_t$ 是噪声调度，$\epsilon_\theta$ 是去噪网络。当噪声项系数为0时，生成过程变为确定性。 -->

---

### 1.2 SDE/ODE视角下的DDIM

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

### 2.1 DDPM的非马尔可夫推广

<!-- 插句题外话 -->
回顾DDPM的优化目标 $\mathcal{L}_{VLB}$ :

$$
\mathcal{L}_{VLB} = - \sum_{t=1}^T \frac{1}{2\sigma_{q}^2(t)} \cdot \frac{(1 - \alpha_t)^2 }{\alpha_t(1 - \bar{\alpha}_t)} \mathbb{E}_{q(\mathbf{x_{t}}|\mathbf{x_0})} \left[ \| \epsilon_t - \epsilon_\theta(\mathbf{x_t}, t)\|^2 \right]
$$

DDIM的核心动机，就来源于DDPM的目标函数只依赖于边缘分布 $q(\mathbf{x_t}|\mathbf{x_0})$，
而不是联合分布 $q(\mathbf{x_{1:T}}|\mathbf{x_0})$。

Note: 以下将只以 $\mathbf{x_0}$ 为条件的类似 $q(\mathbf{x_t}|\mathbf{x_0})$ 的条件分布称为边缘分布。

再回顾DDPM优化目标推导过程的一个中间表达形式：

$$
\frac{1}{2\sigma_{q}^2(t)} \|  \mu_q(\mathbf{x}_t, \mathbf{x}_0) -  \mu_{\theta}(\mathbf{x}_t, t) \|^2
$$

仔细想想不难发现，其实我们关心的只有 $q(\mathbf{x_t}|\mathbf{x_0})$ 和 $q(\mathbf{x_{t-1}}|\mathbf{x_0}, \mathbf{x_t})$ 的均值 $\mu_q(\mathbf{x}_t, \mathbf{x}_0)$

通过对联合分布 $q(\mathbf{x_{1:T}} | \mathbf{x}_0)$ 进行合适的分解，我们可以在保持DDPM优化目标不变的前提下，完成非马尔可夫链的推广形式。根据贝叶斯公式有：

$$
\begin{aligned}
q(\mathbf{x}_{t-1} | \mathbf{x}_0, \mathbf{x}_t) &= \frac{q(\mathbf{x}_t | \mathbf{x}_0, \mathbf{x}_{t-1}) \cdot q(\mathbf{x}_{t-1} | \mathbf{x}_0)}{q(\mathbf{x_{t}} | \mathbf{x}_0)} \\
q(\mathbf{x}_{t} | \mathbf{x}_0, \mathbf{x}_{t-1}) &= \frac{q(\mathbf{x}_{t-1} | \mathbf{x}_0, \mathbf{x}_{t}) \cdot q(\mathbf{x}_{t} | \mathbf{x}_0)}{q(\mathbf{x_{t-1}} | \mathbf{x}_0)} \\
\end{aligned}
$$

{{< figure src="/pic_diff_2/non-markov-forward.png" title="">}}

比如基于上图右的前向过程，对联合分布 $q(\mathbf{x_{1:T}} | \mathbf{x}_0)$ 进行分解

$$
\begin{aligned}
q(\mathbf{x_{1:T}} | \mathbf{x}_0) &= \prod_{t=1}^{T} q(\mathbf{x_{t}} | \mathbf{x}_0, \mathbf{x}_{t-1}) \\
&= \prod_{t=1}^{T} \frac{q(\mathbf{x}_{t-1} | \mathbf{x}_0, \mathbf{x}_{t}) \cdot q(\mathbf{x}_{t} | \mathbf{x}_0)}{q(\mathbf{x_{t-1}} | \mathbf{x}_0)} \\
&= q(\mathbf{x}_{1} | \mathbf{x}_0) \prod_{t=2}^{T} \frac{q(\mathbf{x}_{t-1} | \mathbf{x}_0, \mathbf{x}_{t}) \cdot q(\mathbf{x}_{t} | \mathbf{x}_0)}{q(\mathbf{x_{t-1}} | \mathbf{x}_0)}
\end{aligned}
$$

回顾DDPM的优化目标推导中最核心的一部分：

$$
\begin{aligned}
\log p_\theta(\mathbf{x_0}) &\geq \mathbb{E}_{q(\mathbf{x_{1:T}}|\mathbf{x_0})} \left[ \log \frac{p_\theta(\mathbf{x_{0:T}})}{q(\mathbf{x_{1:T}}|\mathbf{x_0})} \right] \\
&= \mathbb{E}_{q(\mathbf{x_{1:T}}|\mathbf{x_0})} \left[ \log \frac{p_\theta(\mathbf{x_T}) p_\theta(\mathbf{x_0}|\mathbf{x_1}) \prod_{t=2}^T p_\theta(\mathbf{x_{t-1}}|\mathbf{x_t})}{q(\mathbf{x_1}|\mathbf{x_0}) \prod_{t=2}^T q(\mathbf{x_t}|\mathbf{x_{t-1}}, \mathbf{x_0})} \right] \\
&= \mathbb{E}_{q(\mathbf{x_{1:T}}|\mathbf{x_0})} \left[ \log \frac{p_\theta(\mathbf{x_T}) p_\theta(\mathbf{x_0}|\mathbf{x_1})}{q(\mathbf{x_1}|\mathbf{x_0})} \right] + \mathbb{E}_{q(\mathbf{x_{1:T}}|\mathbf{x_0})} \left[ \log \prod_{t=2}^T \frac{p_\theta(\mathbf{x_{t-1}}|\mathbf{x_t})}{q(\mathbf{x_t}|\mathbf{x_{t-1}}, \mathbf{x_0})} \right] \\
&= \mathbb{E}_{q(\mathbf{x_{1:T}}|\mathbf{x_0})} \left[ \log \frac{p_\theta(\mathbf{x_T}) p_\theta(\mathbf{x_0}|\mathbf{x_1})}{q(\mathbf{x_1}|\mathbf{x_0})} \right] + \mathbb{E}_{q(\mathbf{x_{1:T}}|\mathbf{x_0})} \left[ \log \prod_{t=2}^T \frac{p_\theta(\mathbf{x_{t-1}}|\mathbf{x_t})}{q(\mathbf{x_{t-1}}|\mathbf{x_t}, \mathbf{x_0})} \cdot \frac{q(\mathbf{x_{t-1}}| \mathbf{x_0})}{q(\mathbf{x_t}| \mathbf{x_0})} \right] \\
\end{aligned}
$$

可以发现，通过上图构造的非马尔可夫前向过程，对联合分布 $q(\mathbf{x_{1:T}} | \mathbf{x}_0)$ 进行分解后，我们就达成了DDPM前向过程的非马尔可夫推广。

其实到这里，我们就已经实现了**在不改动DDPM推导形式的前提下，得到DDPM前向过程的非马尔可夫推广**。
剩下要做的就是给出 $ q(\mathbf{x_{t-1}}| \mathbf{x_0})$, $q(\mathbf{x_t}| \mathbf{x_0})$,和 
$q(\mathbf{x_{t-1}} | \mathbf{x_0}, \mathbf{x_{t}})$ 的解析形式，得到这3个分布的解析形式，我们就可以推导出最终的优化目标。

DDIM这里采用的方案是直接沿用DDPM中 $ q(\mathbf{x_{t-1}}| \mathbf{x_0}), q(\mathbf{x_t}| \mathbf{x_0}), q(\mathbf{x_{t-1}} | \mathbf{x_0}, \mathbf{x_{t}})$ 的解析形式。这就非常nice了，以为着我们**不仅可以沿用DDPM推导形式，还可以继续使用DDPM的优化目标$\mathcal{L}_{VLB}$**，在将DDPM的形式推广到非马尔可夫形式后，我们甚至无需重新训一个模型，直接拿DDPM训好的模型就行了。

接下来要做的就是证明沿用DDPM中 $ q(\mathbf{x_{t-1}}| \mathbf{x_0}), q(\mathbf{x_t}| \mathbf{x_0}), q(\mathbf{x_{t-1}} | \mathbf{x_0}, \mathbf{x_{t}})$ 的解析形式的正确性。

前面提到，我们希望目标函数所依赖边缘分布 $q(\mathbf{x_t}|\mathbf{x_0})$保持不变，而无关前向的条件分布 $q(\mathbf{x_t}|\mathbf{x_{t-1}})$，因此定义边缘分布 $q(\mathbf{x_t}|\mathbf{x_0})$ 为：

$$
q(\mathbf{x_t}|\mathbf{x_0}) := \mathcal{N}(\mathbf{x_t} | \sqrt{\bar{\alpha}_t} \mathbf{x_0}, (1 - \bar{\alpha}_t) \mathbf{I})
$$

接下来就是确定 $ q(\mathbf{x_{t-1}} | \mathbf{x_0}, \mathbf{x_{t}})$ 的解析形式， DDPM部分提到了 $ q(\mathbf{x_{t-1}} | \mathbf{x_0}, \mathbf{x_{t}})$ 均值部分就是 $\mathbf{x_0} 和 \mathbf{x_t}$ 的线性组合，因此定义 $ q(\mathbf{x_{t-1}} | \mathbf{x_0}, \mathbf{x_{t}})$：

$$
q(\mathbf{x_{t-1}} | \mathbf{x_0}, \mathbf{x_{t}}) := \mathcal{N}(\mathbf{x_{t-1}} | k \mathbf{x_0} + m\mathbf{x_t}, \sigma_t^2 \mathbf{I})
$$

对 $ k, m $ 进行求解：

$$
\begin{aligned}
\mathbf{x_{t-1}} &= k\mathbf{x_0} + m \mathbf{x_t} + \sigma_t \epsilon_t^{'} \\
&= k\mathbf{x_0} + m (\sqrt{\bar{\alpha}_t} \mathbf{x_0} + \sqrt{1 - \bar{\alpha}_t} \epsilon_t^{''}) + \sigma_t \epsilon_t^{'} \\
&= (k + m\sqrt{\bar{\alpha}_t}) \mathbf{x_0}  + \sqrt{m^2(1 - \bar{\alpha}_t) + \sigma_t^2} \cdot \epsilon_t
\end{aligned}
$$

注意这里两个高斯噪声 $\epsilon_t^{'}, \epsilon_t^{''}$ 的合并，一个人为引入的采样随机性，另一个是前向的噪声，视为相互独立的高斯噪声。根据定义有：

$$
\begin{aligned}
q(\mathbf{x_{t-1}}|\mathbf{x_0}) &:= \mathcal{N}(\mathbf{x_t} | \sqrt{\bar{\alpha}_{t-1}} \mathbf{x_0}, (1 - \bar{\alpha}_{t-1}) \mathbf{I}) \\
k + m\sqrt{\bar{\alpha}_t} &= \sqrt{\bar{\alpha}_{t-1}} \\
m^2(1 - \bar{\alpha}_t) + \sigma_t^2 &= 1 - \bar{\alpha}_{t-1}
\end{aligned}
$$

解出来得到 $k, m$ 的取值：

$$
\begin{aligned}
k &= \sqrt{\bar{\alpha}_{t-1}} - \frac{\sqrt{1 - \bar{\alpha}_{t-1}-\sigma_t^2}}{\sqrt{1-\bar{\alpha}_{t}}} \sqrt{\bar{\alpha}_{t}} \\
m &= \frac{\sqrt{1 - \bar{\alpha}_{t-1}-\sigma_t^2}}{\sqrt{1-\bar{\alpha}_{t}}}
\end{aligned}
$$

将 $k, m$ 代入 $q(\mathbf{x_{t-1}} | \mathbf{x_0}, \mathbf{x_{t}}) = \mathcal{N}(\mathbf{x_{t-1}} | k \mathbf{x_0} + m\mathbf{x_t}, \sigma_t^2 \mathbf{I})$ 可得：

$$
\begin{aligned}
\mathbf{x_{t-1}} &= (k + m\sqrt{\bar{\alpha}_t}) \mathbf{x_0}  + \sqrt{m^2(1 - \bar{\alpha}_t) + \sigma_t^2} \cdot \epsilon_t \\
&= \sqrt{\bar{\alpha}_{t-1}} \mathbf{x_0} + \sqrt{1-\bar{\alpha}_{t-1}-\sigma_t^2} \cdot \epsilon_t^{''} + \sigma_t \cdot \epsilon_t^{'}
\end{aligned}
$$

也就是说：

$$
q(\mathbf{x_{t-1}} | \mathbf{x_0}, \mathbf{x_{t}}) = \mathcal{N}(\sqrt{\bar{\alpha}_{t-1}} \mathbf{x_0} + \sqrt{1-\bar{\alpha}_{t-1}-\sigma_t^2} \cdot \epsilon_t^{''}, \sigma_t^2 \mathbf{I})
$$

回顾DDPM中对 $q(\mathbf{x_{t-1}} | \mathbf{x_0}, \mathbf{x_{t}})$ 求解的均值项 $\mu_q(\mathbf{x}_t, \mathbf{x}_0)$：

$$
\begin{aligned}
\mu_q(\mathbf{x}_t, \mathbf{x}_0) &= \frac{(1 - \bar{\alpha}_{t-1})\sqrt{\alpha_t}}{1 - \bar{\alpha}_t} \mathbf{x}_t + \frac{(1 - \alpha_t)\sqrt{\bar{\alpha}_{t-1}}}{1 - \bar{\alpha}_t} \mathbf{x}_0 \\
&= \sqrt{\bar{\alpha}_{t-1}} \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_{t-1} - \sigma_t^2} \cdot \epsilon_t^{''}
\end{aligned}
$$

<!-- $$
\begin{aligned}
\mu_q(\mathbf{x}_t, \mathbf{x}_0) &= \frac{(1 - \bar{\alpha}_{t-1})\sqrt{\alpha_t}}{1 - \bar{\alpha}_t} (\sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\epsilon_t) + \frac{(1 - \alpha_t)\sqrt{\bar{\alpha}_{t-1}}}{1 - \bar{\alpha}_t} \mathbf{x}_0 \\
&= \left( \frac{(1 - \bar{\alpha}_{t-1})\alpha_t}{1 - \bar{\alpha}_t} + \frac{(1 - \alpha_t)\bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \right) \mathbf{x}_0 + \frac{(1 - \bar{\alpha}_{t-1})\alpha_t \sqrt{1-\bar{\alpha}_t}}{1 - \bar{\alpha}_t} \epsilon_t  \\
&= \left( \frac{(1 - \bar{\alpha}_{t-1})\alpha_t + (1 - \alpha_t)\bar{\alpha}_{t-1}(1 - \bar{\alpha}_t)}{1 - \bar{\alpha}_t} \right) \mathbf{x}_0 + \frac{(1 - \bar{\alpha}_{t-1})(1 - \alpha_t)}{1 - \bar{\alpha}_t} \epsilon_t \\
&= \sqrt{\bar{\alpha}_{t-1}} \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_{t-1} - \frac{(1 - \alpha_t)(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t}} \cdot \epsilon_t \\
&= \sqrt{\bar{\alpha}_{t-1}} \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_{t-1} - \sigma^2_q(t)} \cdot \epsilon_t
\end{aligned}
$$ -->

注意这里 $\epsilon_t^{''}$ 对应的就是从 $\mathbf{x_0}$ 到 $\mathbf{x_t}$ 过程中的噪声。
我们的目标就是想要上述的均值项能够匹配上，区别在于，在DDPM中 $\epsilon_t^{''}$ 系数项中的 $ \sigma_{q}^2(t) = \frac{(1 - \alpha_t)(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t}$，这一项是由前向过程中的超参决定。

而DDIM中的 $ \sigma_{q}^2(t)$ 是可以人为设计的，当我们设置DDIM中的 $ \sigma_{q}^2(t) = \frac{(1 - \alpha_t)(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t}$，DDIM与DDPM一致；

当 $ \sigma_{q}^2(t) = 0 $ 时，也就是讲到 DDIM时所默认的形式，此时为确定性推理，因为模型学习的逆向分布 $P_\theta(\mathbf{x_{t-1}} | \mathbf{x_{t}})$ 方差也是 $0$。

总结一下，DDIM为DDPM的非马尔可夫推广，推导和目标函数与 DDPM一致。

#### DDIM的前向过程保持不变：

$$
\mathbf{x_t} = \sqrt{\bar{\alpha}_t} \mathbf{x_0} +  \sqrt{1 - \bar{\alpha}_t} \cdot \epsilon_{t}^{''}
$$

再看采样（逆向）过程，先把 $\mu_q(\mathbf{x}_t, \mathbf{x}_0)$ 转为 $\mathbf{x}_t$ 和已知前向 $\epsilon_t^{''}$ 的组合：

$$
\begin{aligned}
\mu_q(\mathbf{x}_t, \mathbf{x}_0) &= \sqrt{\bar{\alpha}_{t-1}} \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_{t-1} - \sigma_t^2} \cdot \epsilon_t^{''} \\
&= \sqrt{\bar{\alpha}_{t-1}} (\frac{1}{\sqrt{\bar{\alpha}_t}} \mathbf{x}_t - \frac{\sqrt{1 - \bar{\alpha}_t}}{\sqrt{\bar{\alpha}_t}} \cdot \epsilon_t^{''}) + \sqrt{1 - \bar{\alpha}_{t-1} - \sigma_t^2} \cdot \epsilon_t^{''} \\ 
&= \frac{1}{\sqrt{\alpha_t}}  \mathbf{x}_t - (\frac{\sqrt{1 - \bar{\alpha}_t}}{\sqrt{\alpha_t}} - \sqrt{1 - \bar{\alpha}_{t-1} - \sigma_t^2}) \cdot \epsilon_t^{''}
\end{aligned}
$$
<!-- $$
\begin{aligned}
\mu_q(\mathbf{x}_t, \mathbf{x}_0) &= \frac{(1 - \bar{\alpha}_{t-1})\sqrt{\alpha_t}}{1 - \bar{\alpha}_t} \mathbf{x}_t + \frac{(1 - \alpha_t)\sqrt{\bar{\alpha}_{t-1}}}{1 - \bar{\alpha}_t} \mathbf{x}_0 \\
&= \frac{(1 - \bar{\alpha}_{t-1})\sqrt{\alpha_t}}{1 - \bar{\alpha}_t} \mathbf{x}_t + \frac{(1 - \alpha_t)\sqrt{\bar{\alpha}_{t-1}}}{1 - \bar{\alpha}_t} (\frac{\mathbf{x}_t - \sqrt{1 - \bar{\alpha}_t} \epsilon_{t}^{''}}{\sqrt{\bar{\alpha}_t}}) \\
&= \frac{1}{\sqrt{\alpha_t}}  \mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{(1 - \bar{\alpha}_t)}\sqrt{\alpha_t}} \epsilon_{t}^{''}
\end{aligned}
$$ -->

同样地，我们只需要学习一个 $ \epsilon_\theta(\mathbf{x_t}, t) $ 来拟合 $ \epsilon_{t}^{''}$ 即可 (这里因为DDPM训练过程中对loss权重项的特殊处理，噪声项系数中 $\sigma_t$ 的差异，并不会影响DDIM把DDPM训好的模型直接拿过来使用)。然后分解联合分布 $p_\theta(\mathbf{x_{0:T}})$，从 $p_\theta(\mathbf{x_{t-1}} | \mathbf{x_t})$ 中逐步完成采样。

$$
p_\theta(\mathbf{x_{t-1}} | \mathbf{x_t}) = \mathcal{N}(\mathbf{x_{t-1}} | \frac{1}{\sqrt{\alpha_t}}  \mathbf{x}_t - (\frac{\sqrt{1 - \bar{\alpha_t}}}{\sqrt{\alpha_t}} - \sqrt{1 - \bar{\alpha}_{t-1} - \sigma_t^2}) \cdot  \epsilon_\theta(\mathbf{x_t}, t), \sigma_t^2 \mathbf{I})
$$

#### 离散化迭代采样的过程为：

$$
\begin{aligned}
\mathbf{x_{t-1}} &= \sqrt{\bar{\alpha}_{t-1}} (\frac{1}{\sqrt{\bar{\alpha}_t}} \mathbf{x}_t - \frac{\sqrt{1 - \bar{\alpha}_t}}{\sqrt{\bar{\alpha}_t}} \cdot \epsilon_\theta(\mathbf{x_t}, t)) + \sqrt{1 - \bar{\alpha}_{t-1} - \sigma_t^2} \cdot \epsilon_\theta(\mathbf{x_t}, t) + \sigma_t \epsilon_t  \\
&= \frac{1}{ \sqrt{\alpha_t}}  \mathbf{x}_t - (\frac{\sqrt{1 - \bar{\alpha}_t}}{\sqrt{\alpha_t}} - \sqrt{1 - \bar{\alpha}_{t-1} - \sigma_t^2}) \cdot  \epsilon_\theta(\mathbf{x_t}, t) + \sigma_t \epsilon_t 
\end{aligned}
$$

可以人为控制 $ \sigma_{q}^2(t)$: 
- 当 $ \sigma_{q}^2(t) = \frac{(1 - \alpha_t)(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t}$，DDIM与DDPM完全一致，代入到DDPM的推理方程中即可推得俩者等价：

$$
\mathbf{x}_{t-1} = \frac{1}{\sqrt{\alpha_t}} \mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{\alpha_t}\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(\mathbf{x}_t, t)  + \sigma_t \epsilon_t
$$

- 当 $ \sigma_{q}^2(t) = 0 $ 时，即 DDIM的默认形式，此时为确定性推理。

### 2.2 DDIM加速采样

要理解DDIM加速采样的本质，可以从2个视角分析：
- DDPM视角：利用非马尔可夫的推广，调整了联合分布的形式，从而实现加速采样
- ODE视角：采样过程等价逆向过程的概率流ODE的离散化，本质是数值解ODE时的取更大的步长

这里主要讲前者，回顾DDPM/DDIM的推导过程引入的联合分布：

$$
\begin{aligned}
\log p_\theta(\mathbf{x_0}) &= \log \int p_\theta(\mathbf{x_{0:T}}) d\mathbf{x_{1:T}} \quad  \\
&= \log \int p_\theta(\mathbf{x_{0:T}}) \frac{q(\mathbf{x_{1:T}}|\mathbf{x_0})}{q(\mathbf{x_{1:T}}|\mathbf{x_0})} d\mathbf{x_{1:T}}  \\
&= \log \int q(\mathbf{x_{1:T}}|\mathbf{x_0}) \left( \frac{p_\theta(\mathbf{x_{0:T}})}{q(\mathbf{x_{1:T}}|\mathbf{x_0})} \right) d\mathbf{x_{1:T}}  \\
\end{aligned}
$$

上述推导在构造联合分布的时候，引入了 $x_{1:T}$ ，因为在马尔可夫假设中，需要 step by step。
而实现了非马尔可夫推广后，我们可以从 $ \lbrace 1,2,\dots, T \rbrace$ 中挑选一个子集进行 $ \lbrace \tau_1,\tau_2,\dots, \tau_S \rbrace$ 重新构造联合分布，令 $\tau_{0} = 0$：

$$
p_\theta(\mathbf{x_0}, \mathbf{x_{\tau_1}},\mathbf{x_{\tau_2}},\dots, \mathbf{x_{\tau_S}}) = p_\theta(\mathbf{x_{\tau_S}}) \prod_{i=1}^S p_\theta(\mathbf{x_{\tau_{i-1}}} | \mathbf{x_{\tau_i}})
$$

对于 DDIM(确定性采样, $ \sigma_{q}^2(t)=0 $)来说，训练的部分不需要做任何改动，只需要调整在推理采样步骤做一点调整：

可以对比一下：
- 未加速采样

$$
\mathbf{x_{t-1}} = \sqrt{\bar{\alpha}_{t-1}} (\frac{1}{\sqrt{\bar{\alpha}_t}} \mathbf{x}_t - \frac{\sqrt{1 - \bar{\alpha}_t}}{\sqrt{\bar{\alpha}_t}} \cdot \epsilon_\theta(\mathbf{x_t}, t)) + \sqrt{1 - \bar{\alpha}_{t-1} - \sigma_t^2} \cdot \epsilon_\theta(\mathbf{x_t}, t)
$$

- 加速采样

$$
\mathbf{x}_{\tau_{i-1}} = \sqrt{\bar{\alpha}_{\tau_{i-1}}} \left( \frac{1}{\sqrt{\bar{\alpha}_{\tau_i}}} \mathbf{x}_{\tau_i} - \frac{\sqrt{1 - \bar{\alpha}_{\tau_i}}}{\sqrt{\bar{\alpha}_{\tau_i}}} \cdot \epsilon_\theta(\mathbf{x}_{\tau_i}, \tau_i) \right) + \sqrt{1 - \bar{\alpha}_{\tau_{i-1}} } \cdot \epsilon_\theta(\mathbf{x_t}, t)
$$

那么的 DDIM的推导，前向/逆向过程就都梳理完了。
我觉得应该还留下一个疑问，就是为什么加速采样的情况下，为什么能直接使用 DDPM训练好的模型，只需要修改采样部分？
这涉及到是Diffusion model 区别于其他生成方法的核心优势之一 “训练-采样解耦”的特性。
这个问题我觉得站在SMLD或者ODE的视角去分析其本质，就容易理解了。

<!-- $$
\mathbf{x_{\tau_{i-1}}} = \sqrt{\bar{\alpha}_{\tau_{i-1}}} (\frac{1}{\sqrt{\bar{\alpha}_\tau_{i}}} \mathbf{x}_\tau_{i} - \frac{\sqrt{1 - \bar{\alpha}_\tau_{i}}}{\sqrt{\bar{\alpha}_\tau_{i}}} \cdot \epsilon_\theta(\mathbf{x_\tau_{i}}, \tau_{i})) + \sqrt{1 - \bar{\alpha}_{\tau_{i-1}} - \sigma_\tau_{i}^2} \cdot \epsilon_\tau_{i}^{''}
$$ -->

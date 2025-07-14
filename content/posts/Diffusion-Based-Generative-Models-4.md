+++
date = '2025-07-13T14:45:35+08:00'
draft = true
title = 'Diffusion-Based Generative Models <4>: Fokker-Planck方程'
tags = ["diffusion-models", "deep-learning", "generative-AI"]
categories = ["Generative Models"]
mermaid = true
+++

本部分主要介绍Fokker-Planck方程，该方程为Diffusion-Based Generative Models提供了坚实的理论基础。Fokker-Planck方程描述了随机过程中概率密度函数的演化规律，是连接随机微分方程与概率分布的重要桥梁。在扩散模型中，理解Fokker-Planck方程对于分析前向扩散过程和设计反向生成过程至关重要。

本部分主要参考了[Langevin 方程与 Fokker-Planck 方程](https://jiming.site/archives/31/)

## 一. 维纳（Wiener）过程

维纳过程（Wiener Process），也称为布朗运动（Brownian Motion），是随机过程理论中的基础概念，也是理解Fokker-Planck方程的重要前提。

### 1.1 维纳过程的定义

为了理解维纳过程的本质，我们可以从物理现象出发。考虑一个粒子在直线上进行随机运动，这种运动可以通过离散时间步长的随机游走来建模。

设粒子初始位置为 $x=0$，在每个时间步长 $\Delta t$ 内，粒子以相等的概率向左或向右移动距离 $\Delta x$。用 $X(t)$ 表示粒子在时刻 $t$ 的位置，则：

$$X(t) = \sum_{i=1}^{N} \eta_i$$

其中 $N = t/\Delta t$ 是时间步数，$\eta_i$ 是第 $i$ 步的位移：

$$\eta_i = \begin{cases}
+\Delta x, \text{ 概率为 } 1/2 \\
-\Delta x, \text{ 概率为 } 1/2
\end{cases}$$

当时间步数 $N$ 很大时，根据中心极限定理，$X(t)$ 的分布将趋近于正态分布。因此，我们只需要计算其均值和方差就能完全确定分布。

均值的计算：

$$\mathbb{E}[X(t)] = \mathbb{E}\left[\sum_{i=1}^{N} \eta_i\right] = \sum_{i=1}^{N} \mathbb{E}[\eta_i] = N \cdot 0 = 0$$

方差的计算：

$$\text{Var}[X(t)] = \mathbb{E}[X(t)^2] = \mathbb{E}\left[\left(\sum_{i=1}^{N} \eta_i\right)^2\right] = \sum_{i=1}^{N} \mathbb{E}[\eta_i^2] = N \cdot (\Delta x)^2 = \frac{(\Delta x)^2}{\Delta t} \cdot t$$

因此，$X(t) \sim \mathcal{N}(0, \sqrt{(\Delta x)^2 t / \Delta t})$。

维纳过程正是这种离散随机游走在连续时间极限下的结果。当 $\Delta t \to 0$ 且 $\Delta x \to 0$ 时，如果保持 $(\Delta x)^2 / \Delta t = \sigma^2$ 为常数，则得到连续的维纳过程 $W(t) \sim \mathcal{N}(0, \sigma^2 t)$。

基于上述物理直觉，维纳过程 $W(t)$ 的严格数学定义要求满足以下性质：

1. **初始条件**：$W(0) = 0$
2. **独立增量**：对于任意 $0 \leq t_1 < t_2 < \cdots < t_n$，增量 $W(t_2) - W(t_1), W(t_3) - W(t_2), \ldots, W(t_n) - W(t_{n-1})$ 相互独立
3. **正态分布**：对于 $s < t$，增量 $W(t) - W(s) \sim \mathcal{N}(0, \sigma^2(t-s))$
4. **连续路径**：$W(t)$ 的样本路径几乎必然连续

当 $\sigma = 1$ 时，我们得到标准维纳过程（Standard Wiener Process），记为 $B(t)$。标准维纳过程满足：

$$B(t) \sim \mathcal{N}(0, t)$$

<!-- 对于任意 $s < t$，增量 $B(t) - B(s) \sim \mathcal{N}(0, t-s)$。

注意：对于一般的维纳过程 $W(t) \sim \mathcal{N}(0, \sigma^2 t)$，增量 $W(t) - W(s) \sim \mathcal{N}(0, \sigma^2(t-s))$。 -->


### 1.2 数学性质

容易证明，维纳过程 $W(t)$ 具有以下重要性质：

**均值函数**：$\mathbb{E}[W(t)] = 0$

**方差函数**：$\text{Var}[W(t)] = \sigma^2 t$

**自相关函数**：对于 $s \leq t$，$R_W(s,t) = \sigma^2 \min(s,t)$

### 1.3 白噪声/维纳过程的微分形式

为了更深入地理解维纳过程，我们引入白噪声（White Noise）的概念。白噪声 $\xi(t)$ 是维纳过程的广义导数：

$$\xi(t) \overset{\text{formal}}{=} \frac{dW(t)}{dt}$$

但维纳过程 $W(t)$ 几乎必然不可微，因此该导数仅在分布意义下成立。

白噪声 $\xi(t)$ 是一个广义随机过程，具有以下性质：

$$\mathbb{E}[\xi(t)] = 0, \quad \mathbb{E}[\xi(t)\xi(s)] = \delta(t-s)$$

其中 $\delta(t-s)$ 是狄拉克函数。

在形式意义上，我们可以将维纳过程的微分表示为：

$$dW(t) = \xi(t) dt$$

这里 $dW(t)$ 表示维纳过程的微分，具有以下性质：

- $\mathbb{E}[dW(t)] = 0$
- $\mathbb{E}[dW(t)^2] = dt$
- $\mathbb{E}[dW(t) \cdot dt] = 0$

白噪声的引入使得维纳过程的随机性更加直观：在任意时刻 $t$，维纳过程的变化率由白噪声 $\xi(t)$ 决定，而白噪声在不同时刻是相互独立的。

## 二. Itô积分与Fokker-Planck方程

### 2.1 Itô积分

Itô积分是随机微积分中的核心概念。对于确定性函数 $f(t)$ 和维纳过程 $W(t)$，Itô积分定义为：

$$\int_0^t f(s) dW(s) = \lim_{n \to \infty} \sum_{i=1}^n f(t_{i-1}) [W(t_i) - W(t_{i-1})]$$

其中 $0 = t_0 < t_1 < \cdots < t_n = t$ 是区间 $[0,t]$ 的分割，$f(t_{i-1})$ 表示函数 $f$ 在区间 $[t_{i-1}, t_i]$ 左端点的取值。

#### 2.1.1 关键性质推导

**性质1：非预期性**

$$\mathbb{E}\left[\int_0^t f(s) dW(s)\right] = 0$$

**推导**：
$$\begin{aligned}
\mathbb{E}\left[\int_0^t f(s) dW(s)\right] &= \mathbb{E}\left[\lim_{n \to \infty} \sum_{i=1}^n f(t_{i-1}) [W(t_i) - W(t_{i-1})]\right] \\
&= \lim_{n \to \infty} \sum_{i=1}^n f(t_{i-1}) \mathbb{E}[W(t_i) - W(t_{i-1})] \\
&= \lim_{n \to \infty} \sum_{i=1}^n f(t_{i-1}) \cdot 0 = 0
\end{aligned}
$$

其中利用了维纳过程增量的期望为零：$\mathbb{E}[W(t_i) - W(t_{i-1})] = 0$。

**性质2：等距性**

$$\mathbb{E}\left[\left(\int_0^t f(s) dW(s)\right)^2\right] = \sigma^2 \int_0^t f^2(s) ds$$

**推导**：
$$\begin{aligned}
\mathbb{E}\left[\left(\int_0^t f(s) dW(s)\right)^2\right] &= \mathbb{E}\left[\left(\lim_{n \to \infty} \sum_{i=1}^n f(t_{i-1}) [W(t_i) - W(t_{i-1})]\right)^2\right] \\
&= \lim_{n \to \infty} \mathbb{E}\left[\sum_{i=1}^n f^2(t_{i-1}) [W(t_i) - W(t_{i-1})]^2\right] \\
&= \lim_{n \to \infty} \sum_{i=1}^n f^2(t_{i-1}) \mathbb{E}[W(t_i) - W(t_{i-1})]^2 \\
&= \lim_{n \to \infty} \sum_{i=1}^n f^2(t_{i-1}) \sigma^2(t_i - t_{i-1}) \\
&= \sigma^2 \int_0^t f^2(s) ds
\end{aligned}$$

其中利用了维纳过程增量的方差：$\mathbb{E}[W(t_i) - W(t_{i-1})]^2 = \sigma^2(t_i - t_{i-1})$，以及不同区间增量的独立性。

#### 2.1.2 Itô微积分的核心思想

Itô微积分与普通微积分的根本区别在于对二阶项的处理。在普通微积分中，高阶项（如 $(dt)^2$）在极限下趋于零，可以忽略。但在随机微积分中，维纳过程的二阶项与时间的一阶项同阶：

$$(dW(t))^2 = \sigma^2 dt$$

这个关系是Itô微积分的核心。Itô微积分的思想便是保持式中的二阶项 $(dW)^2$，并将其记为一阶项 $\sigma^2 dt$。

这一思想在Itô引理中得到了充分体现：

$$d\phi(X(t)) = \phi'(X(t)) dX(t) + \frac{1}{2} \phi''(X(t)) (dX(t))^2$$

其中 $(dX(t))^2 = g^2(X(t), t) dt$，体现了随机微积分的独特性质。

#### 2.1.3 Itô引理的证明

我们通过泰勒展开来证明Itô引理。对于光滑函数 $\phi(x)$，在 $X(t)$ 处进行泰勒展开：

$$d\phi(X(t)) = \phi'(X(t)) dX(t) + \frac{1}{2} \phi''(X(t)) (dX(t))^2 + \frac{1}{6} \phi'''(X(t)) (dX(t))^3 + \cdots$$

在普通微积分中，高阶项 $(dX(t))^3, (dX(t))^4, \ldots$ 在 $dt \to 0$ 时趋于零。但在随机微积分中，我们需要考虑维纳过程的性质。

对于SDE $dX(t) = f(X(t), t) dt + g(X(t), t) dW(t)$，我们有：

$$
\begin{aligned}
(dX(t))^2 &= [f(X(t), t) dt + g(X(t), t) dW(t)]^2 \\
&= f^2(X(t), t) (dt)^2 + 2f(X(t), t)g(X(t), t) dt \cdot dW(t) + g^2(X(t), t) (dW(t))^2
\end{aligned}
$$

根据Itô微积分的规则：
- $(dt)^2 = 0$（高阶项）：在 $dt \to 0$ 的极限下，$(dt)^2$ 比 $dt$ 更快地趋于零，因此可以忽略
- $dt \cdot dW(t) = 0$（不同阶项）：$dt$ 是确定性的一阶项，$dW(t)$ 是随机的一阶项，它们的乘积在期望意义下为零，且比 $dt$ 更快地趋于零
- $(dW(t))^2 = \sigma^2 dt$（关键关系）：这是维纳过程的核心性质，体现了随机微积分与普通微积分的根本区别

因此：

$$(dX(t))^2 = g^2(X(t), t) dt$$

对于更高阶项，如 $(dX(t))^3$，它们包含 $(dt)^2$ 或更高幂次，在极限下趋于零。

因此，Itô引理为：

$$d\phi(X(t)) = \phi'(X(t)) dX(t) + \frac{1}{2} \phi''(X(t)) g^2(X(t), t) dt$$

这个证明展示了为什么Itô引理中会出现二阶项，以及为什么随机微积分与普通微积分如此不同。

### 2.2 随机微分方程SDE

考虑随机过程 $X(t)$ 满足随机微分方程：

$$dX(t) = f(X(t), t) dt + g(X(t), t) dW(t)$$

其中 $f(X(t), t)$ 是漂移项，$g(X(t), t)$ 是扩散项。

### 2.3 Fokker-Planck方程

Fokker-Planck方程描述了随机过程中概率密度函数的演化规律。对于上述SDE，对应的Fokker-Planck方程为：

$$\frac{\partial p_t(x)}{\partial t} = -\frac{\partial}{\partial x}[f(x,t) p_t(x)] + \frac{1}{2} \frac{\partial^2}{\partial x^2}[g^2(x,t) p_t(x)]$$

其中 $p_t(x)$ 是 $X(t)$ 的概率密度函数。

#### 2.3.1 Fokker-Planck方程的推导

我们从SDE出发推导Fokker-Planck方程。考虑任意光滑函数 $\phi(x)$，计算 $\mathbb{E}[\phi(X(t))]$ 的时间导数：

$$\frac{d}{dt}\mathbb{E}[\phi(X(t))] = \mathbb{E}\left[\frac{d}{dt}\phi(X(t))\right]$$

利用Itô引理：

$$d\phi(X(t)) = \phi'(X(t)) dX(t) + \frac{1}{2}\phi''(X(t)) (dX(t))^2$$

代入SDE: $dX(t) = f(X(t), t) dt + g(X(t), t) dW(t)$ 

$$d\phi(X(t)) = \phi'(X(t))[f(X(t), t) dt + g(X(t), t) dW(t)] + \frac{1}{2}\phi''(X(t)) g^2(X(t), t) dt$$

取期望并利用 $\mathbb{E}[dW(t)] = 0$：

$$\frac{d}{dt}\mathbb{E}[\phi(X(t))] = \mathbb{E}[\phi'(X(t)) f(X(t), t)] + \frac{1}{2}\mathbb{E}[\phi''(X(t)) g^2(X(t), t)]$$

另一方面，利用概率密度函数：

$$\mathbb{E}[\phi(X(t))] = \int \phi(x) p_t(x) dx$$

因此：

$$\frac{d}{dt}\int \phi(x) p_t(x) dx = \int \phi'(x) f(x,t) p_t(x) dx + \frac{1}{2}\int \phi''(x) g^2(x,t) p_t(x) dx$$

对右边进行分部积分：

$$\int \phi'(x) f(x,t) p_t(x) dx = -\int \phi(x) \frac{\partial}{\partial x}[f(x,t) p_t(x)] dx$$

$$\int \phi''(x) g^2(x,t) p_t(x) dx = \int \phi(x) \frac{\partial^2}{\partial x^2}[g^2(x,t) p_t(x)] dx$$

因此：

$$\int \phi(x) \frac{\partial p_t(x)}{\partial t} dx = -\int \phi(x) \frac{\partial}{\partial x}[f(x,t) p_t(x)] dx + \frac{1}{2}\int \phi(x) \frac{\partial^2}{\partial x^2}[g^2(x,t) p_t(x)] dx$$

由于 $\phi(x)$ 是任意的，我们得到Fokker-Planck方程：

$$\frac{\partial p_t(x)}{\partial t} = -\frac{\partial}{\partial x}[f(x,t) p_t(x)] + \frac{1}{2} \frac{\partial^2}{\partial x^2}[g^2(x,t) p_t(x)]$$

#### 2.3.2 方程解释

- 第一项 $-\frac{\partial}{\partial x}[f(x,t) p_t(x)]$ 描述漂移项对概率密度的影响
- 第二项 $\frac{1}{2} \frac{\partial^2}{\partial x^2}[g^2(x,t) p_t(x)]$ 描述扩散项对概率密度的影响

#### 2.3.3 向量形式的Fokker-Planck方程

对于多维随机过程 $\mathbf{X}(t) \in \mathbb{R}^d$，满足向量SDE：

$$d\mathbf{X}(t) = \mathbf{f}(\mathbf{X}(t), t) dt + \mathbf{G}(\mathbf{X}(t), t) d\mathbf{W}(t)$$

对应的向量Fokker-Planck方程为：

$$\frac{\partial p_t(\mathbf{x})}{\partial t} = -\nabla \cdot [\mathbf{f}(\mathbf{x}, t) p_t(\mathbf{x})] + \frac{1}{2} \nabla \cdot [\mathbf{D}(\mathbf{x}, t) \nabla p_t(\mathbf{x})]$$

其中：
- $\nabla \cdot$ 是散度算子
- $\nabla$ 是梯度算子
- $\mathbf{D}(\mathbf{x}, t) = \mathbf{G}(\mathbf{x}, t) \mathbf{G}^T(\mathbf{x}, t)$ 是扩散张量

展开为分量形式：

$$\frac{\partial p_t(\mathbf{x})}{\partial t} = -\sum_{i=1}^d \frac{\partial}{\partial x_i}[f_i(\mathbf{x}, t) p_t(\mathbf{x})] + \frac{1}{2} \sum_{i=1}^d \sum_{j=1}^d \frac{\partial^2}{\partial x_i \partial x_j}\left[\sum_{k=1}^d G_{ik}(\mathbf{x}, t) G_{jk}(\mathbf{x}, t) p_t(\mathbf{x})\right]$$

其中：
- $f_i(\mathbf{x}, t)$ 是漂移向量 $\mathbf{f}(\mathbf{x}, t)$ 的第 $i$ 个分量
- $G_{ik}(\mathbf{x}, t)$ 是扩散矩阵 $\mathbf{G}(\mathbf{x}, t)$ 的第 $(i,k)$ 个元素
- $\sum_{k=1}^d G_{ik}(\mathbf{x}, t) G_{jk}(\mathbf{x}, t)$ 是扩散张量 $\mathbf{D}(\mathbf{x}, t) = \mathbf{G}(\mathbf{x}, t) \mathbf{G}^T(\mathbf{x}, t)$ 的第 $(i,j)$ 个元素

## 三. Fokker-Planck方程与扩散模型

Fokker-Planck方程不仅是一个重要的数学工具，更是理解现代扩散模型的理论基础。本节将详细阐述Fokker-Planck方程如何为扩散模型提供统一的数学框架，以及如何将随机过程理论应用于生成模型。

### 3.1 扩散模型的SDE视角

在ODE/SDE视角下，扩散模型可以统一表示为随机微分方程的形式。这种表示方法将前向扩散过程和反向生成过程都描述为SDE，为理解扩散模型提供了清晰的数学框架。

#### 3.1.1 前向过程（Forward Process）

前向过程描述了数据从原始分布逐渐扩散到噪声分布的过程：

$$d\mathbf{x}(t) = \mathbf{f}(\mathbf{x}, t) dt + g(t) d\mathbf{W}(t)$$

其中：
- $\mathbf{f}(\mathbf{x}, t)$ 是漂移项，控制数据的系统性变化
- $g(t)$ 是扩散系数，控制随机噪声的强度
- $d\mathbf{W}(t)$ 是维纳过程的微分，提供随机性

这个SDE描述了数据点 $\mathbf{x}(t)$ 在时间 $t$ 的演化规律。随着时间推进，数据逐渐被噪声"污染"，最终收敛到噪声分布。

#### 3.1.2 反向过程（Reverse Process）

反向过程是前向过程的逆过程，从噪声分布生成数据：

$$d\mathbf{x}(t) = [\mathbf{f}(\mathbf{x}, t) - g^2(t) \nabla_{\mathbf{x}} \log p_t(\mathbf{x})] dt + g(t) d\mathbf{W}(t)$$

其中关键的一项是得分函数（score function）：
$$\nabla_{\mathbf{x}} \log p_t(\mathbf{x})$$

这个函数描述了在给定时间 $t$ 和位置 $\mathbf{x}$ 时，概率密度的梯度方向。它告诉我们应该向哪个方向移动才能增加数据的概率。

**重要观察**：反向过程比前向过程多了一项 $-g^2(t) \nabla_{\mathbf{x}} \log p_t(\mathbf{x})$，这一项正是基于Fokker-Planck方程推导出来的修正项，确保反向过程能够正确地"逆推"前向过程。

### 3.2 概率流ODE（Probability Flow ODE）

概率流ODE是连接SDE和ODE的重要桥梁，它消除了随机性，提供确定性的采样路径：

$$\frac{d\mathbf{x}(t)}{dt} = \mathbf{f}(\mathbf{x}, t) - \frac{1}{2} g^2(t) \nabla_{\mathbf{x}} \log p_t(\mathbf{x})$$

#### 3.2.1 概率流ODE的性质

1. **概率保持**：ODE轨迹保持与前向SDE相同的概率密度演化
2. **确定性**：消除了随机性，提供确定性采样路径
3. **效率提升**：相比SDE采样，ODE采样更加高效，可以用更大的步长
4. **数值稳定性**：ODE求解器比SDE求解器更稳定

#### 3.2.2 与SDE的关系

概率流ODE可以通过以下方式从SDE得到：

1. 从反向SDE出发：$d\mathbf{x}(t) = [\mathbf{f}(\mathbf{x}, t) - g^2(t) \nabla_{\mathbf{x}} \log p_t(\mathbf{x})] dt + g(t) d\mathbf{W}(t)$
2. 移除随机项：$d\mathbf{x}(t) = [\mathbf{f}(\mathbf{x}, t) - g^2(t) \nabla_{\mathbf{x}} \log p_t(\mathbf{x})] dt$
3. 调整系数：$\frac{d\mathbf{x}(t)}{dt} = \mathbf{f}(\mathbf{x}, t) - \frac{1}{2} g^2(t) \nabla_{\mathbf{x}} \log p_t(\mathbf{x})$

注意系数从 $g^2(t)$ 变为 $\frac{1}{2} g^2(t)$，这是因为ODE中不需要补偿随机项的方差。

### 3.3 得分函数（Score Function）的核心作用

得分函数 $\nabla_{\mathbf{x}} \log p_t(\mathbf{x})$ 是连接SDE和ODE的关键，也是扩散模型的核心组件。

#### 3.3.1 得分函数的含义

得分函数描述了在给定时间 $t$ 和位置 $\mathbf{x}$ 时，概率密度的梯度方向：

$$\nabla_{\mathbf{x}} \log p_t(\mathbf{x}) = \frac{\nabla_{\mathbf{x}} p_t(\mathbf{x})}{p_t(\mathbf{x})}$$

这个函数告诉我们应该向哪个方向移动才能增加数据的概率。在反向过程中，它指导我们如何从噪声中恢复出有意义的数据。

#### 3.3.2 得分函数的估计

在实际应用中，我们通常不知道真实的得分函数，需要通过神经网络来估计：

$$\nabla_{\mathbf{x}} \log p_t(\mathbf{x}) \approx s_\theta(\mathbf{x}, t)$$

其中 $s_\theta(\mathbf{x}, t)$ 是参数为 $\theta$ 的神经网络，称为得分网络（score network）。

#### 3.3.3 训练目标

得分网络的训练目标是最小化得分匹配损失：

$$\mathcal{L}(\theta) = \mathbb{E}_{t, \mathbf{x}(t)} \left[ \| s_\theta(\mathbf{x}(t), t) - \nabla_{\mathbf{x}} \log p_t(\mathbf{x}(t)) \|^2 \right]$$

这个损失函数鼓励网络输出接近真实得分函数的梯度。

### 3.4 扩散模型的统一理论框架

基于Fokker-Planck方程，我们可以建立扩散模型的统一框架，将不同的视角整合在一起：

#### 3.4.1 SDE视角

- **前向SDE**：$d\mathbf{x}(t) = \mathbf{f}(\mathbf{x}, t) dt + g(t) d\mathbf{W}(t)$
- **反向SDE**：$d\mathbf{x}(t) = [\mathbf{f}(\mathbf{x}, t) - g^2(t) \nabla_{\mathbf{x}} \log p_t(\mathbf{x})] dt + g(t) d\mathbf{W}(t)$

#### 3.4.2 ODE视角

- **概率流ODE**：$\frac{d\mathbf{x}(t)}{dt} = \mathbf{f}(\mathbf{x}, t) - \frac{1}{2} g^2(t) \nabla_{\mathbf{x}} \log p_t(\mathbf{x})$

#### 3.4.3 概率视角

- **Fokker-Planck方程**：$\frac{\partial p_t(\mathbf{x})}{\partial t} = -\nabla \cdot [\mathbf{f}(\mathbf{x}, t) p_t(\mathbf{x})] + \frac{1}{2} \nabla \cdot [g^2(t) \nabla p_t(\mathbf{x})]$

#### 3.4.4 三个视角的统一

这三个视角通过得分函数 $\nabla_{\mathbf{x}} \log p_t(\mathbf{x})$ 紧密联系在一起：

1. **SDE → ODE**：通过移除随机项和调整系数
2. **SDE → Fokker-Planck**：通过Itô引理和期望计算
3. **Fokker-Planck → 得分函数**：通过概率密度的梯度

### 3.5 现代扩散模型的应用

这个统一框架为理解现代扩散模型提供了坚实的理论基础：

#### 3.5.1 DDPM（Denoising Diffusion Probabilistic Models）

DDPM可以看作是这个框架的特例，其中：
- $\mathbf{f}(\mathbf{x}, t) = -\frac{1}{2} \beta(t) \mathbf{x}$
- $g(t) = \sqrt{\beta(t)}$
- 得分函数通过噪声预测网络估计

#### 3.5.2 DDIM（Denoising Diffusion Implicit Models）

DDIM是概率流ODE的直接应用，通过确定性采样提高效率。

#### 3.5.3 Score SDE

Score SDE直接基于反向SDE进行采样，是框架的完整实现。

### 3.6 总结

Fokker-Planck方程为扩散模型提供了：

1. **理论基础**：将随机过程理论与生成模型联系起来
2. **统一框架**：SDE、ODE、概率三种视角的统一
3. **实用工具**：为现代扩散模型的设计和实现提供指导
4. **数学严谨性**：确保理论推导的正确性和一致性

这种统一框架不仅帮助我们理解现有的扩散模型，也为设计新的生成模型提供了清晰的思路和工具。





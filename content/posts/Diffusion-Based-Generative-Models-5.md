+++
date = '2025-07-15T21:20:39+08:00'
draft = true
title = 'Diffusion-Based Generative Models <5>: SDE/ODE视角下的扩散模型'
tags = ["diffusion-models", "deep-learning", "generative-AI"]
categories = ["Generative Models"]
mermaid = true
+++


## 一. SDE/ODE

在前面的章节中，我们详细讨论了扩散模型的概率论基础，包括维纳过程、Itô积分、随机微分方程和Fokker-Planck方程。现在我们将这些理论工具应用到扩散模型中，从SDE/ODE的视角来理解扩散过程。

扩散模型的核心思想是将数据分布通过前向过程逐渐转换为噪声分布，然后学习反向过程从噪声恢复数据。这个过程可以用随机微分方程（SDE）来描述，而对应的确定性过程可以用常微分方程（ODE）来描述。

### 1.1 基本形式

<!-- 考虑向量形式的ODE：

$$
d\mathbf{x} = \mathbf{f}(\mathbf{x}, t)dt
$$ -->

考虑一个向量形式的SDE：

$$
d\mathbf{x} = \mathbf{f}(\mathbf{x}, t)dt + \mathbf{G}(t) d\mathbf{W}
$$

其中 $\mathbf{x} \in \mathbb{R}^d$，$\mathbf{f}: \mathbb{R}^d \times [0, T] \to \mathbb{R}^d$ 是漂移函数，$\mathbf{G}: [0, T] \to \mathbb{R}^{d \times d}$ 是扩散矩阵，$\mathbf{W}$ 是 $d$ 维维纳过程。

### 1.2 逆向SDE

对于前向SDE：

$$
d\mathbf{x} = \mathbf{f}(\mathbf{x}, t)dt + \mathbf{G}(t) d\mathbf{W}
$$

其逆向SDE为（逆向SDE的推导基于Girsanov定理和Fokker-Planck方程）：

$$
d\mathbf{x} = \left[\mathbf{f}(\mathbf{x}, t) - \mathbf{G}(t)\mathbf{G}(t)^T \nabla_{\mathbf{x}} \log p_t(\mathbf{x})\right]dt + \mathbf{G}(t) d\bar{\mathbf{W}}
$$

其中 $\bar{\mathbf{W}}$ 是逆向时间维纳过程，$\nabla_{\mathbf{x}} \log p_t(\mathbf{x})$ 是时间 $t$ 时刻状态 $\mathbf{x}$ 的对数概率密度梯度。
<!-- 
#### 1.2.1 逆向SDE的推导

逆向SDE的推导基于以下关键思想：

1. **Girsanov定理**：通过改变测度实现时间反向
2. **Fokker-Planck方程**：描述概率密度的演化
3. **分数函数**：对数概率密度的梯度

**步骤1：Fokker-Planck方程**

前向SDE对应的Fokker-Planck方程为：

$$\frac{\partial p_t(\mathbf{x})}{\partial t} = -\nabla \cdot [\mathbf{f}(\mathbf{x}, t)p_t(\mathbf{x})] + \frac{1}{2}\nabla \cdot [\mathbf{G}(t)\mathbf{G}(t)^T \nabla p_t(\mathbf{x})]$$

**步骤2：概率流分解**

将概率流分解为漂移项和扩散项：

$$\frac{\partial p_t(\mathbf{x})}{\partial t} = -\nabla \cdot [\mathbf{v}(\mathbf{x}, t)p_t(\mathbf{x})]$$

其中 $\mathbf{v}(\mathbf{x}, t)$ 是概率流速度：

$$\mathbf{v}(\mathbf{x}, t) = \mathbf{f}(\mathbf{x}, t) - \frac{1}{2}\mathbf{G}(t)\mathbf{G}(t)^T \nabla_{\mathbf{x}} \log p_t(\mathbf{x})$$

**步骤3：逆向过程**

逆向过程需要满足：

$$\frac{\partial p_t(\mathbf{x})}{\partial t} = \nabla \cdot [\mathbf{v}(\mathbf{x}, t)p_t(\mathbf{x})]$$

这对应于逆向SDE：

$$d\mathbf{x} = \mathbf{v}(\mathbf{x}, t)dt + \mathbf{G}(t)d\bar{\mathbf{W}}$$

#### 1.2.2 分数函数的作用

分数函数 $\nabla_{\mathbf{x}} \log p_t(\mathbf{x})$ 在逆向过程中起到关键作用：

1. **方向指导**：指向概率密度增加的方向
2. **强度控制**：控制去噪的强度
3. **局部信息**：只依赖当前点的局部概率结构

对于高斯分布 $\mathcal{N}(\mathbf{x}; \boldsymbol{\mu}, \boldsymbol{\Sigma})$：

$$\nabla_{\mathbf{x}} \log p(\mathbf{x}) = -\boldsymbol{\Sigma}^{-1}(\mathbf{x} - \boldsymbol{\mu})$$

这表明分数函数指向分布的中心，强度与到中心的距离成正比。 -->

### 1.3 简单SDE分析示例

为了更好地理解SDE和逆向SDE，我们分析一个简单的向量例子：$d\mathbf{x} = \mathbf{A} d\mathbf{W}$，其中 $\mathbf{A}$ 是常数矩阵。

#### 1.3.1 前向过程分析

对于SDE $d\mathbf{x} = \mathbf{A} d\mathbf{W}$：

1. **漂移函数**：$\mathbf{f}(\mathbf{x}, t) = \mathbf{0}$（无漂移）
2. **扩散矩阵**：$\mathbf{G}(t) = \mathbf{A}$（常数扩散矩阵）
3. **初始条件**：假设 $\mathbf{x}(0) = \mathbf{0}$

#### 1.3.2 概率密度演化

对应的Fokker-Planck方程为：

$$\frac{\partial p_t(\mathbf{x})}{\partial t} = \frac{1}{2} \nabla \cdot [\mathbf{A}\mathbf{A}^T \nabla p_t(\mathbf{x})]$$

这是一个多维热传导方程，其解为：

$$p_t(\mathbf{x}) = \frac{1}{\sqrt{(2\pi)^d |\mathbf{A}\mathbf{A}^T| t}} \exp\left(-\frac{1}{2t} \mathbf{x}^T (\mathbf{A}\mathbf{A}^T)^{-1} \mathbf{x}\right)$$

即 $\mathbf{x}(t) \sim \mathcal{N}(\mathbf{0}, \mathbf{A}\mathbf{A}^T t)$，协方差矩阵随时间线性增长。

#### 1.3.3 逆向过程

根据逆向SDE公式：

$$d\mathbf{x} = \left[\mathbf{0} - \mathbf{A}\mathbf{A}^T \nabla_{\mathbf{x}} \log p_t(\mathbf{x})\right]dt + \mathbf{A} d\bar{\mathbf{W}}$$

计算分数函数：

$$\nabla_{\mathbf{x}} \log p_t(\mathbf{x}) = \nabla_{\mathbf{x}} \left[-\frac{1}{2t} \mathbf{x}^T (\mathbf{A}\mathbf{A}^T)^{-1} \mathbf{x} - \frac{1}{2}\log((2\pi)^d |\mathbf{A}\mathbf{A}^T| t)\right] = -\frac{1}{t} (\mathbf{A}\mathbf{A}^T)^{-1} \mathbf{x}$$

因此逆向SDE为：

$$d\mathbf{x} = \frac{1}{t} \mathbf{x} dt + \mathbf{A} d\bar{\mathbf{W}}$$

#### 1.3.4 物理解释

- **前向过程**：从确定性初始状态 $\mathbf{x}(0) = \mathbf{0}$ 开始，通过多维随机游走扩散到多维高斯分布
- **逆向过程**：从多维高斯分布开始，通过漂移项 $\frac{1}{t} \mathbf{x}$ 将分布收缩回原点
- **漂移项作用**：$\frac{1}{t} \mathbf{x}$ 项提供了向原点的"拉力"，抵消了扩散效应

这个向量例子展示了多维SDE如何描述概率分布的演化，以及逆向SDE如何实现分布的收缩。

#### 1.3.5 Python模拟

考虑一个简单的情况，扩散矩阵为单位矩阵的情况，模拟前向和逆向扩散过程的演化过程：

{{< figure src="/pic_diff_5/sde_diffusion_visualization.png" title="">}}
{{< figure src="/pic_diff_5/sde_trajectories.png" title="">}}

## 二. SDE/ODE数值求解

### 2.1 基本方法

#### 2.1.1 ODE求解

对于 $\frac{d\mathbf{x}}{dt} = \mathbf{f}(\mathbf{x}, t)$：

**欧拉方法**：

$$\mathbf{x}_{i+1} = \mathbf{x}_i + h \mathbf{f}(\mathbf{x}_i, t_i)$$

**收敛性分析**：
- 局部截断误差：$O(h^2)$
- 全局截断误差：$O(h)$
- 稳定性：条件稳定，步长需要足够小

**RK4方法**：
$$\begin{aligned}
\mathbf{k}_1 &= \mathbf{f}(\mathbf{x}_i, t_i) \\
\mathbf{k}_2 &= \mathbf{f}(\mathbf{x}_i + \frac{h}{2}\mathbf{k}_1, t_i + \frac{h}{2}) \\
\mathbf{k}_3 &= \mathbf{f}(\mathbf{x}_i + \frac{h}{2}\mathbf{k}_2, t_i + \frac{h}{2}) \\
\mathbf{k}_4 &= \mathbf{f}(\mathbf{x}_i + h\mathbf{k}_3, t_i + h) \\
\mathbf{x}_{i+1} &= \mathbf{x}_i + \frac{h}{6}(\mathbf{k}_1 + 2\mathbf{k}_2 + 2\mathbf{k}_3 + \mathbf{k}_4)
\end{aligned}$$

**收敛性分析**：
- 局部截断误差：$O(h^5)$
- 全局截断误差：$O(h^4)$
- 稳定性：更好的稳定性，适合刚性ODE

**自适应步长方法**：
$$h_{i+1} = h_i \left(\frac{\text{tol}}{\text{error}_i}\right)^{1/p}$$

其中 $\text{tol}$ 是容差，$p$ 是方法的阶数。

#### 2.1.2 SDE求解

对于 $d\mathbf{x} = \mathbf{f}(\mathbf{x}, t)dt + \mathbf{G}(t)d\mathbf{W}$：

**欧拉-丸山方法**：
$$\mathbf{x}_{i+1} = \mathbf{x}_i + \mathbf{f}(\mathbf{x}_i, t_i)h + \mathbf{G}(t_i)\Delta\mathbf{W}_i$$

其中 $\Delta\mathbf{W}_i = \mathbf{W}(t_{i+1}) - \mathbf{W}(t_i) \sim \mathcal{N}(\mathbf{0}, h\mathbf{I})$。

**收敛性分析**：
- 强收敛阶：$O(h^{1/2})$（路径wise收敛）
- 弱收敛阶：$O(h)$（分布收敛）
- 优点：简单易实现
- 缺点：收敛阶较低

**米尔斯坦方法**：
$$\mathbf{x}_{i+1} = \mathbf{x}_i + \mathbf{f}(\mathbf{x}_i, t_i)h + \mathbf{G}(t_i)\Delta\mathbf{W}_i + \frac{1}{2}\mathbf{G}(t_i)\mathbf{G}(t_i)^T[(\Delta\mathbf{W}_i)^2 - h]$$

**收敛性分析**：
- 强收敛阶：$O(h)$
- 弱收敛阶：$O(h)$
- 优点：更高的收敛阶
- 缺点：需要计算二阶项，计算复杂度更高

**随机龙格库塔方法**：
$$\begin{aligned}
\mathbf{K}_1 &= \mathbf{f}(\mathbf{x}_i, t_i)h + \mathbf{G}(t_i)\Delta\mathbf{W}_i \\
\mathbf{K}_2 &= \mathbf{f}(\mathbf{x}_i + \mathbf{K}_1, t_{i+1})h + \mathbf{G}(t_{i+1})\Delta\mathbf{W}_i \\
\mathbf{x}_{i+1} &= \mathbf{x}_i + \frac{1}{2}(\mathbf{K}_1 + \mathbf{K}_2)
\end{aligned}$$

### 2.2 扩散模型应用

#### 2.2.1 前向SDE

$$d\mathbf{x} = -\frac{1}{2}\beta(t)\mathbf{x}dt + \sqrt{\beta(t)}d\mathbf{W}$$

**解析解**：
$$\mathbf{x}(t) = \mathbf{x}(0)e^{-\frac{1}{2}\int_0^t \beta(s)ds} + \int_0^t e^{-\frac{1}{2}\int_s^t \beta(\tau)d\tau}\sqrt{\beta(s)}d\mathbf{W}(s)$$

**数值实现**：
对于线性SDE，可以使用精确的解析解，避免数值误差。

#### 2.2.2 逆向SDE

$$d\mathbf{x} = \left[\mathbf{f}(\mathbf{x}, t) - \mathbf{G}(t)\mathbf{G}(t)^T s_\theta(\mathbf{x}, t)\right]dt + \mathbf{G}(t)d\bar{\mathbf{W}}$$

**关键考虑**：
- **时间反向**：$t=T \to t=0$
- **分数函数估计**：$s_\theta(\mathbf{x}, t)$ 需要准确估计
- **逆向维纳过程**：$\bar{\mathbf{W}}$

**数值挑战**：
1. **分数函数精度**：网络估计的分数函数存在误差
2. **时间反向**：需要处理逆向时间过程
3. **随机性控制**：需要平衡确定性和随机性

#### 2.2.3 概率流ODE

$$\frac{d\mathbf{x}}{dt} = \mathbf{f}(\mathbf{x}, t) - \frac{1}{2}\mathbf{G}(t)\mathbf{G}(t)^T s_\theta(\mathbf{x}, t)$$

**优势**：
- 确定性过程，避免随机性
- 可以使用高阶ODE求解器
- 计算效率更高

**挑战**：
- 失去随机性，可能影响生成质量
- 对分数函数精度要求更高

### 2.3 误差与稳定性

#### 2.3.1 误差来源

1. **截断误差**：数值方法的近似误差
   - 与步长 $h$ 相关
   - 可以通过高阶方法减少

2. **舍入误差**：浮点精度限制
   - 累积误差问题
   - 需要数值稳定性分析

3. **模型误差**：分数函数估计
   - 网络训练误差
   - 泛化误差

4. **采样误差**：随机性影响
   - 有限样本效应
   - 可以通过多次采样减少

#### 2.3.2 误差传播

$$\|\mathbf{x}(t) - \hat{\mathbf{x}}(t)\| \leq \|\mathbf{x}(0) - \hat{\mathbf{x}}(0)\|e^{Lt} + \frac{M}{L}(e^{Lt} - 1)$$

其中 $L$ 是Lipschitz常数，$M$ 是局部误差界。

**误差控制策略**：
1. **自适应步长**：根据局部误差调整步长
2. **高阶方法**：使用更高阶的数值方法
3. **误差估计**：实时监控误差大小

#### 2.3.3 稳定性

- **A稳定性**：对于线性测试方程 $y' = \lambda y$，当 $\text{Re}(\lambda) < 0$ 时解收敛
- **L稳定性**：更强的稳定性条件
- **随机稳定性**：考虑随机项的稳定性

**稳定性分析**：
对于扩散模型中的线性SDE：
$$d\mathbf{x} = -\frac{1}{2}\beta(t)\mathbf{x}dt + \sqrt{\beta(t)}d\mathbf{W}$$

稳定性条件要求 $\beta(t) > 0$，确保漂移项提供收缩力。

### 2.4 实现优化

#### 2.4.1 步长策略

- **固定步长**：简单实现，但可能效率不高
- **自适应步长**：根据误差自动调整，提高效率
- **多尺度方法**：不同时间尺度使用不同步长

**自适应步长算法**：
```python
def adaptive_step_size(error, tol, order):
    return h * (tol / error) ** (1.0 / order)
```

#### 2.4.2 计算效率

- **并行计算**：多个轨迹并行采样
- **GPU加速**：利用GPU并行计算能力
- **内存管理**：避免存储完整轨迹

**优化策略**：
1. **批处理**：同时处理多个样本
2. **内存复用**：重用中间计算结果
3. **缓存机制**：缓存重复计算的结果

#### 2.4.3 数值稳定性

- **条件数控制**：避免病态问题
- **正则化**：添加正则化项提高稳定性
- **梯度裁剪**：防止梯度爆炸

**稳定性技巧**：
1. **归一化**：保持数值在合理范围内
2. **约束**：添加物理约束
3. **平滑化**：使用平滑的激活函数

### 2.5 高级技术

#### 2.5.1 概率流ODE

$$\frac{d\mathbf{x}}{dt} = \mathbf{f}(\mathbf{x}, t) - \frac{1}{2}\mathbf{G}(t)\mathbf{G}(t)^T s_\theta(\mathbf{x}, t)$$

- **高阶方法**：RK4、自适应方法
- **隐式方法**：处理刚性ODE
- **辛方法**：保持几何结构

**辛方法优势**：
- 保持哈密顿结构
- 长期数值稳定性
- 能量守恒

#### 2.5.2 混合方法

- **确定性部分**：使用ODE求解器
- **随机部分**：使用SDE求解器
- **自适应切换**：根据问题特性选择方法

**混合策略**：
1. **早期阶段**：使用SDE，保持随机性
2. **后期阶段**：切换到ODE，提高效率
3. **条件切换**：根据误差大小决定

#### 2.5.3 加速技术

**DDIM采样**：
确定性采样的高效变体，支持快速采样。

**DPM-Solver**：
高阶求解器，减少采样步数。

**PNDM**：
伪数值方法，平衡质量和速度。

### 2.6 总结

数值求解是扩散模型实现的关键：

1. **方法选择**：精度与效率平衡
2. **误差控制**：多源误差管理
3. **稳定性保证**：数值稳定性
4. **效率优化**：计算资源利用

**实际建议**：
- 对于简单问题，使用欧拉方法
- 对于精度要求高的问题，使用RK4或自适应方法
- 对于大规模问题，考虑并行和GPU加速
- 对于稳定性要求高的问题，使用隐式方法

## 三. SMLD和SDE的关系

在前面的章节中，我们讨论了SDE/ODE视角下的扩散模型理论。现在我们将这些理论与实际应用中的Score Matching with Langevin Dynamics (SMLD)方法联系起来，展示离散时间过程如何自然地过渡到连续时间SDE。

### 3.1 SMLD的基本思想

Score Matching with Langevin Dynamics (SMLD) 是一种基于分数函数的生成模型方法，其核心思想是：

1. **分数学习**：学习数据分布的对数概率密度梯度
2. **朗之万采样**：使用朗之万动力学进行采样
3. **噪声调度**：通过逐步添加噪声实现前向过程

#### 3.1.1 分数函数定义

对于数据分布 $p_{\text{data}}(\mathbf{x})$，分数函数定义为：

$$\nabla_{\mathbf{x}} \log p_{\text{data}}(\mathbf{x})$$

这个函数描述了在给定点 $\mathbf{x}$ 处，概率密度增加最快的方向。

**物理意义**：
- **方向信息**：指向概率密度增加的方向
- **强度信息**：梯度大小反映概率变化的快慢
- **局部特性**：只依赖当前点的局部概率结构

**数学性质**：
1. **尺度不变性**：对概率密度的单调变换保持不变
2. **线性性**：$\nabla_{\mathbf{x}} \log [p_1(\mathbf{x})p_2(\mathbf{x})] = \nabla_{\mathbf{x}} \log p_1(\mathbf{x}) + \nabla_{\mathbf{x}} \log p_2(\mathbf{x})$
3. **链式法则**：$\nabla_{\mathbf{x}} \log p(f(\mathbf{x})) = \nabla_{\mathbf{x}} f(\mathbf{x}) \cdot \nabla_{f(\mathbf{x})} \log p(f(\mathbf{x}))$

#### 3.1.2 朗之万动力学

朗之万动力学是一种随机采样方法，其更新规则为：

$$\mathbf{x}_{t+1} = \mathbf{x}_t + \epsilon \nabla_{\mathbf{x}} \log p(\mathbf{x}_t) + \sqrt{2\epsilon} \boldsymbol{\eta}_t$$

其中 $\epsilon$ 是步长，$\boldsymbol{\eta}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ 是高斯噪声。

**物理背景**：
朗之万动力学起源于统计物理学，描述粒子在势场中的随机运动：
- **漂移项**：$\epsilon \nabla_{\mathbf{x}} \log p(\mathbf{x}_t)$ 对应势场的梯度力
- **扩散项**：$\sqrt{2\epsilon} \boldsymbol{\eta}_t$ 对应热噪声

**收敛性质**：
- **稳态分布**：在适当条件下，采样轨迹收敛到目标分布
- **混合时间**：与步长 $\epsilon$ 和分布复杂度相关
- **遍历性**：长时间平均等于空间平均

### 3.2 离散时间SMLD过程

#### 3.2.1 前向过程（噪声添加）

SMLD的前向过程通过逐步添加噪声实现：

$$\mathbf{x}_i = \mathbf{x}_{i-1} + \sqrt{\beta_i} \boldsymbol{\epsilon}_i$$

其中 $\beta_i$ 是噪声调度，$\boldsymbol{\epsilon}_i \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$。

**噪声调度设计**：
1. **线性调度**：$\beta_i = \beta_0 + i \cdot \Delta\beta$
2. **指数调度**：$\beta_i = \beta_0 \cdot \alpha^i$
3. **余弦调度**：$\beta_i = \beta_0 \cos^2(\frac{\pi i}{2N})$

**设计原则**：
- **渐进性**：噪声逐渐增加，避免突变
- **充分性**：最终噪声足够大，覆盖整个空间
- **效率性**：在有限步数内完成转换

#### 3.2.2 逆向过程（去噪）

逆向过程使用学习的分数函数进行去噪：

$$\mathbf{x}_{i-1} = \mathbf{x}_i - \beta_i s_\theta(\mathbf{x}_i, i) + \sqrt{\beta_i} \boldsymbol{\eta}_i$$

其中 $s_\theta(\mathbf{x}_i, i)$ 是学习的分数函数。

**关键考虑**：
1. **分数函数精度**：$s_\theta(\mathbf{x}_i, i)$ 需要准确估计 $\nabla_{\mathbf{x}} \log p_i(\mathbf{x}_i)$
2. **噪声控制**：$\sqrt{\beta_i} \boldsymbol{\eta}_i$ 提供随机性，避免确定性轨迹
3. **步长选择**：$\beta_i$ 需要与噪声调度匹配

### 3.3 从离散到连续：SDE极限

#### 3.3.1 离散过程的数学形式

将离散时间过程写为差分方程：

$$\Delta \mathbf{x}_i = \mathbf{x}_i - \mathbf{x}_{i-1} = \sqrt{\beta_i} \boldsymbol{\epsilon}_i$$

#### 3.3.2 连续时间极限

当时间步长 $\Delta t \to 0$ 时，离散过程收敛到连续SDE：

$$d\mathbf{x} = \mathbf{f}(\mathbf{x}, t)dt + \mathbf{G}(t)d\mathbf{W}$$

其中：
- $\mathbf{f}(\mathbf{x}, t) = \mathbf{0}$（无漂移）
- $\mathbf{G}(t) = \sqrt{\beta(t)}\mathbf{I}$（扩散矩阵）

**收敛性分析**：
- **强收敛**：路径wise收敛到连续过程
- **弱收敛**：分布收敛到连续过程的分布
- **收敛阶**：通常为 $O(\Delta t^{1/2})$

#### 3.3.3 噪声调度的连续化

离散噪声调度 $\{\beta_i\}_{i=1}^N$ 对应连续噪声调度 $\beta(t)$：

$$\beta(t) = \lim_{\Delta t \to 0} \frac{\beta_i}{\Delta t}$$

**连续化策略**：
1. **线性插值**：$\beta(t) = \beta_i + (t - t_i)\frac{\beta_{i+1} - \beta_i}{\Delta t}$
2. **平滑插值**：使用样条函数等平滑方法
3. **解析形式**：直接设计连续函数 $\beta(t)$

### 3.4 SMLD与SDE的对应关系

#### 3.4.1 前向过程对应

| SMLD离散过程 | SDE连续过程 |
|-------------|------------|
| $\mathbf{x}_i = \mathbf{x}_{i-1} + \sqrt{\beta_i} \boldsymbol{\epsilon}_i$ | $d\mathbf{x} = \sqrt{\beta(t)}d\mathbf{W}$ |
| 噪声调度 $\{\beta_i\}$ | 噪声调度 $\beta(t)$ |
| 高斯噪声 $\boldsymbol{\epsilon}_i$ | 维纳过程 $d\mathbf{W}$ |

#### 3.4.2 逆向过程对应

| SMLD离散过程 | SDE连续过程 |
|-------------|------------|
| $\mathbf{x}_{i-1} = \mathbf{x}_i - \beta_i s_\theta(\mathbf{x}_i, i) + \sqrt{\beta_i} \boldsymbol{\eta}_i$ | $d\mathbf{x} = [-\beta(t)s_\theta(\mathbf{x}, t)]dt + \sqrt{\beta(t)}d\bar{\mathbf{W}}$ |
| 分数函数 $s_\theta(\mathbf{x}_i, i)$ | 分数函数 $s_\theta(\mathbf{x}, t)$ |
| 步长 $\beta_i$ | 时间微分 $dt$ |

**对应关系的重要性**：
1. **理论统一**：离散和连续过程在理论上是等价的
2. **实现指导**：连续理论指导离散实现
3. **分析工具**：可以使用连续理论分析离散过程

### 3.5 概率流ODE视角

#### 3.5.1 确定性过程

除了随机SDE过程，我们还可以构造确定性的概率流ODE：

$$\frac{d\mathbf{x}}{dt} = \mathbf{f}(\mathbf{x}, t) - \frac{1}{2}\mathbf{G}(t)\mathbf{G}(t)^T s_\theta(\mathbf{x}, t)$$

对于SMLD对应的SDE：

$$\frac{d\mathbf{x}}{dt} = -\frac{1}{2}\beta(t)s_\theta(\mathbf{x}, t)$$

**ODE的优势**：
1. **确定性**：每次采样得到相同结果
2. **高效性**：可以使用高阶ODE求解器
3. **可控性**：更容易控制采样过程

**ODE的挑战**：
1. **随机性缺失**：可能影响生成多样性
2. **精度要求**：对分数函数精度要求更高
3. **稳定性**：需要更稳定的数值方法

#### 3.5.2 ODE与SDE的关系

- **SDE**：包含随机项，产生多样化的采样轨迹
- **ODE**：确定性过程，产生唯一的采样轨迹
- **关系**：ODE是SDE的期望轨迹

**选择策略**：
- **多样性优先**：选择SDE采样
- **效率优先**：选择ODE采样
- **混合策略**：结合两种方法的优点

### 3.6 实际实现考虑

#### 3.6.1 离散化策略

将连续SDE离散化回离散过程：

$$\mathbf{x}_{i-1} = \mathbf{x}_i - \Delta t \cdot \frac{1}{2}\beta(t_i)s_\theta(\mathbf{x}_i, t_i) + \sqrt{\Delta t \cdot \beta(t_i)} \boldsymbol{\eta}_i$$

**离散化方法**：
1. **欧拉离散化**：最简单的离散化方法
2. **高阶离散化**：使用更高阶的数值方法
3. **自适应离散化**：根据误差自动调整步长

#### 3.6.2 时间嵌入

分数网络需要时间信息：

$$s_\theta(\mathbf{x}, t) = \text{Network}(\mathbf{x}, \text{TimeEmbedding}(t))$$

**时间嵌入方法**：
1. **正弦嵌入**：$\text{TimeEmbedding}(t) = [\sin(\omega_1 t), \cos(\omega_1 t), \ldots, \sin(\omega_d t), \cos(\omega_d t)]$
2. **位置编码**：类似Transformer的位置编码
3. **可学习嵌入**：直接学习时间表示

**嵌入设计考虑**：
- **周期性**：时间嵌入应该具有适当的周期性
- **分辨率**：不同频率捕获不同时间尺度
- **维度**：嵌入维度影响表达能力

#### 3.6.3 噪声调度设计

常见的噪声调度函数：

1. **线性调度**：$\beta(t) = \beta_0 + (\beta_T - \beta_0)t$
   - 优点：简单直观
   - 缺点：可能不够平滑

2. **余弦调度**：$\beta(t) = \beta_0 \cos^2(\frac{\pi t}{2T})$
   - 优点：平滑，在边界处导数小
   - 缺点：计算复杂度稍高

3. **指数调度**：$\beta(t) = \beta_0 e^{\alpha t}$
   - 优点：快速增长
   - 缺点：可能增长过快

**调度选择原则**：
- **数据特性**：根据数据分布特性选择
- **计算资源**：考虑计算复杂度
- **经验调优**：通过实验确定最佳调度

### 3.7 理论优势

#### 3.7.1 统一框架

SDE/ODE视角提供了统一的理论框架：

- **离散SMLD**：实际实现的基础
- **连续SDE**：理论分析的工具
- **概率流ODE**：高效采样的选择

**框架优势**：
1. **理论完备性**：基于坚实的数学基础
2. **实现灵活性**：支持多种实现方式
3. **分析工具丰富**：可以使用多种分析工具

#### 3.7.2 灵活性

- **采样策略**：可以选择SDE或ODE采样
- **时间控制**：可以精确控制采样步数
- **条件生成**：支持各种条件生成任务

**灵活性体现**：
1. **多尺度采样**：不同时间尺度使用不同策略
2. **自适应控制**：根据生成质量调整参数
3. **条件扩展**：容易扩展到条件生成

### 3.8 总结

SMLD和SDE的关系展示了离散时间过程如何自然地过渡到连续时间过程：

1. **理论基础**：SDE提供了坚实的数学基础
2. **实现指导**：离散SMLD提供了实际实现方案
3. **灵活性**：支持多种采样策略和优化方法
4. **扩展性**：为更复杂的生成任务提供框架

**实际应用建议**：
- **理论研究**：使用连续SDE进行分析
- **实际实现**：使用离散SMLD进行实现
- **高效采样**：使用概率流ODE进行快速采样
- **条件生成**：结合条件信息扩展框架

这种对应关系不仅帮助我们理解扩散模型的本质，也为实际应用提供了重要的指导。

## 四. 扩散模型的统一框架

在前面的章节中，我们分别讨论了SDE/ODE理论、数值求解方法以及SMLD与SDE的关系。现在我们将这些内容整合起来，构建一个统一的扩散模型理论框架，展示不同视角和方法之间的内在联系。

### 4.1 统一框架概述

扩散模型的统一框架基于以下核心思想：

1. **概率流视角**：将生成过程视为概率分布的演化
2. **分数函数核心**：以分数函数为桥梁连接不同方法
3. **时间连续性**：从离散时间过程到连续时间过程的自然过渡
4. **采样灵活性**：支持多种采样策略和优化方法

#### 4.1.1 框架组成

统一框架包含以下主要组成部分：

- **理论基础**：SDE/ODE理论、Fokker-Planck方程、分数函数理论
- **数值方法**：ODE/SDE求解器、自适应方法、加速技术
- **实现策略**：离散SMLD、连续SDE、概率流ODE
- **优化技术**：噪声调度、时间嵌入、网络架构

#### 4.1.2 框架优势

1. **理论完备性**：基于坚实的数学基础
2. **实现灵活性**：支持多种实现方式
3. **分析工具丰富**：可以使用多种分析工具
4. **扩展性强**：容易扩展到新的应用场景

### 4.2 核心数学框架

#### 4.2.1 前向过程

统一的前向SDE：

$$d\mathbf{x} = \mathbf{f}(\mathbf{x}, t)dt + \mathbf{G}(t)d\mathbf{W}$$

其中：
- $\mathbf{f}(\mathbf{x}, t)$：漂移函数，描述确定性演化
- $\mathbf{G}(t)$：扩散矩阵，描述随机扰动
- $\mathbf{W}$：维纳过程，提供随机性

**特殊形式**：
对于标准扩散模型：
$$\mathbf{f}(\mathbf{x}, t) = -\frac{1}{2}\beta(t)\mathbf{x}, \quad \mathbf{G}(t) = \sqrt{\beta(t)}\mathbf{I}$$

#### 4.2.2 逆向过程

统一的逆向SDE：

$$d\mathbf{x} = \left[\mathbf{f}(\mathbf{x}, t) - \mathbf{G}(t)\mathbf{G}(t)^T s_\theta(\mathbf{x}, t)\right]dt + \mathbf{G}(t)d\bar{\mathbf{W}}$$

其中 $s_\theta(\mathbf{x}, t)$ 是学习的分数函数。

**关键洞察**：
- 逆向过程包含原始漂移项
- 分数函数提供额外的收缩力
- 随机项保持采样多样性

#### 4.2.3 概率流ODE

确定性版本：

$$\frac{d\mathbf{x}}{dt} = \mathbf{f}(\mathbf{x}, t) - \frac{1}{2}\mathbf{G}(t)\mathbf{G}(t)^T s_\theta(\mathbf{x}, t)$$

**ODE的优势**：
- 确定性采样
- 高效数值求解
- 可控的生成过程

### 4.3 不同视角的统一

#### 4.3.1 离散时间视角

SMLD的离散时间过程：

$$\mathbf{x}_{i-1} = \mathbf{x}_i - \beta_i s_\theta(\mathbf{x}_i, i) + \sqrt{\beta_i} \boldsymbol{\eta}_i$$

**与连续时间的关系**：
- 离散过程是连续SDE的数值近似
- 步长 $\beta_i$ 对应时间微分 $dt$
- 噪声 $\boldsymbol{\eta}_i$ 对应维纳过程增量

#### 4.3.2 连续时间视角

SDE的连续时间过程：

$$d\mathbf{x} = -\frac{1}{2}\beta(t)s_\theta(\mathbf{x}, t)dt + \sqrt{\beta(t)}d\bar{\mathbf{W}}$$

**与离散时间的关系**：
- 连续SDE是离散过程的极限
- 时间 $t$ 对应步数 $i$
- 噪声调度 $\beta(t)$ 对应离散调度 $\{\beta_i\}$

#### 4.3.3 概率流视角

ODE的确定性过程：

$$\frac{d\mathbf{x}}{dt} = -\frac{1}{2}\beta(t)s_\theta(\mathbf{x}, t)$$

**与其他视角的关系**：
- ODE是SDE的期望轨迹
- 确定性过程，无随机性
- 高效但可能缺乏多样性

### 4.4 实现策略的统一

#### 4.4.1 训练策略

统一的训练目标：

$$\mathcal{L} = \mathbb{E}_{t, \mathbf{x}(t)} \left[\|\mathbf{s}_\theta(\mathbf{x}(t), t) - \nabla_{\mathbf{x}} \log p_t(\mathbf{x}(t))\|^2\right]$$

**训练考虑**：
1. **时间采样**：从 $[0, T]$ 中随机采样时间点
2. **数据采样**：从前向过程采样 $\mathbf{x}(t)$
3. **目标计算**：计算真实的分数函数

#### 4.4.2 采样策略

多种采样方法：

1. **SDE采样**：使用逆向SDE
   - 优点：保持随机性，生成质量高
   - 缺点：计算复杂度高

2. **ODE采样**：使用概率流ODE
   - 优点：高效，确定性
   - 缺点：可能缺乏多样性

3. **混合采样**：结合SDE和ODE
   - 优点：平衡质量和效率
   - 缺点：实现复杂

#### 4.4.3 优化技术

1. **噪声调度优化**：
   - 线性调度：简单但可能不够平滑
   - 余弦调度：平滑，边界处导数小
   - 自适应调度：根据数据特性调整

2. **时间嵌入优化**：
   - 正弦嵌入：周期性，多尺度
   - 位置编码：类似Transformer
   - 可学习嵌入：直接优化

3. **网络架构优化**：
   - U-Net：适合图像生成
   - Transformer：适合序列数据
   - 混合架构：结合不同优势

### 4.5 与其他生成模型的比较

#### 4.5.1 与GAN的比较

| 特性 | 扩散模型 | GAN |
|------|----------|-----|
| 训练稳定性 | 稳定 | 不稳定 |
| 模式崩塌 | 无 | 常见 |
| 采样质量 | 高 | 可变 |
| 计算效率 | 低 | 高 |
| 理论基础 | 坚实 | 相对薄弱 |

**优势**：
- 训练稳定性好
- 理论基础坚实
- 生成质量高

**劣势**：
- 采样速度慢
- 计算复杂度高

#### 4.5.2 与VAE的比较

| 特性 | 扩散模型 | VAE |
|------|----------|-----|
| 生成质量 | 高 | 中等 |
| 采样速度 | 慢 | 快 |
| 潜在空间 | 无 | 有 |
| 重构质量 | 高 | 中等 |
| 理论基础 | 坚实 | 坚实 |

**优势**：
- 生成质量高
- 重构质量好
- 理论完备

**劣势**：
- 采样速度慢
- 无显式潜在空间

#### 4.5.3 与Flow模型的比较

| 特性 | 扩散模型 | Flow模型 |
|------|----------|----------|
| 可逆性 | 不可逆 | 可逆 |
| 采样速度 | 慢 | 快 |
| 计算复杂度 | 高 | 中等 |
| 表达能力 | 强 | 受架构限制 |
| 训练稳定性 | 稳定 | 相对稳定 |

**优势**：
- 表达能力更强
- 训练更稳定
- 理论更完备

**劣势**：
- 采样速度慢
- 不可逆

### 4.6 实际应用案例

#### 4.6.1 图像生成

**应用场景**：
- 高分辨率图像生成
- 条件图像生成
- 图像编辑和修复

**技术特点**：
- 使用U-Net架构
- 多尺度噪声调度
- 条件嵌入技术

**成功案例**：
- DDPM：首次展示高质量图像生成
- DDIM：快速确定性采样
- Stable Diffusion：大规模文本到图像生成

#### 4.6.2 音频生成

**应用场景**：
- 音乐生成
- 语音合成
- 音频编辑

**技术特点**：
- 使用Transformer架构
- 时间序列建模
- 多模态融合

**成功案例**：
- AudioCraft：高质量音频生成
- MusicLM：文本到音乐生成
- AudioLDM：音频扩散模型

#### 4.6.3 文本生成

**应用场景**：
- 文本续写
- 风格转换
- 对话生成

**技术特点**：
- 使用Transformer架构
- 离散化处理
- 条件生成

**成功案例**：
- Diffusion-LM：离散扩散语言模型
- CDCD：连续扩散对话生成
- DiffuSeq：序列扩散模型

### 4.7 理论扩展

#### 4.7.1 条件生成

条件扩散模型：

$$d\mathbf{x} = \mathbf{f}(\mathbf{x}, \mathbf{c}, t)dt + \mathbf{G}(t)d\mathbf{W}$$

其中 $\mathbf{c}$ 是条件信息。

**条件嵌入方法**：
1. **交叉注意力**：条件与状态交互
2. **特征融合**：直接特征拼接
3. **调制技术**：条件调制网络参数

#### 4.7.2 多模态生成

多模态扩散模型：

$$d\mathbf{x}_1 = \mathbf{f}_1(\mathbf{x}_1, \mathbf{x}_2, t)dt + \mathbf{G}_1(t)d\mathbf{W}_1$$
$$d\mathbf{x}_2 = \mathbf{f}_2(\mathbf{x}_1, \mathbf{x}_2, t)dt + \mathbf{G}_2(t)d\mathbf{W}_2$$

**多模态融合**：
- 交叉模态注意力
- 共享潜在空间
- 模态特定编码器

#### 4.7.3 可控生成

可控扩散模型：

$$d\mathbf{x} = \mathbf{f}(\mathbf{x}, \mathbf{a}, t)dt + \mathbf{G}(t)d\mathbf{W}$$

其中 $\mathbf{a}$ 是控制信号。

**控制方法**：
- 梯度引导
- 分类器引导
- 能量函数引导

### 4.8 未来发展方向

#### 4.8.1 理论发展

1. **更高效的采样**：
   - 减少采样步数
   - 提高采样质量
   - 自适应采样策略

2. **更好的理论理解**：
   - 收敛性分析
   - 误差界估计
   - 稳定性理论

3. **新的数学工具**：
   - 几何方法
   - 最优传输理论
   - 信息几何

#### 4.8.2 技术发展

1. **架构创新**：
   - 更高效的网络架构
   - 注意力机制优化
   - 多尺度建模

2. **训练优化**：
   - 更稳定的训练策略
   - 更好的损失函数
   - 自适应学习率

3. **应用扩展**：
   - 更多模态支持
   - 更大规模应用
   - 实时应用

#### 4.8.3 应用发展

1. **创意应用**：
   - 艺术创作
   - 设计辅助
   - 内容生成

2. **科学应用**：
   - 分子设计
   - 材料发现
   - 药物设计

3. **工业应用**：
   - 产品设计
   - 质量控制
   - 预测建模

### 4.9 总结

扩散模型的统一框架提供了一个强大而灵活的理论基础：

1. **理论完备性**：基于坚实的数学基础，包括SDE/ODE理论、概率论和数值分析
2. **实现灵活性**：支持多种实现策略，从离散SMLD到连续SDE再到概率流ODE
3. **应用广泛性**：适用于图像、音频、文本等多种模态的生成任务
4. **扩展性强**：容易扩展到条件生成、多模态生成、可控生成等新应用

**核心价值**：
- **统一理解**：将不同方法统一在一个理论框架下
- **实践指导**：为实际应用提供清晰的指导原则
- **发展基础**：为未来的理论和技术发展提供基础

**实际建议**：
- **理论研究**：深入理解SDE/ODE理论和分数函数理论
- **实践应用**：根据具体需求选择合适的实现策略
- **持续学习**：关注最新的理论和技术发展

这个统一框架不仅帮助我们理解扩散模型的本质，也为实际应用和未来发展提供了重要的指导。随着理论和技术的不断发展，扩散模型将在更多领域发挥重要作用。

## 附录：Python实现示例

为了更好地理解扩散模型的核心概念，我们提供一个完整的Python实现示例，展示SDE/ODE视角下的扩散模型。

### A.1 基础设置

```python
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy.integrate import odeint
import seaborn as sns

# 设置随机种子
np.random.seed(42)
torch.manual_seed(42)

# 设置绘图样式
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
```

### A.2 噪声调度函数

```python
def linear_beta_schedule(t, beta_start=0.0001, beta_end=0.02):
    """线性噪声调度"""
    return beta_start + (beta_end - beta_start) * t

def cosine_beta_schedule(t, beta_start=0.0001, beta_end=0.02):
    """余弦噪声调度"""
    return beta_start + (beta_end - beta_start) * np.cos(np.pi * t / 2) ** 2

def exponential_beta_schedule(t, beta_start=0.0001, beta_end=0.02):
    """指数噪声调度"""
    return beta_start * np.exp(np.log(beta_end / beta_start) * t)

# 可视化不同的噪声调度
t = np.linspace(0, 1, 1000)
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(t, linear_beta_schedule(t))
plt.title('Linear Schedule')
plt.xlabel('Time')
plt.ylabel('β(t)')

plt.subplot(1, 3, 2)
plt.plot(t, cosine_beta_schedule(t))
plt.title('Cosine Schedule')
plt.xlabel('Time')

plt.subplot(1, 3, 3)
plt.plot(t, exponential_beta_schedule(t))
plt.title('Exponential Schedule')
plt.xlabel('Time')

plt.tight_layout()
plt.show()
```

### A.3 前向过程实现

```python
class ForwardProcess:
    """前向扩散过程"""
    
    def __init__(self, beta_schedule='linear', T=1000):
        self.T = T
        self.t = np.linspace(0, 1, T)
        
        if beta_schedule == 'linear':
            self.beta = linear_beta_schedule(self.t)
        elif beta_schedule == 'cosine':
            self.beta = cosine_beta_schedule(self.t)
        elif beta_schedule == 'exponential':
            self.beta = exponential_beta_schedule(self.t)
        
        # 预计算累积量
        self.alpha = 1 - self.beta
        self.alpha_bar = np.cumprod(self.alpha)
        self.sqrt_alpha_bar = np.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = np.sqrt(1 - self.alpha_bar)
    
    def sample_forward(self, x0, t):
        """从x0采样xt"""
        if isinstance(t, int):
            t = np.array([t])
        
        noise = np.random.randn(*x0.shape)
        sqrt_alpha_bar_t = self.sqrt_alpha_bar[t]
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bar[t]
        
        xt = sqrt_alpha_bar_t.reshape(-1, 1) * x0 + sqrt_one_minus_alpha_bar_t.reshape(-1, 1) * noise
        return xt, noise
    
    def get_xt_distribution(self, x0, t):
        """获取xt的分布参数"""
        mean = self.sqrt_alpha_bar[t] * x0
        std = self.sqrt_one_minus_alpha_bar[t]
        return mean, std

# 测试前向过程
forward_process = ForwardProcess(beta_schedule='cosine', T=1000)

# 生成一些测试数据
x0 = np.random.randn(100, 2)  # 2D数据
t_steps = [0, 100, 500, 999]

plt.figure(figsize=(12, 3))
for i, t in enumerate(t_steps):
    xt, _ = forward_process.sample_forward(x0, t)
    
    plt.subplot(1, 4, i+1)
    plt.scatter(xt[:, 0], xt[:, 1], alpha=0.6)
    plt.title(f't = {t}')
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)

plt.tight_layout()
plt.show()
```

### A.4 分数网络实现

```python
class ScoreNetwork(nn.Module):
    """分数网络"""
    
    def __init__(self, input_dim=2, hidden_dim=128, time_dim=64):
        super().__init__()
        
        # 时间嵌入
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        
        # 主网络
        self.net = nn.Sequential(
            nn.Linear(input_dim + time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x, t):
        # 时间嵌入
        t_emb = self.time_embed(t.unsqueeze(-1))
        
        # 拼接输入
        x_t = torch.cat([x, t_emb], dim=-1)
        
        # 前向传播
        score = self.net(x_t)
        return score

# 测试分数网络
score_net = ScoreNetwork()
x = torch.randn(10, 2)
t = torch.rand(10)
score = score_net(x, t)
print(f"Score shape: {score.shape}")
```

### A.5 训练函数

```python
def train_score_network(score_net, forward_process, data, epochs=1000, lr=1e-3):
    """训练分数网络"""
    
    optimizer = optim.Adam(score_net.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    losses = []
    
    for epoch in range(epochs):
        # 随机采样时间步
        t = np.random.randint(0, forward_process.T, size=data.shape[0])
        
        # 前向过程采样
        xt, noise = forward_process.sample_forward(data, t)
        
        # 转换为tensor
        xt_tensor = torch.FloatTensor(xt)
        t_tensor = torch.FloatTensor(t / forward_process.T)  # 归一化到[0,1]
        noise_tensor = torch.FloatTensor(noise)
        
        # 预测分数
        predicted_score = score_net(xt_tensor, t_tensor)
        
        # 真实分数（对于高斯噪声，分数就是噪声的负值）
        true_score = -noise_tensor / forward_process.sqrt_one_minus_alpha_bar[t].reshape(-1, 1)
        
        # 计算损失
        loss = criterion(predicted_score, true_score)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
    
    return losses

# 生成训练数据（混合高斯分布）
def generate_mixture_gaussian_data(n_samples=1000):
    """生成混合高斯分布数据"""
    n_components = 3
    means = np.array([[1, 1], [-1, -1], [0, 2]])
    covs = [np.eye(2) * 0.1 for _ in range(n_components)]
    
    data = []
    for _ in range(n_samples):
        component = np.random.randint(0, n_components)
        sample = np.random.multivariate_normal(means[component], covs[component])
        data.append(sample)
    
    return np.array(data)

# 训练模型
data = generate_mixture_gaussian_data(1000)
score_net = ScoreNetwork()
losses = train_score_network(score_net, forward_process, data, epochs=1000)

# 绘制训练损失
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

# 可视化训练数据
plt.subplot(1, 2, 2)
plt.scatter(data[:, 0], data[:, 1], alpha=0.6)
plt.title('Training Data')
plt.tight_layout()
plt.show()
```

### A.6 采样函数

```python
def sample_sde(score_net, forward_process, n_samples=100, n_steps=1000):
    """使用SDE采样"""
    
    # 从噪声开始
    x = torch.randn(n_samples, 2)
    
    # 时间步长
    dt = 1.0 / n_steps
    
    samples = []
    
    for i in range(n_steps):
        t = 1.0 - i * dt  # 反向时间
        t_tensor = torch.full((n_samples,), t)
        
        # 计算分数
        score = score_net(x, t_tensor)
        
        # SDE更新
        beta_t = forward_process.beta[int(t * (forward_process.T - 1))]
        
        # 漂移项
        drift = -0.5 * beta_t * score * dt
        
        # 扩散项
        diffusion = torch.sqrt(torch.tensor(beta_t * dt)) * torch.randn_like(x)
        
        # 更新x
        x = x + drift + diffusion
        
        if i % 100 == 0:
            samples.append(x.detach().numpy())
    
    return samples

def sample_ode(score_net, forward_process, n_samples=100, n_steps=1000):
    """使用ODE采样"""
    
    # 从噪声开始
    x = torch.randn(n_samples, 2)
    
    # 时间步长
    dt = 1.0 / n_steps
    
    samples = []
    
    for i in range(n_steps):
        t = 1.0 - i * dt  # 反向时间
        t_tensor = torch.full((n_samples,), t)
        
        # 计算分数
        score = score_net(x, t_tensor)
        
        # ODE更新（无随机项）
        beta_t = forward_process.beta[int(t * (forward_process.T - 1))]
        x = x - 0.5 * beta_t * score * dt
        
        if i % 100 == 0:
            samples.append(x.detach().numpy())
    
    return samples

# 采样
sde_samples = sample_sde(score_net, forward_process, n_samples=500, n_steps=1000)
ode_samples = sample_ode(score_net, forward_process, n_samples=500, n_steps=1000)

# 可视化采样结果
plt.figure(figsize=(15, 5))

# 原始数据
plt.subplot(1, 3, 1)
plt.scatter(data[:, 0], data[:, 1], alpha=0.6, label='Original Data')
plt.title('Original Data')
plt.legend()

# SDE采样
plt.subplot(1, 3, 2)
plt.scatter(sde_samples[-1][:, 0], sde_samples[-1][:, 1], alpha=0.6, label='SDE Samples')
plt.title('SDE Sampling')
plt.legend()

# ODE采样
plt.subplot(1, 3, 3)
plt.scatter(ode_samples[-1][:, 0], ode_samples[-1][:, 1], alpha=0.6, label='ODE Samples')
plt.title('ODE Sampling')
plt.legend()

plt.tight_layout()
plt.show()
```

### A.7 采样过程可视化

```python
def visualize_sampling_process(samples, title):
    """可视化采样过程"""
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    # 选择不同的时间步
    time_steps = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    
    for i, step in enumerate(time_steps):
        if step < len(samples):
            axes[i].scatter(samples[step][:, 0], samples[step][:, 1], alpha=0.6)
            axes[i].set_title(f'Step {step * 100}')
            axes[i].set_xlim(-3, 3)
            axes[i].set_ylim(-3, 3)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

# 可视化SDE和ODE采样过程
visualize_sampling_process(sde_samples, 'SDE Sampling Process')
visualize_sampling_process(ode_samples, 'ODE Sampling Process')
```

### A.8 性能分析

```python
def analyze_sampling_quality(original_data, samples, method_name):
    """分析采样质量"""
    
    # 计算分布统计
    original_mean = np.mean(original_data, axis=0)
    original_std = np.std(original_data, axis=0)
    
    sample_mean = np.mean(samples, axis=0)
    sample_std = np.std(samples, axis=0)
    
    # 计算KL散度（简化版本）
    def kl_divergence(p, q):
        return np.sum(p * np.log(p / q + 1e-10))
    
    # 计算分布相似度
    mean_diff = np.linalg.norm(original_mean - sample_mean)
    std_diff = np.linalg.norm(original_std - sample_std)
    
    print(f"\n{method_name} Analysis:")
    print(f"Mean difference: {mean_diff:.4f}")
    print(f"Std difference: {std_diff:.4f}")
    print(f"Original mean: {original_mean}")
    print(f"Sample mean: {sample_mean}")
    print(f"Original std: {original_std}")
    print(f"Sample std: {sample_std}")

# 分析采样质量
analyze_sampling_quality(data, sde_samples[-1], "SDE")
analyze_sampling_quality(data, ode_samples[-1], "ODE")
```

### A.9 总结

这个Python实现展示了扩散模型的核心概念：

1. **前向过程**：通过噪声调度逐步添加噪声
2. **分数学习**：训练网络预测分数函数
3. **逆向采样**：使用SDE或ODE进行采样
4. **质量评估**：分析生成样本的质量

**关键观察**：
- SDE采样保持随机性，生成多样化样本
- ODE采样更高效，但可能缺乏多样性
- 分数网络的学习质量直接影响生成质量
- 噪声调度的选择影响训练和采样效果

这个实现为理解扩散模型提供了实用的代码基础，可以进一步扩展到更复杂的应用场景。






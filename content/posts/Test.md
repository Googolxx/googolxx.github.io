+++
date = '2025-05-04T00:27:21+08:00'
draft = false
title = 'Test'
+++

### 正文
如果我们能够知道真实的后验分布 $p(z|X)$，似乎就能解决这个问题：
$$
\begin{equation}
\begin{aligned}
p_{\theta}(x) &= \frac{p_{\theta}(X, z)}{p(z|X)} \\
    &= \frac{P_{\theta}(X|z)p(z)}{p(z|X)}
\end{aligned}
\end{equation}
$$
(注意这里可能会有歧义，我用 $p(z|X)$替代了 $p_{\theta}(z|X)$，但最后都会优化掉)。因为 $ p(z) \sim \mathcal{N}(z|0, I) $，而 $p_{\theta}(X|z)$ 就是Decoder所生成的分布，如果知道真实后验分布 $p(z|x)$，那我们也可以直接优化目标函数。但核心 $p(z|x)$ 是untracble的 （当然更严谨一点讲，也可以用hybird MC等方式来逼近，但就不在这里的讨论范畴了）。

于是在VAE中，我们可以用变分贝叶斯，引入一个Encoder，生成 $ q_{\phi}(z|X) \sim \mathcal{N}(z|\mu(X;\phi), \sigma(X;\phi)I)$ 来逼近真实后验分布 $p(z|X)$ （类似地，这里协方差矩阵也为对角矩阵）。重新推导目标函数：
$$
\begin{equation}
\begin{aligned}
\log p_{\theta}(X) &= \log p_{\theta}(X) \int q_{\phi}(z|X) dz \\
    &= \int q_{\phi}(z|X) \log p_{\theta}(X) dz \\ 
    &= \int q_{\phi}(z|X) \log \frac{p_{\theta}(X,z)}{p(z|X)}  \\ 
    &= \int q_{\phi}(z|X) \log \frac{p_{\theta}(X,z)}{q_{\phi}(z|X)} \cdot \frac{q_{\phi}(z|X)}{p(z|X)} dz \\
    &= \int q_{\phi}(z|X) \log \frac{p_{\theta}(X,z)}{q_{\phi}(z|X)} dz +  \int q_{\phi}(z|X) \log \frac{q_{\phi}(z|X)}{p(z|X)} dz \\
    &= \mathbb{E}_{q_{\phi}(z|X)} \left[ \log \frac{p_{\theta}(X,z)}{q_{\phi}(z|X)} \right] + D_{\text{KL}}(q_{\phi}(z|X) \| p(z|X)) \\
    &\geq \mathbb{E}_{q_{\phi}(z|X)} \left[ \log \frac{p_{\theta}(X,z)}{q_{\phi}(z|X)} \right]
\end{aligned}
\end{equation}
$$

因为真实后验分布 $p(z|X)$ 没有解析解，且KL散度这一项 $ D_{\text{KL}}(q_{\phi}(z|X) \| p(z|X)) $ 始终是大于0的，因此目标优化函数可以改为最大化 第一项。在变分贝叶斯方法中，这个损失函数被称为**变分下界或证据下界（variational lower bound, or evidence lower bound）** ：
$$
\begin{equation}
\begin{aligned}
\mathbb{E}_{q_{\phi}(z|X)} \left[ \log \frac{p_{\theta}(X,z)}{q_{\phi}(z|X)} \right] &= \log p_{\theta}(X) - D_{\text{KL}}(q_{\phi}(z|X) \| p(z|X))
\end{aligned}
\end{equation}
$$

不难发现，最大化变分下界 等价于最大化 $p_{\theta}(x)$ 并最小化 $D_{\text{KL}}(q_{\phi}(z|X) \| p(z|X))$，这2个目标都恰恰是我们希望优化的。那么来计算下该变分下界的解析解：
$$
\begin{equation}
\begin{aligned}
\mathbb{E}_{q_{\phi}(z|X)} \left[ \log \frac{p_{\theta}(X,z)}{q_{\phi}(z|X)} \right] &= \mathbb{E}_{q_{\phi}(z|X)} \left[ \log \frac{p_{\theta}(X｜z)p(z)}{q_{\phi}(z|X)} \right] \\
    &= \mathbb{E}_{q_{\phi}(z|X)} \left[ \log p_{\theta}(X|z)\right] - D_{\text{KL}}(q_{\phi}(z|X) \| p(z))
\end{aligned}
\end{equation}
$$

完美，第一项的解析解在前面我们已经算过了，通过MC采样我们可以近似求出其解析解，区别在于隐变量 $z$ 之前是从 $p(z) \sim \mathcal{N}(z|0, I) $ 中采样，而现在是从 $q_{\phi}(z|X) \sim \mathcal{N}(z|\mu(X;\phi), \sigma(X;\phi)I) $ 中采样；而第二项中，2个高斯分布间的KL散度也可以直接算出来解析解：
$$
\begin{equation}
\begin{aligned}
D_{\text{KL}}\left(\mathcal{N}(\mu_0, \Sigma_0) \parallel \mathcal{N}(\mu_1, \Sigma_1)\right) = \frac{1}{2} \left[
\operatorname{tr}(\Sigma_1^{-1}\Sigma_0) + (\mu_1 - \mu_0)^\top \Sigma_1^{-1}(\mu_1 - \mu_0) - k + \log \frac{\det \Sigma_1}{\det \Sigma_{0}} \right]
\end{aligned}
\end{equation}
$$

最后一步，让我们来计算优化目标最后的解析解。第一项，我们假设 $p_{\theta}(X|z)$ 的协方差为全为 $\frac{1}{2}$ 的对角矩阵；第二项，设隐变量 $z$ 维度为 $d$带入上述KL散度的解析解。优化目标的最终形式可以表示为：
$$
\begin{equation}
\begin{aligned}
\mathbb{E}_{q_{\phi}(z|X)} \left[ \log \frac{p_{\theta}(X,z)}{q_{\phi}(z|X)} \right] &= \mathbb{E}_{q_{\phi}(z|X)} \left[ \log \frac{p_{\theta}(X｜z)p(z)}{q_{\phi}(z|X)} \right] \\
    &= \mathbb{E}_{q_{\phi}(z|X)} \left[ \log p_{\theta}(X|z)\right] - D_{\text{KL}}(q_{\phi}(z|X) \| p(z)) \\
    &\propto \frac{1}{m} \sum_{i=0}^{m} \sum_{k=0}^{K} \left( x^{(k)} - f(z_i;\theta)^{(k)} \right)^{2} - \frac{1}{2} \sum_{j=0}^{d} \left( -1 +  \mu^{j^{2}}(X;\phi) + \sigma^{j^{2}}(X;\phi) -\log \sigma^{j^{2}}(X;\phi) \right) \\ 
\end{aligned}
\end{equation}
$$

好，得到目标损失函数了，可以训练VAE模型看看效果了，注意这里还是只对 $q_{\phi}(z|X) \sim \mathcal{N}(z|\mu(X;\phi), \sigma(X;\phi)I) $ 采样一次，因为我们引入Encoder构建约束关系后，采样出来的隐变量 $z$ 大部分都是有意义的，所以也不太担心会严重的模式坍塌现象了。类似的，在CelebA上，我训练了一个VAE模型，以下是损失曲线和生成图像的可视化结果：
{{< figure src="/pic_vae/vae_mse.png" title="">}}
{{< figure src="/pic_vae/vae_kld.png" title="">}}
{{< figure src="/pic_vae/vae_recons.png" title="">}}

可以发现，无论是第一项的重建损失项还是第二项拟合真实后验分布的 $q_{\phi}(z|X)$ 和先验分布 $p(z)$ 间的KL散度，都是正常收敛的状态。可视化的结果也还不错，至少比较多样，不会崩塌到“平均脸”。但还有一个问题值得注意，因为我们优化的是变分下界，$D_{\text{KL}}(q_{\phi}(z|X) \| p(z|X))$ 这一项可能会引入一些误差，但这个误差项又很难直观表现出来，这也是很多后续工作优化/讨论的地方。

其实关于VAE，还有很多值得讨论研究的点。比如最终优化目标的重建损失和KL散度，其实KL散度更像是正则项，重建损失和KL散度之间构成一个trade-off关系：更具体地来说，我们虽然会假设 $p_{\theta}(X|z)$ 的协方差矩阵为常数，但实际采样过程中，并不会再加一个高斯噪声来模拟，而是直接取均值，那如果这里对协方差矩阵的假设为一个与可控常数 $ \beta$ 相关的对角矩阵，最终优化目标可以（不严谨地）等价为以下形式：
$$
\begin{equation}
\begin{aligned}
\beta \cdot \mathbb{E}_{q_{\phi}(z|X)} \left[ \log p_{\theta}(X|z)\right] - D_{\text{KL}}(q_{\phi}(z|X) \| p(z)) 
\end{aligned}
\end{equation}
$$
那就等价于Beta-VAE了，当然Beta-VAE的出发点不同，是将等式重写为KKT条件下的拉格朗日形式。

VAE还有很多有意思的点，待后续学习&更新吧。
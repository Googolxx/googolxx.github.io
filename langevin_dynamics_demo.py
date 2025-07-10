import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def create_mixture_gaussian():
    """创建混合高斯分布"""
    # 定义两个高斯分布的参数
    mu1 = np.array([-2, -2])
    mu2 = np.array([3, 3])
    sigma1 = np.array([[1, 0.5], [0.5, 1]])
    sigma2 = np.array([[1.5, -0.3], [-0.3, 1.5]])
    
    # 混合权重
    w1, w2 = 0.6, 0.4
    
    return mu1, mu2, sigma1, sigma2, w1, w2

def log_probability_density(x, mu1, mu2, sigma1, sigma2, w1, w2):
    """计算对数概率密度"""
    # 计算两个高斯分布的概率密度
    p1 = multivariate_normal.pdf(x, mean=mu1, cov=sigma1)
    p2 = multivariate_normal.pdf(x, mean=mu2, cov=sigma2)
    
    # 混合分布的概率密度
    p_mixture = w1 * p1 + w2 * p2
    
    # 避免log(0)的情况
    return np.log(p_mixture + 1e-10)

def gradient_log_probability_analytical(x, mu1, mu2, sigma1, sigma2, w1, w2):
    """解析计算对数概率密度的梯度"""
    # 计算两个高斯分布的概率密度
    p1 = multivariate_normal.pdf(x, mean=mu1, cov=sigma1)
    p2 = multivariate_normal.pdf(x, mean=mu2, cov=sigma2)
    
    # 混合分布的概率密度
    p_mixture = w1 * p1 + w2 * p2
    
    # 计算梯度（解析形式）
    # 对于高斯分布 N(μ, Σ)，log p(x) = -0.5 * (x-μ)^T Σ^(-1) (x-μ) + C
    # 所以 ∇ log p(x) = -Σ^(-1) (x-μ)
    
    # 第一个分布的梯度
    grad1 = -np.linalg.solve(sigma1, x - mu1)
    # 第二个分布的梯度
    grad2 = -np.linalg.solve(sigma2, x - mu2)
    
    # 混合分布的梯度（加权平均）
    grad_mixture = (w1 * p1 * grad1 + w2 * p2 * grad2) / (p_mixture + 1e-10)
    
    return grad_mixture

def langevin_dynamics_sampling(initial_points, mu1, mu2, sigma1, sigma2, w1, w2, 
                              tau=0.01, T=1000, burn_in=100):
    """朗之万动力学采样"""
    samples = []
    trajectories = []
    
    for init_point in initial_points:
        x = init_point.copy()
        trajectory = [x.copy()]
        
        for t in range(T):
            # 计算梯度
            grad = gradient_log_probability_analytical(x, mu1, mu2, sigma1, sigma2, w1, w2)
            
            # 生成噪声
            noise = np.random.normal(0, 1, 2)
            
            # 朗之万动力学更新
            x = x + tau * grad + np.sqrt(2 * tau) * noise
            
            trajectory.append(x.copy())
        
        trajectories.append(trajectory)
        # 只保留burn-in后的样本
        samples.extend(trajectory[burn_in:])
    
    return np.array(samples), trajectories

def plot_results(mu1, mu2, sigma1, sigma2, w1, w2, samples, initial_points, trajectories):
    """绘制结果"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 创建网格用于绘制真实分布
    x = np.linspace(-6, 8, 100)
    y = np.linspace(-6, 8, 100)
    X, Y = np.meshgrid(x, y)
    positions = np.vstack([X.ravel(), Y.ravel()]).T
    
    # 计算真实分布
    true_density = np.zeros(positions.shape[0])
    for i, pos in enumerate(positions):
        true_density[i] = np.exp(log_probability_density(pos, mu1, mu2, sigma1, sigma2, w1, w2))
    
    true_density = true_density.reshape(X.shape)
    
    # 左上图：真实分布和采样点
    axes[0,0].contour(X, Y, true_density, levels=20, alpha=0.7, colors='blue')
    axes[0,0].scatter(samples[:, 0], samples[:, 1], alpha=0.6, s=1, color='red', label='采样点')
    axes[0,0].scatter(initial_points[:, 0], initial_points[:, 1], color='green', s=50, label='初始点')
    axes[0,0].set_title('朗之万动力学采样结果')
    axes[0,0].set_xlabel('x')
    axes[0,0].set_ylabel('y')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # 右上图：采样点分布直方图
    axes[0,1].hist2d(samples[:, 0], samples[:, 1], bins=50, cmap='Blues', alpha=0.8)
    axes[0,1].set_title('采样点分布直方图')
    axes[0,1].set_xlabel('x')
    axes[0,1].set_ylabel('y')
    axes[0,1].grid(True, alpha=0.3)
    
    # 左下图：采样轨迹
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for i, trajectory in enumerate(trajectories):
        trajectory = np.array(trajectory)
        axes[1,0].plot(trajectory[:, 0], trajectory[:, 1], color=colors[i], alpha=0.7, linewidth=0.5)
        axes[1,0].scatter(trajectory[0, 0], trajectory[0, 1], color=colors[i], s=50, marker='o')
        axes[1,0].scatter(trajectory[-1, 0], trajectory[-1, 1], color=colors[i], s=50, marker='s')
    
    axes[1,0].contour(X, Y, true_density, levels=20, alpha=0.5, colors='black')
    axes[1,0].set_title('采样轨迹')
    axes[1,0].set_xlabel('x')
    axes[1,0].set_ylabel('y')
    axes[1,0].grid(True, alpha=0.3)
    
    # 右下图：能量函数随时间变化
    energies = []
    for trajectory in trajectories:
        traj_energies = [-log_probability_density(x, mu1, mu2, sigma1, sigma2, w1, w2) for x in trajectory]
        energies.append(traj_energies)
    
    for i, energy in enumerate(energies):
        axes[1,1].plot(energy, color=colors[i], alpha=0.7, label=f'轨迹 {i+1}')
    
    axes[1,1].set_title('能量函数随时间变化')
    axes[1,1].set_xlabel('迭代次数')
    axes[1,1].set_ylabel('能量 (-log p(x))')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('langevin_dynamics_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("图像已保存为 'langevin_dynamics_results.png'")

def main():
    """主函数"""
    print("开始朗之万动力学采样演示...")
    
    # 创建混合高斯分布
    mu1, mu2, sigma1, sigma2, w1, w2 = create_mixture_gaussian()
    
    # 设置初始点（从不同位置开始）
    initial_points = np.array([
        [-5, -5],  # 远离分布的点
        [0, 0],    # 分布中间的点
        [5, 5],    # 另一个远离分布的点
        [-1, 2],   # 随机点
        [4, -1]    # 随机点
    ])
    
    # 朗之万动力学采样参数
    tau = 0.01      # 步长
    T = 2000        # 总迭代次数
    burn_in = 200   # 预热期
    
    print(f"采样参数: τ={tau}, T={T}, burn_in={burn_in}")
    print(f"混合高斯分布参数:")
    print(f"  分布1: μ={mu1}, 权重={w1}")
    print(f"  分布2: μ={mu2}, 权重={w2}")
    
    # 执行采样
    samples, trajectories = langevin_dynamics_sampling(initial_points, mu1, mu2, sigma1, sigma2, w1, w2, 
                                                     tau, T, burn_in)
    
    print(f"采样完成，共获得 {len(samples)} 个样本")
    
    # 绘制结果
    plot_results(mu1, mu2, sigma1, sigma2, w1, w2, samples, initial_points, trajectories)
    
    # 计算一些统计信息
    print(f"\n采样统计信息:")
    print(f"  样本均值: {np.mean(samples, axis=0)}")
    print(f"  样本标准差: {np.std(samples, axis=0)}")
    print(f"  样本范围: x∈[{np.min(samples[:, 0]):.2f}, {np.max(samples[:, 0]):.2f}], "
          f"y∈[{np.min(samples[:, 1]):.2f}, {np.max(samples[:, 1]):.2f}]")

if __name__ == "__main__":
    main() 
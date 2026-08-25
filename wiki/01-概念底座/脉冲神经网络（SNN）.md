# 脉冲神经网络（SNN）

**脉冲神经网络**（Spiking Neural Networks, SNN）是一类用离散脉冲（spike）模拟神经元发放的神经网络，属于[[类脑计算与 SNN]]主题下的核心模型。

## 核心机制

- 神经元在膜电位超过阈值时发放脉冲（0/1 离散信号）
- 发放行为通常用[[阶跃函数（Heaviside）]]建模
- 时间维度上的脉冲序列携带信息，而非传统 ANN 的连续激活值

## 训练难点

SNN 的前向传播保留离散发放特性，但[[阶跃函数（Heaviside）]]在阈值处不可微，导致[[反向传播（Backpropagation）]]无法直接穿透发放节点。典型解法见[[代理梯度（Surrogate Gradient）]]，降维对照见[[SNN 训练中的代理梯度]]。

## 来源

- [[sources/2026-08-21-snn-surrogate-gradient]]（raw: `raw/2026-08-21-snn-surrogate-gradient.md`）

## 相关链接

- [[阶跃函数（Heaviside）]]
- [[代理梯度（Surrogate Gradient）]]
- [[反向传播（Backpropagation）]]
- [[SNN 训练中的代理梯度]]
- [[类脑计算与 SNN]]

## 苏格拉底追问

> 生物神经元的发放真的是硬阈值阶跃吗？若真实发放机制带有随机性或连续累积过程，SNN 的前向模型应如何修正，又是否仍需要[[代理梯度（Surrogate Gradient）]]这类技巧？

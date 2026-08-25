# 脉冲神经网络（SNN）中的代理梯度算法（Surrogate Gradient）

- Date: 2026-08-21
- Category: 神经网络 / 类脑计算

## Raw Notes
脉冲神经网络（Spiking Neural Networks, SNN）使用阶跃函数（Heaviside step function）模拟神经元的脉冲发放。但阶跃函数在阈值处的导数为无穷大，其余地方导数为 0，导致传统反向传播（Backpropagation）无法计算导数，面临严重的“不可微/梯度消失”问题。

为了训练深层 SNN，代理梯度（Surrogate Gradient）方法在前向传播时保留阶跃函数的离散 0/1 发放特性；而在反向传播时，用连续可微的平滑函数（如 Arctan 或 Sigmoid 的导数）替代原阶跃函数的导数，从而让梯度顺畅穿透不可微节点，实现端到端的参数更新。
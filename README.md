# DA-TransResUNet: 基于深度感知 Transformer-Residual 网络的测井地层自动对比

[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

本仓库是论文 **"Automated Stratigraphic Correlation from Well Logs Using a Depth-Aware Transformer-Residual Network"** 的官方 PyTorch 实现。

---

## 📖 简介 (Introduction)

地层对比（Stratigraphic correlation）是油气勘探开发中的基础任务。然而，受旋回沉积影响导致的**“同层异名（Lithological Homonymy）”**现象，以及深部井段的非平稳噪声，严重制约了自动化对比的精度。

**DA-TransResUNet** 提出了一种新型的深度感知混合神经网络框架。我们将空间位置信息（归一化深度）作为硬约束融入特征提取与 CRF 推理的全过程中，打破了岩性特征在空间上的对称性。结合 **SE-ResNet** 的局部纹理感知和 **Transformer** 的长程旋回建模能力，本模型在包含 1170 口井的工业级数据集上实现了 **0.9851** 的 F1 分数，并在 294 口独立盲测井上取得了 **0.982** 的宏平均 F1 分数。

## ✨ 核心亮点 (Highlights)

* **深度感知机制 (Depth-Aware Mechanism)**: 显式注入深度特征，并构建深度先验掩码 (Depth Prior Mask)，从物理层面消除“同层异名”误判，实现 0 跨代错配。
* **微观与宏观的双流提取**: SE-ResNet 捕捉局部岩性纹理（动态调整测井曲线权重），Transformer 瓶颈层捕捉宏观沉积旋回模式。
* **物理约束的自适应后处理**: 随深度动态调整的过滤规则，有效压制深部高频噪声，防止过分割。

## 🏗️ 模型架构 (Architecture)

<p align="center">
  <img src="moxing.png" alt="DA-TransResUNet 架构图" width="600px">
</p>

---

## ⚙️ 安装与环境配置 (Installation)

1. **克隆本仓库**：
   ```bash
   git clone [https://github.com/YourUsername/DA-TransResUNet.git](https://github.com/YourUsername/DA-TransResUNet.git)
   cd DA-TransResUNet

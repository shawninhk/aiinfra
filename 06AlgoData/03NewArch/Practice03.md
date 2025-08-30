<!--Copyright © ZOMI 适用于[License](https://github.com/Infrasys-AI/AIInfra)版权许可-->
RetNet 循环/分块/FlashAttention 混合训练实战

​​内容​​：

∙
实现 RetNet 的三种计算模式切换逻辑（训练时全连接，推理时分块）

∙
对比不同模式在 LLaMA-2 7B 上的训练吞吐量（Tokens/sec）

∙
在 MSMARCO 数据集上测试长上下文检索任务的准确率

∙
可视化混合模式下的梯度传播路径差异
<!--Copyright © ZOMI 适用于[License](https://github.com/Infrasys-AI/AIInfra)版权许可-->

# 08. PP 流水并行原理

> Author by：高亮

接下来将深入解析​​流水线并行（Pipeline Parallelism, PP）​​的核心原理与优化策略。从最基础的朴素流水并行开始，阐述其前向和反向传播中数据在多个设备间传递的工作方式，并引出其核心性能瓶颈——​​空泡（Bubble）​​。之后会重点介绍​​空泡率的​​计算，分析如何通过增大 micro-batch 来降低空泡、提升设备利用率。随后，会探讨Google提出的​​GPipe​​如何通过​​前向与反向交错执行​​的调度策略来高效利用硬件。

## 朴素流水并行原理

朴素流水并行原理相对简单。如图所示，四种不同的色块代表不同的 rank（或 GPU）。

![pipeline原理](./images/09pipeline00.png)

* 在前向传播过程中：rank0 将本地计算得到的激活值传递给 rank1(即图中的F0传递过程)，rank1 再传递给 rank2，直至 rank3 完成前向流水线的最后一步。
* 在反向传播过程中：rank3 首先计算本阶段的梯度并将其传递给 rank2（同理，图中的B0传递过程），依次传递回 rank0，最终得到各个 rank 的梯度。
* 在完成一次前向与反向的流水线传输后，每个 rank 的权重参数完成更新，至此完成一轮迭代训练。

## Bubble 空泡率

在上一小节我们已经说明了朴素流水并行（PP）的执行流程：模型被切成 $p$ 个 stage，micro-batch 依次在各个 rank 上做前向，再按相反方向回传梯度完成反向。如何量化这种流水线的“忙闲程度”？我们需要一个能刻画“等待/空转”开销的指标——这就是 **并行空泡（Parallelism Bubble）** 与其占比 **Bubble Ratio**。

> 直观理解：流水线并非一开始就“满负荷”运转。前期要把各个 stage 依次“灌满”（warmup），尾部还要把在途的 micro-batch“排空”（cooldown）。这两段头尾空转时间合起来，就是 Bubble 时间；其在总迭代时间中的占比，就是 Bubble Ratio。

### Bubble 时间（头尾空转时长）

$$
t_{\text{bubble}} \;=\; (p-1)\,\bigl(t_f + t_b\bigr)
$$

符号与物理含义：

* $p$：流水线并行度（stage/pipe 的数量，亦即 GPU 串接的段数）。
* $m$：micro-batch 数量（将一个 global batch 切成的份数，决定流水线上可“塞”多少小包）。
* $t_f$：单个 micro-batch 的前向时间（在某一 stage 上完成该 micro-batch 的前向计算所需时间）。
* $t_b$：单个 micro-batch 的反向时间（在某一 stage 上完成该 micro-batch 的反向计算所需时间）。

> 解释：空泡时间为什么是 $(p-1)(t_f+t_b)$？
把时间线想象成由多个“时隙”（slot）组成：前向 slot 长约 $t_f$，反向 slot 长约 $t_b$。流水线并行过程分为两个阶段：warmup合cooldown。其中，warmup要让第一个 micro-batch 抵达最后一段（rank $p{-}1$）才能出现稳定并行，此前需要额外 $(p{-}1)$ 个前向 slot。cooldown：最后一个 micro-batch 的反向结果要回传到 rank 0，尾部还需 $(p{-}1)$ 个反向 slot。将两端合并，记为一组 $(t_f{+}t_b)$ 的“头尾配对开销”，因此得到 $(p{-}1)(t_f{+}t_b)$。

### 理想迭代时间（不计空泡的纯计算时长）

$$
t_{\text{ideal}} \;=\; m\,\bigl(t_f + t_b\bigr)
$$

这里假设已经进入稳定阶段，流水线每“推进”一个 slot，都会有一个 micro-batch 的前向或反向在各个 stage 上同时进行。

### Bubble 占有率（空泡率）

空泡在总迭代时间中的比例为Bubble radio：

$$
\mathit{bubble\ ratio}
\;=\;
\frac{t_{\text{bubble}}}{\,t_{\text{bubble}} + t_{\text{ideal}}\,}
\;=\;
\frac{(p-1)(t_f+t_b)}{(p-1)(t_f+t_b)+m(t_f+t_b)}
\;=\;
\frac{p-1}{m+p-1}
$$

结论与实践启示：

* micro-batch 越多（$m\uparrow$），空泡占比越低
  $\displaystyle \lim_{m\to\infty}\frac{p-1}{m+p-1}=0$。这是 Gpipe/1F1B 要求 $m \gg p$ 的根本原因。
实例：$p{=}4,\,m{=}8$，则 $\text{bubble ratio} = \tfrac{3}{8+3}=\tfrac{3}{11}\approx 27\%$。若将 $m$ 提升到 32：$\tfrac{3}{32+3}=\tfrac{3}{35}\approx 8.6\%$，空泡显著降低。

* 并行度越大（$p\uparrow$），若 $m$ 不同步增，则空泡变重
  盲目加深流水线会放大 warmup/cooldown 的头尾开销。

* 隐含假设
  上述推导默认各 stage 的前/反向用时近似均衡（stage balance）且采用同步调度；若 stage 时长失衡或存在通信阻塞，实际空泡会更大，需要后续的交错调度（Gpipe/1F1B）与通信-计算重叠来进一步抹平。

> 小结：Bubble Ratio 为流水线并行算法的重要评估指标。它把“结构性空转”与“有效计算”分离出来，能直接指导三个针对流水线算法的优化：加大 $m$、控制/平衡 $p$、通过调度与 overlap 把不可避免的头尾空转“隐藏”到计算之中。

## Gpipe 原理解析

### Google Gpipeline 核心思想：前向反向交错调度

基于上述定义：
$$
t_{\text{bubble}}=(p-1)(t_f+t_b),\quad
t_{\text{ideal}}=m(t_f+t_b),\quad
\Rightarrow\;
\mathit{bubble\ ratio}
=\frac{t_{\text{bubble}}}{t_{\text{bubble}}+t_{\text{ideal}}}
=\frac{p-1}{m+p-1}.
$$

我们可以计算一下朴素流水并行的空泡率，即当 **$m=1$**（不切分 micro-batch）时：

$$
\mathit{bubble\ ratio}=\frac{p-1}{p},\qquad
\text{利用率 }U=1-\mathit{bubble\ ratio}=\frac{1}{p}.
$$

当并行度$p=4$ 仅有25%利用率，即大部分时间都在空转。因此为为降低空泡、提升利用率，基于上节推导需要增大 $m$。Google 引入的Gpipe正是基于上述推导：将 batch 切成多个 micro-batch 并采用前向/反向交错调度，在有限显存下实现 $m\gg p$，显著降低空泡并提升吞吐，其原理如图 (a)所示：

![Gpipeline原理](./images/09pipeline01.png)

### 对比朴素 PP：吞吐提升、延迟降低

GPipe使用模型并行方案，将模型切分成一连串stage，每个stage放在独立的设备（GPU/TPU）上，实现对超大规模模型的支持，同时利用Pipeline的方案，提高了模型并行模式下的设备利用率，如图 (b)所示。

GPipe将mini-batch进一步划分成更小的micro-batch，同时利用pipipline方案，每次处理一个micro-batch的数据，得到结果后，将该micro-batch的结果发送给下游设备，同时开始处理后一个 micro-batch的数据，通过这套方案减小设备中的Bubble。

$$
t_{\text{bubble}}=(p-1)(t_f+t_b),\qquad
t_{\text{ideal}}=m(t_f+t_b).
$$

空泡率为，其中 $m>1$：
$$
\mathit{bubble\ ratio}
=\frac{t_{\text{bubble}}}{\,t_{\text{bubble}}+t_{\text{ideal}}\,}
=\frac{(p-1)(t_f+t_b)}{(p-1)(t_f+t_b)+m(t_f+t_b)}
=\frac{p-1}{m+p-1}.
$$

利用率：
$$
U \;=\; 1-\frac{t_{\text{bubble}}}{t_{\text{bubble}}+t_{\text{ideal}}}
\;=\; \frac{m}{\,m+p-1\,}.
$$


对于朴素流水，等价于 $m=1$

$$
\text{空泡率}=\frac{p-1}{p},\qquad
U(1)=\frac{1}{p}.
$$


因此在同一 $p$ 下，相对朴素流水的吞吐提升

$$
\text{提升倍数}
=\frac{\dfrac{m}{m+p-1}}{\dfrac{1}{p}}
=\frac{p\,m}{\,m+p-1\,}.
$$

> 含义：增大 $m$ 可将结构性空泡 $(p-1)$ 摊薄，空泡率从 $\frac{p-1}{p}$ 降至 $\frac{p-1}{m+p-1}$，利用率由 $U(1)=\frac{1}{p}$ 提升为 $U(m)=\frac{m}{m+p-1}$；当 $m\gg p$ 时，$U(m)\to 1$。


## 动态内存峰值分析

### 前向缓存激活值（activation）的内存压力

GPipe具有两段式调度（即先全前向、再全反向）的执行特点，从单台计算节点的视角看，由于每个节点进行了多组（ $m$ 组）micro-batch的前向计算，且在反向过程开始前不允许丢弃其结果，因此需要将前向产生的中间激活其存储在内存中，如图 (b)所示。该节点在前向阶段末尾会同时“攥着”本段为 $m$ 个 micro-batch 产生的全部可反向激活，形成一次性动态内存峰值；进入反向后，随着该节点依次完成每个 micro-batch 的反向，对应激活才逐步释放、内存才开始单调下降。

动态内存峰值的原因可以直观总结为以下四点，由Gpipe的特点决定：

* **两段式时序**：先全部前向（all-forward），再全部反向（all-backward）。
* **必须缓存**：反向需要用到对应的前向中间量；而反向要等前向全部完成才开始，故每个 stage 对所有 $m$ 个micro-batch的中间激活都要留存到反向阶段才能释放。
* **峰值出现**：前向阶段结束、反向尚未开始的瞬间——每个 stage 手里都有 $m$ 份本段的激活，这是动态内存峰值的主要来源。
* **直观结论**：在 GPipe 中，激活显存 $\propto m$（随 micro-batch 数线性增长）；提升吞吐用 $m \uparrow$ 摊薄空泡，会很快触到显存上限。

### Gpipe 中激活值生命周期管理

此处深度分析一下Gpipe中的激活值生命周期，以便于对峰值内存理解：

在 GPipe 的“两段式”执行中，单个 stage 上某一条 micro-batch 的激活从**生成—驻留—释放**经历一段确定的时间线。对第 $i$ 个 stage 的第 $j$ 个 micro-batch 而言，前向结束时本段会产生一组“可反向的边界激活”（例如 Transformer 中的隐状态、残差分支输入、LayerNorm 输入，视实现还可能包含 Q/K/V 或其替代物）。这组边界激活一方面被异步发送到下游 stage 作为其输入，另一方面必须本地长期保留，直到本段对该 micro-batch 的反向真正开始时才能被读取并释放，图 (b)清晰的展现了这一过程。由于 GPipe 采用“先全前向，再全反向”，反向的触发顺序是从末段、从最后一个 micro-batch 往回“倒序”推进：stage $p{-}1$ 先处理 $j=m{-}1$，再处理 $m{-}2,\dots$；前面的 stage $i$ 只有在梯度从 $i{+}1$ 回传到来时才会对 $j=m{-}1,m{-}2,\dots$ 依次启动本段反向。于是，对 $(i,j)$ 这份激活而言，其驻留跨度包含两部分：一是等待“全前向结束并轮到本段可反向”的结构性等待，量级约为 $(p-1-i)$ 个时隙；二是倒序消化到编号 $j$ 之前的序号等待，量级约为 $(m-1-j)$ 个时隙。用直观的“时隙”近似表示，其驻留时间可写成

$$
\Delta t_{i,j}\;\approx\;\bigl(p-1-i\bigr)\cdot t_f\;+\;\bigl(m-1-j\bigr)\cdot t_b,
$$

其中 $t_f,t_b$ 分别表示单 micro-batch 在单个 stage 的前/反向时长（均衡假设）。这意味着编号越靠前的 micro-batch（$j$ 小），在前段 stage（$i$ 小）上需要等待的时间越长；反之，在末段或靠后的 micro-batch，其驻留时间更短。

简单来说， Gpipe 中的激活值生命周期可以概括为：
* **生成（前向）**：stage $i$ 处理 micro-batch $j$ 的前向，产生活动张量 $F_{i,j}^{(F)}$；必须缓存到对应的反向 $B_{i,j}$ 在 stage $i$ 执行时。
* **驻留（前向→反向间）**：由于 GPipe 不交错 F/B，所有 $F_{i,0\ldots m-1}^{(F)}$ 将一直驻留到全局前向结束。
* **释放（反向）**：当 stage $i$ 对 micro-batch $j$ 的反向 $B_{i,j}$ 完毕，方可释放 $F_{i,j}^{(F)}$。

### 内存峰值公式估算

Max Memory ∝ Layer Params + Activations × $m$

* $p$：流水线并行度（stage 数）
* $m$：micro-batch 数
* $L_i$：stage $i$ 含的层数
* $A_{\text{layer}}$：单层、单 micro-batch 在可反向训练中的需缓存激活体量（字节）

  * 常与 $b$（micro-batch 内样本数）、$s$（序列长度）、$h$（隐藏维度）、$\text{bytes}$（精度字节数）近似线性相关
  * 经验：$A_{\text{layer}} \approx c \cdot b \cdot s \cdot h \cdot \text{bytes}$（系数 $c$ 取决于是否保存 QKV/残差/归一化、是否使用 FlashAttention 等实现细节）
* $P_i$：stage $i$ 的参数+优化器+梯度占用（字节）；$P_{\max}=\max_i P_i$

#### 内存峰值估算

* stage 级激活峰值

  $$
  M^{(i)}_{\text{act, peak}} \;\approx\; m \cdot L_i \cdot A_{\text{layer}}
  $$
* 训练时单卡峰值

  $$
  M_{\text{peak}} \;\approx\; \underbrace{P_{\max}}_{\text{参数系}} \;+\; \underbrace{\max_i \big(m L_i A_{\text{layer}}\big)}_{\text{激活系}}
  $$

* **结论**：在 GPipe 下，激活项对 $m$ 是严格线性的；增大 $m$ 摊薄空泡的同时，激活峰值同步线性升高。

### 通过梯度检查点降低内存峰值

核心思想是：不长期保存层内中间激活（activations），只保留“分段边界”的必要张量；等到反向时，把该段前向重新计算一次取回所需中间量。这样将“长期驻留”的激活体量从“每层都存”降为“只存段边界”，梯度检查点方法也叫**重计算**。

#### 开启重计算内存估算

将 stage $i$ 的 $L_i$ 层切分为 $g_i$ 段，仅缓存分段边界激活，层内中间量反向时重算：

  $$
  M^{(i)}_{\text{act, peak}} \;\approx\; m \cdot A_{\text{boundary}} \;+\; \alpha \cdot L_i \cdot A_{\text{layer}}
  $$

峰值仍随 $m$ 线性增长，但把“长期驻留”的系数从 $L_iA_{\text{layer}}$ 降为远小的 $A_{\text{boundary}}$；$\alpha$ 为段内短期工作集的常数项（与 $m$ 无关）。

与不开启重计算的动态峰值内存相比（同一 stage），不开启情况下：$M^{(i)}_{\text{act, peak}} \approx m\, L_i\, A_{\text{layer}}$，开启后：$m\,A_{\text{boundary}} + \alpha L_i A_{\text{layer}}$，可以看到在相同 $m$ 下重计算显著降低了内存峰值，从而可以把 $m$ 调得更大以降低空泡、提升吞吐。

* **粒度选择**：
  * 粗粒度（按 Transformer Block/子层）：易用，收益中等；
  * 细粒度（在 Attention/MLP 内再切）：峰值更低，但重算路径更长、对内核/融合要求更高。
* **段边界最小化**：尽量将残差输入、LN 输入、少量投影输出作为边界；边界越小，$A_{\text{boundary}}$ 越小。
* **与高效内核配合**：使用 FlashAttention 等“训练时不保留 $O(s^2)$ 张量”的实现，可显著降低 $A_{\text{layer}}$ 的常数项，叠加 Checkpoint 收益更佳。

* **收益**：在相同 $m$ 下直接降低内存峰值，缓解内存压力，从而可以在相同显存下扩大 $m$（空泡 $\frac{p-1}{m+p-1}$ 更小）提高吞吐。
* **代价**：5–15% 额外 FLOPs（视粒度、kernel 优化而定）；可与通信/IO 重叠部分隐藏。

> 结论：梯度检查点或重计算的本质是**用少量重算换长期驻留显存**。在 GPipe 场景下，它把与 $m$ 同步放大的“长期激活体量”压缩到仅段边界，从而在不牺牲稳定性的前提下，释放可观的显存空间或换取更大的 $m$。

## 总结与思考

!!!!!!!!!!!!!!补充

## 参考与引用

!!!!!!!!!!!!!!补充
加上参考的论文
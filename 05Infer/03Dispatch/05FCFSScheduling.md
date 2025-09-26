<!--Copyright © ZOMI 适用于 [License](https://github.com/Infrasys-AI/AIInfra) 版权许可 -->

# 07. LLM 推理请求调度：FCFS 调度

> 作者：Yuhao Chen

!!!!!!!全文基本上都是大模型生成的，文章没有灵魂，希望是你自己阅读大量的文章和内容后，然后再进行总结，总结完再用大模型润色下文字。

!!!!!!!
为什么大模型推理需要专门的请求调度？队列调度的核心作用是什么？

**FCFS（First-Come-First-Served，先来先服务）** 是在线推理中最常见、最容易落地的排队策略。在 **连续批处理（Continuous Batching）** 成为主流之后，FCFS 更像一台稳定的“节拍器”，帮助我们将 **稳定的解码产出（Decode）** 与 **低首 token 延迟（TTFT）的预填充（Prefill）** 协同起来。本文结合 vLLM 的工程实现与社区讨论，系统阐述 FCFS 的原理、优缺点、工程增强手段。

!!!!!!!
1和 2 进行合并，深入讲解 FCFS 的原理和定义，并加上自己画的流程图，特别是在大模型推理全流程的那一个位置？

## 1. FCFS 的定义

### 1.1 定义与直观理解

- **定义**：按请求的**到达时间**排序，先到的请求优先被调度。
- **直观理解**：将 API 服务视为超市收银台——谁先排队，谁先结账。其特点是**公平、可解释、易于观测**。

!!!!!!!!!图片打开不
![FCFS直觉图](./images/fcfs_intro.png)

### 1.2 与连续批处理的契合

现代推理服务采用 **迭代级调度（iteration-level scheduling）**：每个调度周期，调度器会组装一个**混合批次**（包含正在解码的序列 + 新到达或分块的预填充请求），并提交执行。在此机制下，FCFS 扮演的是**补位规则**：

1. **优先续跑解码任务**（保障稳定的产出速率 / TPOT）；
2. **用剩余资源按队列顺序注入预填充任务**（降低 TTFT）；
3. **下一轮重新组合批次**——这正是 Continuous Batching 的核心思想。

## 2. 两阶段负载与 FCFS 的角色

### 2.1 推理的两个阶段简述

- **预填充（Prefill）**：一次性处理整个输入上下文，计算量大，属于**计算密集型**（compute-bound）。
- **解码（Decode）**：每步仅生成一个 token，计算量小，但频繁访问内存，属于**访存密集型**（memory-bound），高度依赖 KV 缓存读取。

> 结论：**Prefill 与 Decode 的最优批大小不同**。若使用静态批处理将二者强行绑定，GPU 利用率会出现“断崖式气泡”。而 FCFS 在连续批处理中提供了一个**稳定且低认知负担**的“谁先补位”决策逻辑。

### 2.2 吞吐 vs 有效吞吐

- **吞吐量（TPS）** 很重要，但 **SLO 延迟指标**（如 TTFT、TPOT、P95/P99）同样关键。
- **端到端延迟 = 排队等待时间 + 预填充时间 + 多轮解码时间**。FCFS 直接决定了“排队等待”的分布，从而显著影响用户体验。

## 3. vLLM 中的 FCFS

### 3.1 关键配置（`SchedulerConfig` 摘要）

- `policy: "fcfs"`：**调度策略**，默认为 FCFS；也支持 `"priority"`。
- `max_num_batched_tokens`：**每轮调度的 token 总预算**（Prefill + Decode 共享）。
- `max_num_seqs`：**每轮最大并发序列数**（批次“座位数”）。
- `enable_chunked_prefill` / `max_num_partial_prefills`：**分块预填充**，将长 prompt 切分为小块分多轮注入，缓解**队头阻塞（Head-of-Line Blocking, HoL）**。
- `max_long_partial_prefills` / `long_prefill_token_threshold`：对“长/短 prompt”进行分级，在分块窗口内**优先填充短块**，有效降低尾部延迟。
- `num_lookahead_slots`：为推测解码预留的槽位（会占用 token 预算）。
- `disable_hybrid_kv_cache_manager`：关闭混合 KV 缓存管理（默认不建议开启）。
- `async_scheduling`：实验性**异步调度**，用于降低 CPU 调度开销。

> **重要校验**：若**未启用分块预填充**，且 `max_num_batched_tokens < max_model_len`，vLLM 会拒绝长序列请求——这是生产环境中高频事故点，上线前务必确认配置。

### 3.2 一次调度迭代如何“填批”

1. **续跑解码任务**：优先将上一轮正在执行的序列加入本轮（保障产出稳定性）。
2. **注入分块预填充**：按配额将队列头部的长/短 prompt **分块**填入（降低 TTFT，缓解队头阻塞）。
3. **补充完整预填充**：若预算和座位仍有余量，再按 FCFS 顺序填充完整 prompt。
4. **提交执行**：形成包含 **Decode + Prefill** 的混合批次。

> 实战基线：**解码优先 + 适度分块预填充配额 + KV 缓存水位控制**。

!!!!!!!!
优缺点放在 1 和 2 的合并里面

## 4. FCFS 的优缺点

### 4.1 优点

- **极简上手**：零认知负担，观测与调参成本低，发生问题后易于追溯。
- **到达公平性**：无请求饥饿，多租户场景下提供“保底”体验。
- **与连续批处理天然契合**：自然维持稳定产出的节奏，避免频繁切换带来的性能波动。

### 4.2 缺点

- **队头阻塞（HoL）**：若队首存在**超长 prompt 或长生成任务**，后续短请求会被整体拖慢，导致 **TTFT 和尾部延迟显著升高**。
- **无法感知请求长度或业务价值**：在混合负载（交互式 + 批量生成）场景下，系统整体效率（吞吐 × 用户满意度）下降。
- **难以支持细粒度 SLO**：面对“金牌/银牌/青铜”租户，纯 FCFS 无法实现差异化服务保障。

!!!!!!!!!
VLLM 的工程实现和方案统一一个内容，不要 3,5,6,7 各零散的内容，大模型写出来都是重点，用自己语言和理解总结。

## 5. FCFS 策略的工程化增强

1. **分块预填充（Chunked Prefill）**

   - 将长 prompt 切分为 256/512 token 的小块，分多轮注入，**显著缓解队头阻塞**。
   - 通过 `max_num_partial_prefills` 控制并发分块数，避免过度打断解码流程。

2. **长短请求分级 + “短块优先”窗口**

   - 使用 `long_prefill_token_threshold` 划分长短请求；
   - 设置 `max_long_partial_prefills < max_num_partial_prefills`，使短块在分块窗口内**优先调度**，有效降低 P95/P99 延迟。

3. **预算与座位的“硬性护栏”**

   - 合理设置 `max_num_batched_tokens` 和 `max_num_seqs`，防止单次预填充耗尽全部资源；
   - 通过 KV 缓存高/低水位控制，**优先保障解码任务的稳定执行**。

4. **轻量级优先级（温和改造）**

   - 在 FCFS 基础上引入少量 `priority` 字段（如“交互流量 > 批量任务”），但**限制抢占频率**，避免产出抖动。

5. **KV 缓存管理与容量优化**

   - 采用混合/分页/压缩 KV 缓存技术，扩大并发容量，减少因缓存不足导致的请求排队膨胀。

## 6. 何时切换策略：从 FCFS 到工作负载感知调度

若满足以下任一条件，可有考虑升级调度策略：

- **混合负载显著**：交互式请求与长文本生成/离线批处理共存，**尾部延迟上升、用户抱怨增多**。
- **多租户需分级 SLO**：存在明确的金银铜用户或业务优先级，需要**差异化服务质量保障**。
- **追求极致成本/吞吐**：需在高并发下进一步压缩资源空泡，提升 TPS。

**可选演进方向（循序渐进）**：

- **带权 FCFS / 优先级调度**：在保持到达公平的基础上引入**有限分层**。
- **近似最短作业优先（SJF）或学习排序（LTR）**：基于长度或剩余步数预测，实现“短任务优先”。
- **基于预估执行时间的可抢占调度**：在安全同步点进行**低成本抢占**，避免频繁上下文切换。

## 7. 可复用的 vLLM 配置模板

### 7.1 低 TTFT（交互优先）

```yaml
policy: fcfs
enable_chunked_prefill: true
max_num_partial_prefills: 2
max_long_partial_prefills: 1
long_prefill_token_threshold: 0.04 * max_model_len
max_num_batched_tokens: max(DEFAULT_MAX_NUM_BATCHED_TOKENS, 2 * max_model_len)
max_num_seqs: 128
```

### 7.2 高吞吐（离线/批量任务）

```yaml
policy: fcfs
enable_chunked_prefill: true
max_num_partial_prefills: 1
max_long_partial_prefills: 1
max_num_batched_tokens: 4 * max_model_len
max_num_seqs: 128
cuda_graph_sizes: [1,2,4,8,16,32,64,128]   # 可选
```

### 7.3 通用抗阻塞（长上下文混合场景）

```yaml
policy: fcfs
enable_chunked_prefill: true
max_num_partial_prefills: 2
max_long_partial_prefills: 1
long_prefill_token_threshold: 0.06 * max_model_len
max_num_batched_tokens: max(2 * max_model_len, DEFAULT_MAX_NUM_BATCHED_TOKENS)
max_num_seqs: 128
disable_hybrid_kv_cache_manager: false
```

---
!!!!!!
FCFS有局限性，所以可以引出 HoL blocking调度。
调度策略的整体分类和演进，简单介绍其他调度策略。

## 8. 与其他调度策略的对比

| 维度             | FCFS                     | 带权/优先级         | 近似 SJF / LTR       | 可抢占（估时/细粒度）     |
|------------------|--------------------------|--------------------|----------------------|--------------------------|
| 到达公平性       | ★★★★★                    | ★★★★☆              | ★★☆☆☆                | 取决于具体实现           |
| TTFT（交互体验） | ★★★★☆（配合分块）        | ★★★★☆              | ★★★★★                | ★★★★☆                    |
| 稳定产出（TPOT） | ★★★★☆（解码优先）        | ★★★☆☆              | ★★★☆☆                | ★★★☆☆                    |
| 实现复杂度       | ★★☆☆☆                    | ★★★☆☆              | ★★★★☆                | ★★★★★                    |
| 适用场景         | 单业务/中等并发           | 多租户分层          | 混合负载显著          | 极致 SLO / 成本优化      |

## 9. 总结与思考

> **FCFS 是默认策略，不代表着保守和低性能**。在推理框架运用 **连续批处理** 时代，通过 **解码优先 + 适度分块预填充 + 预算与缓存水位控制** 的工程基线，FCFS 能同时兼顾**公平性、稳定性与效率**。但是当当混合负载和分级 SLO 的需求日益突出时，也可以考虑将FCFS**渐进式升级**到工作负载感知的调度策略以达到性能的最大优化。

## 参考与引用


!!!!!!!哪些地方引用了这些文章了？

- [1] vLLM Community, “Is FCFS scheduling holding back vLLM’s performance in production?”, vLLM Forum, 2024. [Online]. Available: [https://discuss.vllm.ai/t/is-fcfs-scheduling-holding-back-vllms-performance-in-production/1584](https://discuss.vllm.ai/t/is-fcfs-scheduling-holding-back-vllms-performance-in-production/1584)
- [2] Y. Yu et al., “Orca: A Distributed Serving System for Transformer-Based Generative Models,” OSDI, 2022. [Online]. Available: [https://www.usenix.org/conference/osdi22/presentation/yu](https://www.usenix.org/conference/osdi22/presentation/yu)
- [3] W. Kwon et al., “Efficient Memory Management for Large Language Model Serving with PagedAttention,” 2024. [Online]. Available: [https://arxiv.org/abs/2403.02310](https://arxiv.org/abs/2403.02310)
- [4] Hugging Face, “LLM performance: prefill vs decode under concurrent requests,” Blog, 2024. [Online]. Available: [https://huggingface.co/blog/tngtech/llm-performance-prefill-decode-concurrent-requests](https://huggingface.co/blog/tngtech/llm-performance-prefill-decode-concurrent-requests)
- [5] Hugging Face, “LLM performance: request queueing strategies,” Blog, 2024. [Online]. Available: [https://huggingface.co/blog/tngtech/llm-performance-request-queueing](https://huggingface.co/blog/tngtech/llm-performance-request-queueing)
- [6] Databricks, “LLM Inference Performance Engineering Best Practices,” Blog, 2024. [Online]. Available: [https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices)
- [7] vLLM, “SchedulerConfig & Continuous Batching (source code, version-dependent),” 2024. [Online]. Available: [https://github.com/vllm-project/vllm](https://github.com/vllm-project/vllm) (see `vllm/core/scheduler`)
```
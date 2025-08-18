<!--Copyright © ZOMI 适用于[License](https://github.com/Infrasys-AI/AIInfra)版权许可-->

# 通信域与 PyTorch 实现

Author by: SingularityKChen

本章内容聚焦通信域与 PyTorch 实现，涵盖通信域、进程、进程组与Rank 间关系、PyTorch 使用分布式功能调用集合通信 和 PyTorch 执行时计算与通信并行底层原理等内容。

## 课程位置

本章节是集合同学系列课程的第五章。如下图所示，属于集合通信概览中最后一部分内容。

## 通信域与 Rank

### 通信域（Communicator）

通信域（Communicator）是 MPI 与深度学习分布式系统的核心抽象：包含 **上下文（context）**、**进程组（group）** 与 **虚拟拓扑（topology）**。一个通信域对应一个进程组；同一进程可同时加入多个通信域，互不干扰。

### 进程、进程组与 Rank

- **进程 process**：由 OS 管理，PID 唯一；同一进程可属于多个进程组。
- **进程组 group / process group**：一组要相互通信的进程集合；每个进程在组内有 **rank**（0…group_size-1）。
- **rank**：默认全局进程组（`WORLD`）的规模与序号；**local_rank** 是节点内 GPU/NPU 序号。

### PTD 并行在集群里面与模型的关系

- **TP/PP 通信域**：面向模型内部的张量/流水切分；各小组在**模型内**完成横向/纵向并行通信。
- **DP/MP 通信域**：面向跨节点的数据/混合并行；完成全局或分片梯度同步。

## PyTorch 通信调用

### PyTorch 分布式训练依赖与结构

- PyTorch 的分布式能力位于 `torch.distributed`，对上层提供 **P2P** 和 **Collective** 两类 API，对下通过 **ProcessGroup** 适配多种通信库（如 NCCL/HCCL/Gloo/MPI）。
  - Point-2-Point Communication：提供 send 和recv 语义，用于任务间通信；
  - Collective Communication：提供 scatter/broadcast/gather/reduce/all reduce/all gather 通信操作；
- PyTorch 用户直接感知的是 `distributed.py` 中定义的数据类型 `DDP:class:torch.nn.parallel.DistributedDataParallel`

### 后端通信库支持度

- **Gloo（CPU）**：覆盖基础集合通信与 P2P，用于通用 CPU 环境。
- **MPI（CPU/GPU）**：语义较全。
- **NCCL（GPU）/HCCL（NPU）**：面向深度学习高带宽低延迟互联，**重点支持 AllReduce / AllGather / ReduceScatter / AllToAll / Broadcast / Barrier** 等训练常用原语。

### P2P Communication 操作

1. 初始化：
  - PyTorch 分布式通信操作（原语）使用前，需要对分布式模块进行初始化
  - PyTorch 分布式模块通过 `torch.distributed.init_process_group` 来完成
2. 通信逻辑：
  - 通过 `rank_id` 来区分当前应该执行哪一个 rank 业务逻辑；
  - PyTorch 通过 `torch.distributed.send()` 来实现 tensor 发送，其中 send 是同步函数 isend 是异步函数；
  - PyTorch 中通过 `torch.distributed.recv()` 来实现 tensor 接收，其中 recv 是同步函数 irecv 是异步函数；
3. 任务启动：
  - 使用 `torch.multiprocessing` 来启动多进程，其是对 python 库中 multiprocessing 封装；
  - `multiprocessing.set_start_method` 用于创建 child process 方式，可选值为 fork、spawn 和 forkserver。
  - 使用 spawn，child process 仅会继承 parent process 的必要 resource，file descriptor 和 handle 均不会继承。

## PyTorch 计算与通信并行

### Stream & Event 基本概念

- **Stream**：设备上的异步命令队列；PyTorch **内存池与 Stream 绑定**。
  - 使用 CANN/CUDA 的程序一般需要处理海量数据，内存带宽经常会成为主要瓶颈。
  - 通过 Stream CANN/CUDA 的程序可以有效地将内存读取和数值运算并行，提升数据吞吐量。
  - CANN/CUDA Stream 是指一堆异步 Kernel 操作，其按照host 代码调用顺序执行在 device 上。
- **Event**：CUDA 编程中记录/等待的轻量级同步原语；用于时序与测时。
  - 允许CUDA Stream 中创建标记点，记录特定操作Start and End time，从而测量这些 Kernel 执行时间;
  - Event 可以被视为一种轻量级同步机制，不影响 GPU 执行速度，其本身不执行计算只是用来记录时间。

### Stream & Event 通信计算并行

PyTorch 通信与计算并行，主要通过 Stream（并行能力）与 Event（时序控制）这两个提供的底层能力来实现。
- 分为 host 下发与 device 执行，二者间异步：
  - 第一个 query 结果 not ready，因为 OP1 还未执行完毕，event 在等待
  - OP2 下发 Stream2，由于 OP2 前下发了 wait stream2
  - 那么wait Stream2 后下发任务都必须要等 event 完成后才执行
  - `synchronize()` 是同步接口，所以host 阻塞直到event 完成后才返回
  - 之后再调用query，返回的则是ready
- ProcessGroupXCCL 中集合通信接口会调用 `ProcessGroupXCCL::collective` 接口，该接口接收一个函数 FN:XCCL 集合通信接口调用函数。FN 下发到集合通信流（xcclStreams）上集合通信操作。
- 设 Inputs/outputs 都是默认计算流上 Tensor，那么：OP1 输出作为 HCCL1 输入 Tensor，此时该 Tensor 内存归属与计算流上内存池，PyTorch 下发是异步，会出现 OP1 写内存，而 HCCL1 在读内存。
- 需要进行集合通信的 Tensor 其内存，是在 Stream 上内存池上管理（PyTorch 内存池与Stream 绑定）。
- `collective()` 通过 `syncStream()` 函数解决异步问题：计算流下发event，在通信流下发一个 notify wait 等待 OP1 完成。Tensor 先写后读，消除并发读写问题。
- `work.wait()` 则做好时序控制，`wait()` 函数调用到 `ProcessGroupXCCL::WorkXCCL::synchronizeStreams()`，通过 block 使得 currentStream 转成计算流而非通信流。
- 那么record 是在哪里被调用的？`collective()` 中通过 xcclEndEvents_ 在xcclStream 上调用record，用于 `work.wait()` 函数执行block。

## 小结与思考

## 本节视频

<html>
<iframe src="https://player.bilibili.com/player.html?aid=1155715743&bvid=BV1VZ421g7jY&cid=1582802300&page=1&as_wide=1&high_quality=1&danmaku=0&autoplay=0" width="100%" height="500" scrolling="no" border="0" frameborder="no" framespacing="0" allowfullscreen="true"></iframe>
</html>
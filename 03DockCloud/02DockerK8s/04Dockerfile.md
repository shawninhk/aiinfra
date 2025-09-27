<!--Copyright © ZOMI 适用于[License](https://github.com/Infrasys-AI/AIInfra)版权许可-->

# Docker 隔离技术深度解析：从 Dockerfile 到运行态容器的全过程

Author by: 张柯帆

在云原生世界里，`Dockerfile` 远不止是一个简单的构建脚本；它是容器世界的“创世蓝图”，定义了一个独立、可移植、被隔离环境。然而，这份蓝图本身并不具备任何魔力。它的魔力，源于 Docker Engine 在背后对 Linux 内核三大核心技术——**Namespaces**、**Cgroups** 和 **UFS**——出神入化的编排与应用。

本文将以一个典型的 `Dockerfile` 为线索，深入剖析这三大技术在镜像构建 (`docker build`) 和容器运行 (`docker run`) 两个核心阶段，是如何协同工作，并具体产生了怎样的影响。

## 一份经典的 Dockerfile

让我们从一份构建简单 Python 应用的 `Dockerfile` 开始：

```dockerfile
# Phase 1: Base Image
FROM ubuntu:22.04

# Phase 2: Set up Environment & Application Code
WORKDIR /app
COPY . .

# Phase 3: Install Dependencies
RUN apt-get update && \
    apt-get install -y python3 python3-dev python3-pip && \
    pip3 install -r requirements.txt

# Phase 4: Define Runtime
CMD ["python3", "main.py"]
```

## `docker build` - UFS 保证一致性

当我们执行 `docker build -t my-python-app .` 时，Docker Engine 开始解读这份“蓝图”，**Union File System**（我们以其现代实现 **Overlay2** 为例）开始工作。`build` 的过程，本质上就是利用 UFS 的分层能力，将 `Dockerfile` 的每一条指令固化为一个个不可变的镜像层（Image Layer）。

1.  **`FROM ubuntu:22.04`**: Docker 会检查本地是否存在 `ubuntu:22.04` 镜像。如果存在，它会直接将该镜像的所有只读层作为本次构建的基础。在 Overlay2 里，这相当于确定了未来容器文件系统的 `lowerdir` 堆栈的底层部分。**这实现了极致的复用。** 上百个依赖于 Ubuntu 的应用，在磁盘上共享同一份基础系统文件，极大地节省了存储空间。

2.  **`WORKDIR /app`**, **`COPY . .`**, **`RUN ...`**: 这里的每一条指令，只要它会修改文件系统（创建目录、复制文件、安装软件包），Docker 就会执行以下动作：
    *   启动一个临时的中间容器。
    *   在该容器中执行指令。
    *   将该指令所产生的文件系统变更（增、删、改）记录下来，形成一个**新的、独立的只读层**。
    *   这个新层会“堆叠”在前一个层之上，成为新的 `lowerdir` 的一部分。

    **具体到 Overlay2 的实现**：每一个 `RUN` 命令执行后产生的文件变更，都会被保存在 `/var/lib/docker/overlay2/` 下的一个新的目录中。例如，`apt-get install` 安装的二进制文件和库文件，会出现在一个新的层目录里。如果一个指令删除了下层的一个文件，那么新层里会包含一个 whiteout 标记（一个设备号为 0/0 的字符设备文件），用以在最终的联合视图中“遮蔽”该文件。

    **其影响是：构建缓存与分发效率。** Docker 的构建缓存机制正是基于这些分层。如果你再次构建，只要 `Dockerfile` 的某一行以及它之前的内容没有改变，Docker 就会直接复用对应的缓存层，从而实现秒级构建。同时，当你 `docker push` 或 `docker pull` 这个镜像时，远端的 Registry 也只会传输你本地不存在的层，大大提升了镜像分发的效率。

至此，`docker build` 完成。我们得到的 `my-python-app` 镜像，并不是一个巨大的单体文件，而是一个指向一系列只读层的元数据列表。这为我们下一幕的轻量级实例化做好了完美的铺垫。

## `docker run` - 隔离

当我们执行 `docker run -d --name my-app -m 512m --cpus=".5" my-python-app` 时，Docker 就会开始应用、编排 Namespaces、Cgroups、UFS 等技术。

### 步骤 1：UFS 创建可写层

*   **动作**：Docker Storage Driver (Overlay2) 以 `my-python-app` 镜像的所有只读层作为 `lowerdir`，然后在其之上创建一个**新的、空的、可写的目录**作为 `upperdir`。同时，它会创建一个 `merged` 目录，将 `upperdir` 联合挂载在 `lowerdir` 之上，形成一个统一的视图。
*   **影响**：这个 `merged` 目录将成为容器的根文件系统 (`/`)。容器内所有的写操作，例如应用 `main.py` 写入日志文件，都会通过 **写时复制 (Copy-on-Write)** 机制，发生在 `upperdir` 中，而底层的镜像层永远保持只读和不变。这保证了镜像的纯洁性，并使得启动成百上千个基于同一镜像的容器成为可能，因为它们的只读部分是完全共享的，每个容器只额外消耗一个极薄的可写层所需的空间。**容器的销毁也变得极其廉价，只需删除这个 `upperdir` 即可。**

#### 步骤 2：Namespaces 构建隔离视界

在准备好文件系统的同时，Docker 开始为即将诞生的容器进程创建独立的命名空间。这是通过调用 Linux 的 `clone()` 或 `unshare()` 系统调用，并传入一系列 `CLONE_NEW*` 标志来实现的。

*   **MNT Namespace (`CLONE_NEWNS`)**: 这是第一个被创建的 Namespace。Docker 将上一步准备好的 `merged` 目录挂载为这个 Namespace 内部的根 (`/`)。
    *   **影响**: 容器内的进程从此拥有了与宿主机完全隔离的文件系统视图。它无法看到宿主机的文件，也无法看到其他容器的文件，它眼中的世界，就是由它的镜像和可写层构成的。

*   **UTS Namespace (`CLONE_NEWUTS`)**: 容器被赋予自己的主机名和域名。
    *   **影响**: 容器在网络上拥有独立的身份标识，实现了主机名隔离。

*   **IPC Namespace (`CLONE_NEWIPC`)**: 容器拥有独立的进程间通信资源（信号量、消息队列等）。
    *   **影响**: 容器内的进程无法与宿主机或其他容器的进程通过常规的 IPC 方式通信，防止了信息泄露和干扰。

*   **PID Namespace (`CLONE_NEWPID`)**: 这是实现进程隔离的核心。
    *   **影响**: 在这个 Namespace 内创建的第一个进程（即 `CMD` 指令定义的 `python3 main.py`），其 PID 为 1。它成为了这个 Namespace 内所有其他进程的“祖先”。容器内的进程无法看到或操作宿主机上的任何进程，甚至无法感知它们的存在。这对于安全性至关重要。

*   **NET Namespace (`CLONE_NEWNET`)**: 容器获得了全新的、独立的网络栈。
    *   **影响**: 容器拥有自己的网络设备（如 `veth` 对）、IP 地址、路由表和端口空间。`docker run` 时的 `-p 8080:80` 参数，其本质就是在宿主机的 NET Namespace 和容器的 NET Namespace 之间建立了一个端口映射规则。这使得容器的网络行为被严格限制和管理。

#### 步骤 3：Cgroups 精确限制资源

在进程的隔离视界被完美构建之后，Docker 还需要使用 **Cgroups** 确保这个命名空间不会过度消耗宿主机的资源。

*   **动作**：
    1.  Docker 会在 `/sys/fs/cgroup/` 下的各个子系统（如 `memory`, `cpu`）中，为这个新容器创建一个唯一的控制组目录（例如 `/sys/fs/cgroup/memory/docker/<container_id>/`）。
    2.  然后，Docker 会解析 `docker run` 命令中的资源限制参数，并将它们转换成对 Cgroups 文件系统的写操作：
        *   `--m 512m` 会被转换为向 `memory.limit_in_bytes` 文件写入 `536870912`。
        *   `--cpus=".5"` 会被转换为向 `cpu.cfs_period_us` 写入 `100000` 并向 `cpu.cfs_quota_us` 写入 `50000`。
    3.  最后，也是最关键的一步，Docker 将容器主进程的 PID 写入这些控制组目录下的 `tasks` 文件中。

*   **影响**: 从这一刻起，Linux 内核调度器会严格执行这些限制。容器内的所有进程（包括其后续创建的子进程，因为它们会自动继承父进程的 Cgroup）的内存使用量总和不能超过 512MB，否则内核的 OOM Killer 会介入。它们在每 100ms 的周期内，总共能获得的 CPU 时间不会超过 50ms。**Cgroups 提供了资源隔离的“硬边界”，确保了容器之间、以及容器与宿主机之间的公平与稳定，是多租户环境和高密度部署的基石。**

### 结论：一场精妙的内核级协同

回顾全程，我们可以看到：

*   **`Dockerfile`** 是静态的定义，是隔离的蓝图。
*   **Union File System** 在构建时将蓝图固化为高效、可复用的分层镜像；在运行时提供轻量的可写环境。它解决了 **“环境是什么”** 的问题。
*   **Namespaces** 在运行时为进程构建了一个全方位的隔离视图，让它误以为自己独占系统。它解决了 **“进程在哪里”** 的问题。
*   **Cgroups** 在运行时为这个隔离环境设定了不可逾越的资源边界。它解决了 **“能用多少”** 的问题。

Docker 的伟大，不在于发明了这些技术，而在于它将这些原本孤立、复杂的 Linux 内核特性，通过一个简单、一致的接口（`Dockerfile`, `docker` CLI）完美地封装和编排起来，最终为我们呈现出轻量、秒级、可移植的容器体验。从 `Dockerfile` 的一行文本，到运行态容器的每一个隔离细节，这趟内核之旅深刻地揭示了现代云原生技术的精髓——**对底层能力的极致抽象与协同**。

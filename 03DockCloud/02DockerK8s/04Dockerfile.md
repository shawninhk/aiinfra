<!--Copyright © ZOMI 适用于[License](https://github.com/Infrasys-AI/AIInfra)版权许可-->

# 04. Dockerfile 到容器

> Author by: 张柯帆

在云原生世界里，`Dockerfile` 远不止是一个简单的构建脚本；它是容器世界的“创世蓝图”，定义了一个独立、可移植、可隔离的运行环境。然而，这份蓝图本身并不具备任何魔力。它的魔力，源于 Docker Engine 在背后对 Linux 内核三大核心技术——**Namespaces**、**Cgroups** 和 **UFS（Union File System）**——的巧妙编排与运用。

![](./images/04Dockerfile01.png)

本文将以一份典型的 `Dockerfile` 为线索，深入剖析这三大技术在镜像构建（`docker build`）和容器运行（`docker run`）两个核心阶段，是如何协同工作，并具体产生了怎样的影响。

> 注：以下讨论基于 Linux

!!!!!!!!补充说明：
!!!!!!!!!“Dockerfile → Docker 镜像 → Docker 容器”的转化关系，帮助理解后续技术拆解的核心链路：
![](./images/04Dockerfile02.png)

## 1. 经典 Dockerfile

让我们从一份构建简单 Python 应用的 `Dockerfile` 开始，其指令逻辑对应容器环境的核心构建步骤：

```dockerfile
# Phase 1: 基础镜像（构建环境的起点）
FROM ubuntu:22.04

# Phase 2: 环境配置与代码注入
WORKDIR /app  # 设定容器内工作目录
COPY . .      # 将本地当前目录文件复制到容器工作目录

# Phase 3: 依赖安装（构建可运行的应用环境）
RUN apt-get update && \
    apt-get install -y python3 python3-dev python3-pip && \
    pip3 install -r requirements.txt

# Phase 4: 定义运行时指令（容器启动后执行的命令）
CMD ["python3", "main.py"]
```

## 2. docker build 与 UFS

当我们执行 `docker build -t my-python-app .` 时，Docker Engine 开始解读这份“蓝图”，而 **Union File System（联合文件系统）**（此处以其现代实现、当前 Docker 默认存储驱动 **Overlay2** 为例）则是构建过程的核心支撑。`docker build` 的本质，就是利用 UFS 的分层能力，将 `Dockerfile` 的每一条指令“固化”为一个个不可变的镜像层（Image Layer）。

1.  **`FROM ubuntu:22.04`**：Docker 会先检查本地是否存在 `ubuntu:22.04` 基础镜像。若存在，直接将该镜像的所有只读层作为本次构建的“底层基础”；若不存在，则自动从远端镜像仓库（如 Docker Hub）拉取。在 Overlay2 机制中，这相当于确定了未来容器文件系统 `lowerdir`（Overlay2 中的“底层目录”，存储只读层数据）堆栈的最底层部分。**这一设计实现了极致的资源复用**：上百个依赖 Ubuntu 基础镜像的应用，在宿主机磁盘上可共享同一份基础系统文件，极大节省了存储空间。

2.  **`WORKDIR /app`**、**`COPY . .`**、**`RUN ...`**：对于这类会修改文件系统（如创建目录、复制文件、安装软件）的指令，Docker 会执行一套标准化流程：
    *   启动一个临时的“中间容器”（基于上一步的镜像层创建，构建完成后自动销毁）；
    *   在该中间容器内执行当前指令；
    *   将指令产生的文件系统变更（新增、删除、修改）完整记录下来，生成一个**新的、独立的只读层**；
    *   这个新层会“堆叠”在前一个镜像层之上，成为下一条指令构建时 `lowerdir` 堆栈的一部分。

    **具体到 Overlay2 的实现细节**：每一条 `RUN` 指令执行后产生的文件变更，都会被保存在 `/var/lib/docker/overlay2/` 目录下的一个独立子目录中。例如，`apt-get install` 安装的 Python 二进制文件、依赖库，会被记录在一个新的镜像层目录内。若某条指令删除了下层镜像中的文件（如 `RUN rm /tmp/test.txt`），Overlay2 不会直接修改底层只读文件，而是在新层中创建一个 **whiteout 标记**（设备号为 0/0 的特殊字符设备文件），用于在最终的“联合视图”中“遮蔽”底层的目标文件（即让容器“看不到”被删除的文件）。

    **核心影响：构建缓存与分发效率的提升**：Docker 的“构建缓存”机制正是依托这些不可变的分层实现的。若再次执行构建，只要 `Dockerfile` 中某一行指令及其之前的所有指令未发生改变，Docker 就会直接复用本地已有的对应镜像层（即“命中缓存”），从而实现“秒级构建”。同时，当执行 `docker push`（推送镜像到仓库）或 `docker pull`（拉取镜像）时，远端仓库只会传输本地缺失的镜像层，而非整个镜像，大幅减少了网络传输量，提升了镜像分发效率。

至此，`docker build` 流程完成。我们最终得到的 `my-python-app` 镜像，并非一个单一的“大文件”，而是一个包含“镜像层列表、镜像元数据（如启动命令、环境变量）”的集合。这种分层设计，也为后续容器的“轻量级实例化”奠定了基础。

## 3. docker run 关键步骤

当我们执行 `docker run -d --name my-app -m 512m --cpus=".5" my-python-app` 时，Docker 会启动一套“组合拳”——协同 UFS、Namespaces、Cgroups 三大技术，将静态的镜像转化为动态的运行态容器。

### 3.1 UFS 建可写系统

*   **核心动作**：Docker 存储驱动（此处为 Overlay2）会以 `my-python-app` 镜像的所有只读层作为 `lowerdir`，同时在其之上创建一个**新的、空的可写目录**作为 `upperdir`（Overlay2 中的“上层目录”，存储容器运行时的写操作数据）；此外，还会创建一个 `merged` 目录（Overlay2 中的“联合目录”），将 `upperdir` 与 `lowerdir` 进行“联合挂载”，形成一个统一的文件系统视图。
*   **关键影响**：这个 `merged` 目录会被作为容器的根文件系统（`/`）。容器运行过程中的所有写操作（如应用 `main.py` 写入日志文件、创建临时文件），都会通过 **写时复制（Copy-on-Write，简称 CoW）** 机制，仅发生在 `upperdir` 中——底层的镜像只读层永远保持不变。这一设计带来两大优势：
    1.  保障了镜像的“纯洁性”，避免因容器写操作污染基础镜像；
    2.  实现了容器的“轻量级启动”：成百上千个基于同一镜像的容器，可共享所有只读层，每个容器仅需额外占用 `upperdir`（通常仅几 KB 到几十 MB）的存储空间。**而容器销毁也极为“廉价”——只需删除对应的 `upperdir` 和挂载关系即可，不会影响底层镜像。**

#### 3.2 Namespaces 隔离

在文件系统准备的同时，Docker 会通过 Linux 内核提供的 `clone()` 或 `unshare()` 系统调用（传入 `CLONE_NEW*` 系列标志），为即将启动的容器进程创建一套独立的 **Namespaces**，实现“进程看不到外部、外部看不到进程”的隔离效果。

*   **`MNT Namespace`（`CLONE_NEWNS`）**：这是第一个被创建的 Namespace。Docker 会将上一步准备好的 `merged` 目录，挂载为该 Namespace 内部的根目录（`/`）。
    *   **隔离效果**：容器内进程看到的文件系统，仅包含镜像只读层与 `upperdir` 联合后的内容，无法访问宿主机或其他容器的文件，实现了文件系统的完全隔离。

*   **`UTS Namespace`（`CLONE_NEWUTS`）**：为容器分配独立的主机名（如 `--name my-app` 设定的名称）和域名。
    *   **隔离效果**：容器在网络环境中拥有独立的“身份标识”，避免了主机名冲突，同时也让容器内应用可基于自身主机名运行（如某些服务依赖主机名配置）。

*   **`IPC Namespace`（`CLONE_NEWIPC`）**：为容器创建独立的进程间通信（IPC）资源池，包括信号量、消息队列、共享内存等。
    *   **隔离效果**：容器内进程无法与宿主机或其他容器的进程通过标准 IPC 方式通信，防止了跨容器的信息泄露与运行干扰。

*   **`PID Namespace`（`CLONE_NEWPID`）**：实现进程 ID 隔离的核心，也是容器“轻量化”的关键特性之一。
    *   **隔离效果**：在该 Namespace 内启动的第一个进程（即 `Dockerfile` 中 `CMD` 指令定义的 `python3 main.py`），其 PID 会被设为 1（相当于宿主机的 `init` 进程），成为容器内所有后续进程的“祖先”。容器内进程无法看到或操作宿主机的任何进程（即使宿主机 PID 为 1 的 `systemd` 进程），甚至无法感知宿主机的存在，极大提升了运行安全性。

*   **`NET Namespace`（`CLONE_NEWNET`）**：为容器创建一套全新的、独立的网络栈，包括虚拟网卡（如 `veth` 对的一端）、IP 地址、路由表、端口空间等。
    *   **隔离效果**：容器拥有独立的“网络环境”，其端口（如应用监听的 80 端口）与宿主机、其他容器的端口完全独立。`docker run` 中的 `-p 8080:80` 参数，本质就是在宿主机 `NET Namespace` 与容器 `NET Namespace` 之间建立端口映射规则（将宿主机 8080 端口的流量转发到容器 80 端口），实现外部对容器应用的访问。

#### 3.3 Cgroups 设资源边界

在完成进程隔离后，Docker 还需要通过 **Cgroups（Control Groups，控制组）** 技术，限制容器对宿主机资源的使用，避免单个容器“过度占用资源”影响其他容器或宿主机的稳定性。

*   **核心动作**：
    1.  Docker 会在 Linux 内核的 Cgroups 挂载点（通常为 `/sys/fs/cgroup/`）下，为新容器在各个资源子系统（如 `memory` 内存子系统、`cpu` CPU 子系统）中创建独立的控制组目录。例如，内存子系统的控制组目录可能为 `(/sys/fs/cgroup/memory/docker/<container_id>/)`；
    2.  解析 `docker run` 中的资源限制参数，并转化为对 Cgroups 文件系统的写操作：
        *   `--memory 512m`（简写 `--m 512m`）：向 `memory.limit_in_bytes` 文件写入 `536870912`（即 512MB 对应的字节数），限制容器内存使用上限；
        *   `--cpus=".5"`：向 `cpu.cfs_period_us` 文件写入 `100000`（即 100ms 的调度周期），同时向 `cpu.cfs_quota_us` 文件写入 `50000`（即每个周期内容器可使用的 CPU 时间上限为 50ms）；
    3.  将容器主进程（即 PID Namespace 中 PID 为 1 的进程）的实际 PID，写入上述控制组目录下的 `tasks` 文件——这一步是“生效关键”，表示该进程及后续衍生的子进程，均受当前控制组规则约束。

*   **关键影响**：从 `tasks` 文件写入 PID 的那一刻起，Linux 内核会严格执行资源限制：
    *   容器内所有进程的内存使用总和，一旦超过 512MB，内核的 OOM Killer（内存溢出杀手）会触发，自动终止容器内占用内存过高的进程，避免影响宿主机整体内存稳定性；
    *   在每 100ms 的 CPU 调度周期内，容器内所有进程可使用的 CPU 时间总和不超过 50ms（即占用宿主机 50% 的单个 CPU 核心资源），实现了容器间的 CPU 资源公平分配。

    **Cgroups 的核心价值**：为容器资源使用划定了“硬边界”，是云环境中“多租户部署”“高密度容器运行”的技术基石——它确保了容器之间、容器与宿主机之间的资源隔离，避免了“一个容器故障拖垮整个系统”的风险。

## 4. 技术协同结论

回顾从 `Dockerfile` 到运行态容器的全流程，我们可以清晰看到三大技术的分工与协作：

*   **`Dockerfile`**：是静态的“隔离蓝图”，定义了容器的基础环境、依赖与启动逻辑；
*   **Union File System**：构建阶段将“蓝图”固化为高效可复用的分层镜像，运行阶段提供轻量级可写环境，解决了“容器环境是什么”的核心问题；
*   **Namespaces**：运行阶段为进程构建全方位的隔离视图，使其误以为“独占整个系统”，解决了“进程在哪个隔离空间运行”的问题；
*   **Cgroups**：运行阶段为隔离环境设定不可逾越的资源边界，解决了“容器能使用多少宿主机资源”的问题。

Docker 的伟大，并非发明了 Namespaces、Cgroups 或 UFS 这些技术——这些都是 Linux 内核早已存在的能力——而是将这些孤立、复杂的底层技术，通过一套简单、一致的工具链（`Dockerfile` 语法、`docker` 命令行）封装、编排起来，最终为用户呈现出“轻量、秒级启动、可移植”的容器体验。从 `Dockerfile` 中的一行指令，到运行态容器的每一个隔离细节，这趟“内核之旅”恰恰揭示了现代云原生技术的精髓——**对底层能力的极致抽象与协同**。

## 5. 总结与思考

通过拆解 `Dockerfile` 到运行态容器的全流程，我们能跳出“容器是‘轻量级虚拟机’”的误区，更深刻地理解：容器本质是 Linux 内核三大技术（Namespaces、Cgroups、UFS）协同作用的“进程隔离环境”。Docker 工具链的价值，在于将复杂的内核能力转化为“开发者友好”的接口，让“一次构建，到处运行”的理念落地。

## 6. 参考与引用

- Docker 官方文档：[Docker Storage Drivers](https://docs.docker.com/storage/storagedriver/)
- Linux 内核手册：[Namespaces](https://man7.org/linux/man-pages/man7/namespaces.7.html)
- [Cgroups](https://man7.org/linux/man-pages/man7/cgroups.7.html)

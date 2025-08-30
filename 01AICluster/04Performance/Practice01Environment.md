<!--Copyright © ZOMI 适用于[License](https://github.com/Infrasys-AI/AIInfra)版权许可-->

# GPU与CUDA计算：从硬件基础到性能可视化

## 实验概述

本实验将带您深入了解GPU硬件架构和CUDA编程环境，通过实践操作学习如何获取GPU信息、分析性能指标并进行可视化比较。我们将使用PyCUDA库来与GPU设备交互，并用Matplotlib创建直观的性能对比图表。

## 1 环境配置与准备工作

在开始实验之前，我们需要配置合适的开发环境。这包括安装必要的软件包和验证CU环境是否可用。

> **知识背景**: CUDA是NVIDIA推出的并行计算平台和编程模型，它允许开发者利用NVIDIA GPU的强大计算能力进行通用计算，而不仅仅是图形渲染。

```python
# 环境配置检查脚本
import sys
import subprocess
import platform

def check_cuda_available():
    """检查系统是否安装了CUDA"""
    try:
        # 尝试运行nvcc命令检查CUDA版本
        result = subprocess.run(
            ["nvcc", "--version"], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True
        )
        if result.returncode == 0:
            print("CUDA已安装:")
            print(result.stdout.split('\n')[3])  # 提取版本信息行
            return True
        else:
            print("未检测到CUDA。请先安装CUDA Toolkit。")
            print("下载地址: https://developer.nvidia.com/cuda-toolkit")
            return False
    except FileNotFoundError:
        print("未检测到CUDA。请先安装CUDA Toolkit。")
        print("下载地址: https://developer.nvidia.com/cuda-toolkit")
        return False

# 执行环境检查
print("=== CUDA Python开发环境配置检查 ===")
print(f"操作系统: {platform.system()} {platform.release()}")
print(f"Python版本: {sys.version.split()[0]}")

# 检查CUDA是否可用
cuda_available = check_cuda_available()
```

执行上述代码后，如果系统已安装CUDA，将显示CUDA版本信息；否则会提示您安装CUDA Toolkit。

## 2 GPU硬件基础与架构差异

GPU与CPU在架构设计上有根本性的不同，理解这些差异对于充分利用GPU计算能力至关重要。

> **核心概念**: CPU通常具有少数几个高性能核心，擅长处理复杂的串行任务和控制逻辑；而GPU则包含数百到数千个较小的处理核心，专为并行任务设计。

```python
# 安装必要的Python包
import sys
import subprocess

def install_packages():
    """安装必要的Python包"""
    packages = [
        "pycuda",      # CUDA的Python绑定
        "numpy",       # 数值计算库
        "matplotlib",  # 数据可视化库
        "pandas"       # 数据处理库
    ]
    
    print("\n开始安装必要的Python包...")
    for package in packages:
        try:
            # 尝试导入包，检查是否已安装
            __import__(package)
            print(f"{package} 已安装")
        except ImportError:
            # 未安装则进行安装
            print(f"正在安装 {package}...")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", package]
            )
    
    print("\n所有必要的包已安装完成")

# 执行安装
install_packages()
```

安装完成后，我们可以验证PyCUDA是否能正常工作：

```python
# 验证PyCUDA环境
def verify_environment():
    """验证PyCUDA是否能正常工作"""
    try:
        import pycuda.autoinit
        import pycuda.driver as drv
        
        print("\nPyCUDA环境验证成功!")
        print(f"检测到 {drv.Device.count()} 个NVIDIA GPU设备")
        
        # 显示第一个GPU的信息
        if drv.Device.count() > 0:
            device = drv.Device(0)
            print(f"第一个GPU设备: {device.name()}")
            print(f"计算能力: {device.compute_capability()}")
            print(f"总内存: {device.total_memory() / (1024**2):.2f} MB")
        
        return True
    except Exception as e:
        print(f"\nPyCUDA环境验证失败: {str(e)}")
        print("请检查CUDA安装和PyCUDA配置")
        return False

# 执行验证
verify_environment()
```

## 3 GPU信息查询与性能分析

现在我们来创建一个完整的GPU信息查询工具，它可以获取GPU的详细参数并计算关键性能指标。

> **技术背景**: 现代AI研究高度依赖GPU计算能力，深入了解GPU性能特征对于优化深度学习模型至关重要。

```python
# GPU信息查询工具
import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
import pandas as pd

class GPUInfo:
    """GPU信息查询类，用于获取和展示NVIDIA GPU的详细信息"""
    
    def __init__(self):
        """初始化，获取系统中所有GPU设备"""
        self.num_devices = drv.Device.count()
        self.devices = [drv.Device(i) for i in range(self.num_devices)]
        self.device_info = []
        
    def get_device_details(self, device_index=0):
        """
        获取指定GPU设备的详细信息
        
        参数:
            device_index: GPU设备索引，默认为0
            
        返回:
            包含GPU详细信息的字典
        """
        if device_index < 0 or device_index >= self.num_devices:
            raise ValueError(f"设备索引无效，有效范围: 0 到 {self.num_devices-1}")
            
        device = self.devices[device_index]
        props = device.get_attributes()
        
        # 计算理论峰值性能 (FLOPS)
        compute_capability = device.compute_capability()
        sm_count = props[drv.device_attribute.MULTIPROCESSOR_COUNT]
        
        # 根据计算能力估算每个SM的核心数量
        if compute_capability[0] >= 8:  # Ampere及以上架构
            cores_per_sm = 128
        elif compute_capability[0] == 7:  # Volta架构
            cores_per_sm = 64
        elif compute_capability[0] == 6:  # Pascal架构
            cores_per_sm = 128
        elif compute_capability[0] == 5:  # Maxwell架构
            cores_per_sm = 128
        elif compute_capability[0] == 3:  # Kepler架构
            cores_per_sm = 192
        else:  # 早期架构
            cores_per_sm = 80
            
        total_cores = sm_count * cores_per_sm
        
        # 获取GPU时钟频率 (kHz) 并转换为GHz
        clock_rate = props[drv.device_attribute.CLOCK_RATE] / 1e6  # GHz
        
        # 计算理论峰值FP32性能 (TFLOPS)
        peak_fp32 = (total_cores * clock_rate * 2) / 1e3  # 转换为TFLOPS
        
        # 获取内存带宽 (GB/s)
        memory_bus_width = props[drv.device_attribute.GLOBAL_MEMORY_BUS_WIDTH]
        memory_clock_rate = props[drv.device_attribute.MEMORY_CLOCK_RATE] / 1e6  # GHz
        memory_bandwidth = (memory_bus_width * memory_clock_rate * 2) / 8  # GB/s
        
        info = {
            "设备名称": device.name(),
            "设备索引": device_index,
            "计算能力": f"{compute_capability[0]}.{compute_capability[1]}",
            "SM数量": sm_count,
            "估计核心总数": total_cores,
            "基础时钟频率 (GHz)": round(clock_rate, 2),
            "理论峰值FP32性能 (TFLOPS)": round(peak_fp32, 2),
            "总内存": f"{device.total_memory() / (1024**3):.2f} GB",
            "内存带宽 (GB/s)": round(memory_bandwidth, 2),
            "内存总线宽度 (位)": memory_bus_width,
            "支持的CUDA核心版本": props[drv.device_attribute.CUDA_CORE_VERSION],
            "最大线程块大小": props[drv.device_attribute.MAX_BLOCK_DIM_X],
            "每个SM的最大线程数": props[drv.device_attribute.MAX_THREADS_PER_MULTIPROCESSOR],
            "L2缓存大小 (MB)": props[drv.device_attribute.L2_CACHE_SIZE] / (1024**2)
        }
        
        self.device_info.append(info)
        return info
```

> **性能计算原理**: GPU的理论峰值性能计算公式为`核心数量 × 基础频率 × 2`，其中的"×2"是因为每个CUDA核心在每个时钟周期可以执行2个单精度浮点运算(FP32)。这个指标反映了GPU在理想情况下的最大计算能力。

接下来，我们添加一些方法来展示和比较GPU信息：

```python
# 继续GPUInfo类的方法
def get_all_devices_info(self):
    """获取系统中所有GPU设备的信息"""
    self.device_info = []  # 清空现有信息
    for i in range(self.num_devices):
        self.get_device_details(i)
    return self.device_info

def print_device_info(self, device_info=None):
    """打印GPU设备信息"""
    if device_info is None:
        if not self.device_info:
            self.get_all_devices_info()
        for info in self.device_info:
            self._print_single_device_info(info)
    else:
        self._print_single_device_info(device_info)

def _print_single_device_info(self, info):
    """打印单个GPU设备的信息"""
    print(f"\n=== GPU设备信息: {info['设备名称']} ===")
    print(f"设备索引: {info['设备索引']}")
    print(f"计算能力: {info['计算能力']}")
    print(f"SM数量: {info['SM数量']}")
    print(f"估计核心总数: {info['估计核心总数']}")
    print(f"基础时钟频率: {info['基础时钟频率 (GHz)']} GHz")
    print(f"理论峰值FP32性能: {info['理论峰值FP32性能 (TFLOPS)']} TFLOPS")
    print(f"总内存: {info['总内存']}")
    print(f"内存带宽: {info['内存带宽 (GB/s)']} GB/s")
    print(f"内存总线宽度: {info['内存总线宽度 (位)']} 位")
    print(f"L2缓存大小: {info['L2缓存大小 (MB)']} MB")
```

为了使比较更加直观，我们预定义一些常见GPU的性能数据：

```python
# 预定义一些常见GPU的性能数据，用于对比
common_gpus_data = [
    {
        "设备名称": "NVIDIA V100",
        "设备索引": -1,
        "计算能力": "7.0",
        "SM数量": 80,
        "估计核心总数": 5120,
        "基础时钟频率 (GHz)": 1.38,
        "理论峰值FP32性能 (TFLOPS)": 14.1,
        "总内存": "16.00 GB",
        "内存带宽 (GB/s)": 900.0,
        "内存总线宽度 (位)": 4096,
        "支持的CUDA核心版本": "7.0",
        "最大线程块大小": 1024,
        "每个SM的最大线程数": 2048,
        "L2缓存大小 (MB)": 6.0
    },
    {
        "设备名称": "NVIDIA A100",
        "设备索引": -1,
        "计算能力": "8.0",
        "SM数量": 108,
        "估计核心总数": 14016,
        "基础时钟频率 (GHz)": 1.41,
        "理论峰值FP32性能 (TFLOPS)": 31.2,
        "总内存": "40.00 GB",
        "内存带宽 (GB/s)": 1555.0,
        "内存总线宽度 (位)": 5120,
        "支持的CUDA核心版本": "8.0",
        "最大线程块大小": 1024,
        "每个SM的最大线程数": 2048,
        "L2缓存大小 (MB)": 40.0
    },
    {
        "设备名称": "NVIDIA GeForce RTX 3090",
        "设备索引": -1,
        "计算能力": "8.6",
        "SM数量": 82,
        "估计核心总数": 10496,
        "基础时钟频率 (GHz)": 1.40,
        "理论峰值FP32性能 (TFLOPS)": 29.17,
        "总内存": "24.00 GB",
        "内存带宽 (GB/s)": 936.0,
        "内存总线宽度 (位)": 384,
        "支持的CUDA核心版本": "8.6",
        "最大线程块大小": 1024,
        "每个SM的最大线程数": 2048,
        "L2缓存大小 (MB)": 6.0
    }
]
```

现在，我们可以使用这个工具来查询和比较GPU信息：

```python
# 使用GPUInfo工具
print("=== GPU信息查询工具 ===")

# 创建GPU信息查询实例
gpu_info = GPUInfo()

print(f"检测到 {gpu_info.num_devices} 个NVIDIA GPU设备")

# 获取并打印所有设备信息
gpu_info.get_all_devices_info()
gpu_info.print_device_info()

# 创建包含常见GPU的比较数据框
comparison_df = gpu_info.create_comparison_dataframe(common_gpus_data)
print("\n=== GPU性能比较表 ===")
print(comparison_df.to_string(index=False))
```

## 4 GPU性能可视化分析

数据可视化可以帮助我们更直观地理解GPU性能特征和差异。下面我们创建一个可视化工具来生成各种性能对比图表。

> **设计思路**: 良好的可视化能够揭示数据中隐藏的模式和关系，帮助研究者快速把握关键信息。

```python
# GPU性能可视化工具
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 设置中文字体支持
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

class GPUVisualizer:
    """GPU性能可视化工具，用于创建不同GPU性能指标的对比图表"""
    
    def __init__(self, comparison_df=None):
        """
        初始化可视化工具
        
        参数:
            comparison_df: 包含GPU比较数据的DataFrame，如未提供则自动生成
        """
        if comparison_df is None:
            # 如果未提供数据，自动生成包含系统GPU和常见GPU的比较数据
            gpu_info = GPUInfo()
            gpu_info.get_all_devices_info()
            self.comparison_df = gpu_info.create_comparison_dataframe(common_gpus_data)
        else:
            self.comparison_df = comparison_df
    
    def plot_fp32_performance(self, save_path=None):
        """
        绘制不同GPU的FP32性能对比图
        
        参数:
            save_path: 图表保存路径，如为None则直接显示图表
        """
        # 按性能排序
        df_sorted = self.comparison_df.sort_values(by="FP32性能 (TFLOPS)")
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(df_sorted["设备名称"], df_sorted["FP32性能 (TFLOPS)"], 
                      color='skyblue', edgecolor='black')
        
        # 在柱状图上添加数值
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=10)
        
        plt.title('不同GPU的FP32计算性能对比 (TFLOPS)', fontsize=14)
        plt.xlabel('GPU型号', fontsize=12)
        plt.ylabel('FP32性能 (TFLOPS)', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"FP32性能对比图已保存至: {save_path}")
        else:
            plt.show()
```

让我们继续添加更多的可视化方法：

```python
# 继续GPUVisualizer类的方法
def plot_memory_bandwidth(self, save_path=None):
    """
    绘制不同GPU的内存带宽对比图
    
    参数:
        save_path: 图表保存路径，如为None则直接显示图表
    """
    # 按内存带宽排序
    df_sorted = self.comparison_df.sort_values(by="内存带宽 (GB/s)")
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(df_sorted["设备名称"], df_sorted["内存带宽 (GB/s)"], 
                  color='lightgreen', edgecolor='black')
    
    # 在柱状图上添加数值
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 10,
                f'{height:.0f}', ha='center', va='bottom', fontsize=10)
    
    plt.title('不同GPU的内存带宽对比 (GB/s)', fontsize=14)
    plt.xlabel('GPU型号', fontsize=12)
    plt.ylabel('内存带宽 (GB/s)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"内存带宽对比图已保存至: {save_path}")
    else:
        plt.show()

def plot_memory_size(self, save_path=None):
    """
    绘制不同GPU的内存大小对比图
    
    参数:
        save_path: 图表保存路径，如为None则直接显示图表
    """
    # 按内存大小排序
    df_sorted = self.comparison_df.sort_values(by="内存大小 (GB)")
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(df_sorted["设备名称"], df_sorted["内存大小 (GB)"], 
                  color='salmon', edgecolor='black')
    
    # 在柱状图上添加数值
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.0f} GB', ha='center', va='bottom', fontsize=10)
    
    plt.title('不同GPU的内存大小对比', fontsize=14)
    plt.xlabel('GPU型号', fontsize=12)
    plt.ylabel('内存大小 (GB)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"内存大小对比图已保存至: {save_path}")
    else:
        plt.show()
```

为了更深入地理解GPU性能特征，我们创建一个散点图来展示计算性能与内存带宽的关系：

```python
# 继续GPUVisualizer类的方法
def plot_performance_vs_bandwidth(self, save_path=None):
    """
    绘制计算性能与内存带宽的散点图，展示两者关系
    
    参数:
        save_path: 图表保存路径，如为None则直接显示图表
    """
    plt.figure(figsize=(10, 6))
    
    # 绘制散点图
    plt.scatter(self.comparison_df["内存带宽 (GB/s)"], 
               self.comparison_df["FP32性能 (TFLOPS)"],
               s=self.comparison_df["内存大小 (GB)"] * 10,  # 用点的大小表示内存大小
               alpha=0.7, edgecolors="w", linewidth=0.5)
    
    # 添加标签
    for i, row in self.comparison_df.iterrows():
        plt.annotate(row["设备名称"], 
                    (row["内存带宽 (GB/s)"], row["FP32性能 (TFLOPS)"]),
                    fontsize=9, ha='right')
    
    plt.title('GPU计算性能与内存带宽关系', fontsize=14)
    plt.xlabel('内存带宽 (GB/s)', fontsize=12)
    plt.ylabel('FP32性能 (TFLOPS)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"性能与带宽关系图已保存至: {save_path}")
    else:
        plt.show()

def generate_all_plots(self, output_dir="."):
    """
    生成所有可用的对比图表并保存
    
    参数:
        output_dir: 图表保存目录
    """
    self.plot_fp32_performance(f"{output_dir}/fp32_performance.png")
    self.plot_memory_bandwidth(f"{output_dir}/memory_bandwidth.png")
    self.plot_memory_size(f"{output_dir}/memory_size.png")
    self.plot_performance_vs_bandwidth(f"{output_dir}/performance_vs_bandwidth.png")
    print("所有图表已生成并保存")
```

现在，让我们使用这个可视化工具来创建图表：

```python
# 使用GPUVisualizer工具
print("=== GPU性能可视化工具 ===")

# 创建可视化实例
visualizer = GPUVisualizer()

# 生成并显示所有图表
visualizer.plot_fp32_performance()
visualizer.plot_memory_bandwidth()
visualizer.plot_memory_size()
visualizer.plot_performance_vs_bandwidth()

# 或者一次性保存所有图表
# visualizer.generate_all_plots("./gpu_plots")
```

## 5 实验分析与总结

通过本实验，我们深入了解了GPU硬件架构和性能特征，并创建了实用的工具来查询和可视化GPU信息。

> **深度思考**: 现代AI研究正越来越多地依赖高性能计算资源，理解GPU性能特征对于优化深度学习工作流程至关重要。通过工具化的方式管理GPU资源，可以大大提高研究效率。

### 关键知识点总结

1.  **GPU架构特点**: GPU采用大规模并行架构，拥有数千个计算核心，适合处理高度并行的计算任务。

2.  **性能指标**:
    *   **理论峰值性能**: 反映GPU在理想条件下的最大计算能力
    *   **内存带宽**: 决定数据在GPU内存和计算核心之间的传输速度
    *   **显存容量**: 影响GPU能处理的数据集大小和模型复杂度

3.  **工具开发价值**: 自动化GPU信息查询和可视化工具可以帮助研究人员快速了解硬件能力，为实验规划和资源分配提供依据。

### 进一步探索方向

1.  **实际性能测试**: 除了理论性能，还可以添加实际基准测试（如矩阵乘法、卷积运算等）来评估GPU的真实表现

2.  **能效分析**: 加入功耗监测功能，评估不同GPU的能效比

3.  **温度监控**: 集成GPU温度监控，帮助优化散热和性能稳定性

4.  **多GPU协同**: 探索多GPU系统的性能特征和优化策略

通过这些扩展，可以构建一个更加全面的GPU性能分析和监控工具，为深度学习和科学计算工作提供更有力的支持。

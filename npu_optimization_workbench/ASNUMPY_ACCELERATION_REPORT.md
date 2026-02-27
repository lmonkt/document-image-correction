# AsNumpy 编译与加速测试报告

日期：2026-02-25

## 目标

在当前 Docker 容器中完成 `asnumpy` 源码编译安装，并基于其已实现模块编写可复现的 NPU 加速程序与测试结果。

## 安装过程

源码来源：`asnumpy-dtypes.zip`（已解压到 `asnumpy-dtypes/`）

按 README 流程执行：

1. `python3 -m pip install -r asnumpy-dtypes/requirements.txt`
2. `python3 -m build`（在 `asnumpy-dtypes/`）
3. `python3 -m pip install dist/*.whl`

## 关键问题与处理

### 问题 1：构建缺少 third_party 依赖

报错：`third_party/fmt/fmt-12.0.0.tar.gz` 缺失。

处理：补齐 `third_party/fmt` 与 `third_party/pybind11` 归档文件。

### 问题 2：pybind11 包损坏导致 CMake 指令缺失

报错：`Unknown CMake command "pybind11_add_module"`。

根因：`pybind11-3.0.1.tar.gz` 损坏（压缩流 EOF）。

处理：重新下载并校验 tar 包后重跑构建。

## 安装验证

验证命令（已通过）：

```bash
python3 - <<'PY'
import numpy as np
import asnumpy as ap

a=np.random.rand(64,64).astype(np.float32)
b=np.random.rand(64,64).astype(np.float32)

an=ap.ndarray.from_numpy(a)
bn=ap.ndarray.from_numpy(b)
cn=ap.multiply(an,bn)
print(float(ap.sum(cn)))
print(float(np.sum(a*b)))
PY
```

## 加速程序

文件：`npu_optimization_workbench/asnumpy_acceleration_benchmark.py`

覆盖两个已实现能力场景：

1. `multiply + sum`
2. `sin + cos + add + sum`

运行命令：

```bash
python3 npu_optimization_workbench/asnumpy_acceleration_benchmark.py \
  --rows 2048 --cols 2048 --warmup 2 --repeat 3
```

## 测试结果

结果文件：`npu_optimization_workbench/results/asnumpy_benchmark_20260225_083031.json`

- `multiply+sum`
  - NumPy: `6.355 ms`
  - AsNumpy: `3.651 ms`
  - 加速比：`1.741x`
  - 结果差值：`7.32e-04`

- `sin+cos+add+sum`
  - NumPy: `46.312 ms`
  - AsNumpy: `4.202 ms`
  - 加速比：`11.020x`
  - 结果差值：`1.10e-03`

## 与现有文档图像算法的关系

当前 `asnumpy` 不包含 `fft2/fftshift` 接口（已检查），所以不能直接完整替换当前倾斜校正主链路中的 FFT 部分。

建议：

- FFT 链路继续使用已验证的 `torch_npu` 实现。
- 对可分离的数学/规约子过程，可逐步尝试迁移到 `asnumpy`。

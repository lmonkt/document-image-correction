# NPU 适配工作台（torch_npu + asnumpy）

本目录用于在 Ascend NPU 环境下，对文档图像倾斜角估计算法做独立适配与基准测试。

## 目录结构

- `method_torch_npu.py`：`torch_npu` 版独立实现
- `method_numpy_reference.py`：NumPy 参考实现（用于对照）
- `run_benchmark.py`：统一 benchmark（速度 + 一致性）
- `asnumpy_acceleration_benchmark.py`：AsNumpy 加速演示基准（NumPy vs AsNumpy）
- `requirements_npu.txt`：NPU 依赖清单
- `results/`：基准结果 JSON
- `NPU_ADAPTATION_REPORT.md`：本次适配报告
- `ASNUMPY_ACCELERATION_REPORT.md`：AsNumpy 编译与测试记录

## 环境准备

当前容器已验证可用：

- Python `3.11.13`
- `torch==2.7.1`
- `torch_npu==2.7.1.post2`
- `npu-smi` 可见 `Ascend910B4`

若需补装：

```bash
python3 -m pip install -r npu_optimization_workbench/requirements_npu.txt
```

## 快速运行

单图推理：

```bash
python3 npu_optimization_workbench/method_torch_npu.py \
  assets/image-20260122211453-r856wkw.png --device npu:0
```

基准测试：

```bash
python3 npu_optimization_workbench/run_benchmark.py \
  --input assets --max-images 15 --warmup 2 --repeat 3 --device npu:0
```

## 当前实测结果

结果文件：`npu_optimization_workbench/results/benchmark_npu_20260225_075202.json`

- `numpy_ref`: `423.202 ms`
- `torch_npu(npu:0)`: `99.301 ms`
- `speedup`: `4.262x`
- `MAE`: `1.295e-06`
- `max_abs_diff`: `2.087e-06`

结论：在当前 Ascend 910B4 容器环境中，NPU 适配在保持角度结果一致的前提下，整体推理耗时显著下降。

## AsNumpy 加速演示

AsNumpy 采用源码编译方式安装（`python -m build` + `pip install dist/*.whl`），在当前容器已完成构建与安装验证。

运行命令：

```bash
python3 npu_optimization_workbench/asnumpy_acceleration_benchmark.py \
  --rows 2048 --cols 2048 --warmup 2 --repeat 3
```

结果文件：`npu_optimization_workbench/results/asnumpy_benchmark_20260225_083031.json`

- `multiply+sum`: `1.741x`
- `sin+cos+add+sum`: `11.020x`

说明：当前 `asnumpy` 暂不提供 `fft2/fftshift` 接口，因此本次先覆盖其已实现的数学运算模块加速；文档图像主链路 FFT 部分仍建议保留 `torch_npu` 方案。

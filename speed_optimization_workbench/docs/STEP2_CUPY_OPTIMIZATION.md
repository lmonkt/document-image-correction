# Step 2 - CuPy GPU 优化记录

日期：2026-02-24

## 目标

在 Step 1 基础上，将高耗时链路迁移到 GPU：

1. `np.fft.fft2` -> `cp.fft.fft2`
2. `np.fft.fftshift` -> `cp.fft.fftshift`
3. `np.abs/sum` -> `cp.abs/cp.sum`
4. 径向投影改为 GPU 广播 + gather + reduce

## 前置条件与安装

### 硬件与驱动

- GPU: NVIDIA GeForce RTX 3090
- Driver: 591.44
- `nvidia-smi` 显示 CUDA Version: 13.1

### 安装

```bash
/home/tjy1234/doc_img_corr/venv/bin/python -m pip install cupy-cuda12x
```

> 说明：即使 `nvidia-smi` 显示 13.1，`cupy-cuda12x` 在该驱动下可用。

## 改动说明

文件：`speed_optimization_workbench/jdeskew_variants.py`

### A. 新增 `get_angle_with_cupy`

流程：

1. CPU 侧完成 resize / 灰度 / adaptiveThreshold
2. 上传到 GPU（`cp.asarray`）
3. 在 GPU 做 FFT + shift + abs
4. 在 GPU 做径向索引广播、gather、sum
5. 取 argmax 并回传标量角度

### B. 同步计时

- 关键点调用 `cp.cuda.Stream.null.synchronize()`
- 避免异步导致计时偏小

### C. 修复点

- 初次实现中 `cp.int32(array)` 导致隐式 NumPy 转换错误
- 已改为 `array.astype(cp.int32)`，保证纯 GPU 张量路径

## 基准命令

```bash
/home/tjy1234/doc_img_corr/venv/bin/python speed_optimization_workbench/run_benchmark.py \
  --stage cupy --input assets --max-images 15 --warmup 2 --repeat 3
```

## 结果

结果文件：`speed_optimization_workbench/results/benchmark_cupy_20260224_201336.json`

- original: `224.354 ms`
- cpu_numpy_vec: `212.409 ms`（`1.056x`）
- cpu_numba_jit: `179.368 ms`（`1.251x`）
- cupy: `24.593 ms`（`9.123x`）

分阶段观察（cupy）：

- FFT：`21.189 ms`
- 径向投影：`1.947 ms`

## 结论

1. GPU 迁移在当前环境下收益显著，已达到数量级提速。
2. FFT 迁移是最大贡献点，径向投影 GPU 向量化也有明显收益。
3. 当前版本仍有 CPU 前处理（灰度/阈值），后续可继续评估是否迁移。

## 风险与注意

- 小图/单图场景可能受 CPU<->GPU 拷贝影响。
- 需持续检查 CPU 与 GPU 版本的角度数值一致性（允许微小浮点误差）。
- 若扩展到多进程 + GPU，需要重新设计并发策略，避免同卡争用。

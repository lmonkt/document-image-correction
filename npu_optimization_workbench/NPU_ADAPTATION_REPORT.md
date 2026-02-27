# 文档图像角度估计 NPU 适配报告

日期：2026-02-25

## 目标

将现有 CuPy/GPU 加速思路迁移到 Ascend NPU，使用 `torch_npu` 完成独立实现，并给出可复现实测结果。

## 适配范围

新建目录：`npu_optimization_workbench/`

新增内容：

1. `method_torch_npu.py`
2. `method_numpy_reference.py`
3. `run_benchmark.py`
4. `requirements_npu.txt`
5. `results/benchmark_npu_20260225_075202.json`

## 实现说明

核心流程保持与原算法一致：

1. `resize`
2. `gray + adaptiveThreshold`
3. 频域变换：`torch.fft.fft2 + torch.fft.fftshift`
4. 径向投影：广播构造 `(theta, radius)` 索引后 gather + reduce
5. `a_init / a_correct` 判定输出

计时口径：

- `avg_ms`
- `p95_ms`
- `fft_avg_ms`
- `proj_avg_ms`
- `speedup_vs_numpy_ref`

一致性口径：

- `mae`
- `max_abs_diff`

## 关键兼容性问题与修复

问题：Ascend `torch_npu` 当前不支持对 `complex64` 直接执行 `torch.abs`。

报错关键字：`Tensor self not implemented for DT_COMPLEX64`。

修复：在 NPU 版 FFT 幅值计算中改为：

`magnitude = sqrt(real^2 + imag^2)`

修复后 benchmark 正常完成。

## 测试命令

```bash
python3 npu_optimization_workbench/run_benchmark.py \
  --input assets --max-images 15 --warmup 2 --repeat 3 --device npu:0
```

## 测试环境

- OS: Ubuntu 22.04.5 LTS（Docker）
- Python: 3.11.13
- torch: 2.7.1
- torch_npu: 2.7.1.post2
- NPU: Ascend910B4

## 测试结果

结果文件：`npu_optimization_workbench/results/benchmark_npu_20260225_075202.json`

- `numpy_ref`
  - avg: `423.202 ms`
  - p95: `666.499 ms`
  - FFT: `357.799 ms`
  - Projection: `63.962 ms`

- `torch_npu(npu:0)`
  - avg: `99.301 ms`
  - p95: `151.675 ms`
  - FFT: `52.581 ms`
  - Projection: `45.544 ms`

- 对比
  - 速度提升：`4.262x`
  - MAE：`1.295e-06`
  - max_abs_diff：`2.087e-06`

## 结论

1. `torch_npu` 方案在当前环境中已可替代 CuPy 路线用于 NPU 侧加速。
2. 在 15 张样本、45 次推理下，端到端平均耗时由 `423.202 ms` 降到 `99.301 ms`。
3. 与 NumPy 参考实现的角度结果误差保持在 `1e-6` 量级，满足一致性要求。

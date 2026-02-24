# jdeskew 速度优化工作台

本目录用于**逐步**推进 `jdeskew exp` 算法的速度优化，并保留完整过程记录。

## 目录结构

- `jdeskew_variants.py`：算法多实现（original / cpu_numpy_vec / cpu_numba_jit / cupy）
- `run_benchmark.py`：统一基准脚本（同参数、同图片、同统计口径）
- `methods/`：四种方法的独立可导入实现（互不依赖，可单独运行）
- `docs/`：每一步改动文档（为什么改、改了什么、风险点）
- `results/`：每一步基准结果 JSON

`methods/` 目录说明：

- `method_original.py`
- `method_cpu_numpy_vec.py`
- `method_cpu_numba_jit.py`
- `method_cupy.py`
- `example_usage.py`（同图对比示例）

运行示例：

```bash
/home/tjy1234/doc_img_corr/venv/bin/python -m speed_optimization_workbench.methods.example_usage \
  --image assets/image-20260122211453-r856wkw.png
```

## 基准原则

1. 使用统一参数：`amax=45, V=2048, W=304, D=0.55`
2. 使用统一输入：`assets`（本次取前 15 张）
3. 每图重复 `repeat=3`，预热 `warmup=2`
4. 指标统一：`avg_ms`、`p95_ms`、`fft_avg_ms`、`proj_avg_ms`、`speedup_vs_original`

## 步骤记录

### Step 1：CPU 先提速（NumPy 向量化 + 减少重复分配 + Numba JIT）

- 结果文件：`results/benchmark_cpu_20260224_201050.json`
- 变更文档：`docs/STEP1_CPU_OPTIMIZATION.md`

关键结果（avg_ms，越小越好）：

- original: `218.026 ms`
- cpu_numpy_vec: `207.202 ms`（`1.052x`）
- cpu_numba_jit: `183.703 ms`（`1.187x`）

结论：CPU 侧最有效的是 Numba JIT 径向投影，FFT 仍是主瓶颈。

### Step 2：CuPy GPU 化（FFT + 径向投影广播+reduce）

- 结果文件：`results/benchmark_cupy_20260224_201336.json`
- 变更文档：`docs/STEP2_CUPY_OPTIMIZATION.md`

关键结果（avg_ms，越小越好）：

- original: `224.354 ms`
- cpu_numba_jit: `179.368 ms`（`1.251x`）
- cupy: `24.593 ms`（`9.123x`）

结论：在当前 RTX 3090 环境下，GPU 方案收益显著，且 FFT 加速贡献最大。

## 复现实验命令

### 1) CPU 阶段

```bash
/home/tjy1234/doc_img_corr/venv/bin/python speed_optimization_workbench/run_benchmark.py \
  --stage cpu --input assets --max-images 15 --warmup 2 --repeat 3
```

### 2) CuPy 阶段

```bash
/home/tjy1234/doc_img_corr/venv/bin/python speed_optimization_workbench/run_benchmark.py \
  --stage cupy --input assets --max-images 15 --warmup 2 --repeat 3
```

## 环境记录（本次）

- Python: 3.10.12 (venv)
- GPU: NVIDIA GeForce RTX 3090
- Driver: 591.44
- NVIDIA-SMI 显示 CUDA Version: 13.1
- 新增依赖：`numba`、`cupy-cuda12x`

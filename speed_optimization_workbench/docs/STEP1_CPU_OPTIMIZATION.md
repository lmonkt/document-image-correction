# Step 1 - CPU 优化记录

日期：2026-02-24

## 目标

在不改动算法判定逻辑的前提下，优先做低风险 CPU 提速：

1. NumPy 向量化
2. 减少重复分配
3. Numba JIT 径向投影

## 改动说明

文件：`speed_optimization_workbench/jdeskew_variants.py`

### A. 新增 `cpu_numpy_vec`

将径向投影从“逐角度 Python 循环 + 高频索引”改为：

- 角度批量广播
- 生成 2D 索引矩阵（`y`, `x_proj`）
- 一次性 gather
- 沿 axis=1 做 reduce（`sum`）

收益点：减少 Python 解释器循环开销，充分使用 NumPy 向量化。

### B. 新增 `cpu_numba_jit`

对径向投影核心循环使用 `@njit(cache=True, fastmath=True)` 编译。

收益点：

- 保留易理解的 for-loop 结构
- 把热点循环下沉为机器码
- 显著降低 `radial_projection` 阶段耗时

### C. 保持一致性

- 仍使用相同参数：`amax=45, V=2048, W=304, D=0.55`
- 仍输出 `a_init/a_correct` 后按 `D` 判定
- 不改动 FFT 流程（便于隔离 CPU 径向优化效果）

## 基准命令

```bash
/home/tjy1234/doc_img_corr/venv/bin/python speed_optimization_workbench/run_benchmark.py \
  --stage cpu --input assets --max-images 15 --warmup 2 --repeat 3
```

## 结果

结果文件：`speed_optimization_workbench/results/benchmark_cpu_20260224_201050.json`

- original: `218.026 ms`
- cpu_numpy_vec: `207.202 ms`（`1.052x`）
- cpu_numba_jit: `183.703 ms`（`1.187x`）

分阶段观察：

- FFT 平均耗时基本不变（仍约 167~170ms）
- 径向投影：
  - original: `46.256 ms`
  - cpu_numpy_vec: `35.779 ms`
  - cpu_numba_jit: `12.150 ms`

## 结论

1. CPU 优化已验证有效，尤其 Numba JIT 对径向投影提升明显。
2. 但总体提速受限于 FFT 主瓶颈未动。
3. 下一步应优先 GPU 化 FFT，再评估是否继续深挖 CPU 侧。

## 风险与注意

- Numba 首次运行包含 JIT 编译开销，需 warmup 后看稳定值。
- 向量化版本会占用更多瞬时内存（2D 索引矩阵）。

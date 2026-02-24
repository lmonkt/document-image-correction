# jdeskew exp 去倾斜算法性能分析（单进程基线）

适用脚本：`for_eval_deskew/jdeskew_exp_eval.py`（算法逻辑）+ `for_eval_deskew/jdeskew_exp_single_process_profile.py`（性能分析）

---

## 1. 分析目标与范围

本次分析只回答一个问题：

> **校正一张图片时，时间到底花在了哪里？**

用于支撑后续“是否值得迁移到 GPU”的决策。

本轮刻意使用**单进程**，避免多进程调度噪声，重点观察算法本体热点。

---

## 2. 环境与依赖

- OS: Linux
- Python: `3.10.12`（项目 `venv`）
- 新增依赖：
  - `py-spy`
  - `line_profiler`

安装命令（已执行）：

```bash
/home/tjy1234/doc_img_corr/venv/bin/python -m pip install py-spy line_profiler
```

---

## 3. 新增脚本说明

新增文件：`for_eval_deskew/jdeskew_exp_single_process_profile.py`

功能：

1. 单进程串行执行（无 `multiprocessing`）
2. 分阶段计时：
   - `read_s`（读图）
   - `resize_s`
   - `gray_s`
   - `optimal_square_s`
   - `adaptive_threshold_s`
   - `fft_and_magnitude_s`
   - `radial_projection_s`
   - `decision_s`
3. 输出统计（mean / P95）并可保存 JSON
4. 兼容：
   - `py-spy` 火焰图
   - `line_profiler`（`kernprof -l -v`）

---

## 4. 复现实验步骤

### 4.1 单进程分阶段统计

```bash
/home/tjy1234/doc_img_corr/venv/bin/python for_eval_deskew/jdeskew_exp_single_process_profile.py \
  --input assets \
  --max-images 10 \
  --warmup 2 \
  --repeat 3 \
  --save-json
```

产物：

- `for_eval_deskew/profile_reports/jdeskew_exp_single_profile_20260224_194304.json`

### 4.2 py-spy 火焰图

```bash
/home/tjy1234/doc_img_corr/venv/bin/py-spy record \
  -o for_eval_deskew/profile_reports/jdeskew_exp_pyspy_flamegraph.svg \
  -- /home/tjy1234/doc_img_corr/venv/bin/python for_eval_deskew/jdeskew_exp_single_process_profile.py \
  --input assets --max-images 20 --warmup 1 --repeat 3
```

产物：

- `for_eval_deskew/profile_reports/jdeskew_exp_pyspy_flamegraph.svg`

### 4.3 line_profiler 逐行统计

```bash
/home/tjy1234/doc_img_corr/venv/bin/kernprof -l -v \
  for_eval_deskew/jdeskew_exp_single_process_profile.py \
  --input assets --max-images 8 --warmup 1 --repeat 2
```

产物：

- `jdeskew_exp_single_process_profile.py.lprof`

如果一年后需要重新查看 `.lprof`：

```bash
/home/tjy1234/doc_img_corr/venv/bin/python -m line_profiler jdeskew_exp_single_process_profile.py.lprof
```

---

## 5. 本次结果（可直接用于决策）

> 以下数值来自 `jdeskew_exp_single_profile_20260224_194304.json`（10 张图、每图 3 次重复）

### 5.1 端到端与算法总耗时

- 端到端平均：`228.31 ms / 图`
- 算法平均：`219.54 ms / 图`
- 读图平均：`7.66 ms / 图`

说明：瓶颈在算法，不在 I/O。

### 5.2 算法内部耗时占比（按平均耗时）

以 `algo_total_s_mean_ms = 219.54 ms` 为基准：

- `fft_and_magnitude_s`: `149.37 ms`（`68.04%`）
- `radial_projection_s`: `47.38 ms`（`21.58%`）
- `adaptive_threshold_s`: `16.51 ms`（`7.52%`）
- 其余（resize / gray / pad / decision）：合计约 `2.86%`

结论：

1. **第一热点：FFT + shift + abs（绝对主瓶颈）**
2. **第二热点：径向投影循环（次瓶颈）**
3. 自适应阈值有成本，但远小于前两项

### 5.3 line_profiler 关键逐行结论

- `get_fft_magnitude_timed`：
  - `np.fft.fft2` 为最大耗时行（约 71%）
  - `np.fft.fftshift`、`np.abs` 次之
- `get_angle_adaptive_timed`：
  - 循环内的索引采样与 `np.sum` 是主要耗时
  - 尤其是 `vals = m[y[valid], x_proj[valid]]` 与后续求和
- `get_angle_timed`：
  - `get_fft_magnitude_timed` + `get_angle_adaptive_timed` 基本占满总时间

---

## 6. 对“迁移 GPU”的可行性判断

### 6.1 结论（当前阶段）

**可行，且优先级高。**

原因：

- 最大头部耗时在 FFT（天然适合 GPU）
- 次热点是大规模向量化采样与求和（也适合 GPU 并行）

### 6.2 建议的改造优先级

1. **优先迁移 FFT 链路**（`adaptiveThreshold` 之后到 `magnitude`）
2. **其次迁移径向投影积分**（角度循环 + 采样 + 累加）
3. 最后再评估 `adaptiveThreshold` 是否迁移（收益可能小于前两步）

### 6.3 风险与注意点

- 若单张图很小，GPU 拷贝开销可能吃掉收益
- 需避免 CPU/GPU 来回拷贝（一次上卡，尽量在卡上完成后续）
- 若未来仍走多进程，需重新评估“多进程 + GPU”的资源竞争策略

---

## 7. 一年后复盘时的最小操作手册

1. 跑基线：

```bash
/home/tjy1234/doc_img_corr/venv/bin/python for_eval_deskew/jdeskew_exp_single_process_profile.py --input assets --max-images 20 --warmup 2 --repeat 5 --save-json
```

2. 看火焰图：

```bash
/home/tjy1234/doc_img_corr/venv/bin/py-spy record -o for_eval_deskew/profile_reports/jdeskew_exp_pyspy_flamegraph.svg -- /home/tjy1234/doc_img_corr/venv/bin/python for_eval_deskew/jdeskew_exp_single_process_profile.py --input assets --max-images 20 --warmup 1 --repeat 3
```

3. 看逐行：

```bash
/home/tjy1234/doc_img_corr/venv/bin/kernprof -l -v for_eval_deskew/jdeskew_exp_single_process_profile.py --input assets --max-images 8 --warmup 1 --repeat 2
```

判定标准（简化）：

- 若 `fft_and_magnitude` 仍 > 60%，先做 FFT GPU 化
- 若 `radial_projection` 仍 > 20%，再做径向投影 GPU 化

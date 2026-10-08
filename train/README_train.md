# PINN-SPM 训练复现说明

本目录包含对仓库根目录两个预训练 PINN（`SPM NE` / `SPM PE`，CIDETEC 原版只发布推理端、无训练代码）的**完整训练复现**，以及独立的数值参考解与三方验证。

## 1. 复现结果（最终验证数据）

**浓度场精度（vs 独立数值参考解，全参数域 9 bc × 100 t × 32 r 网格）**

| 电极 | 原模型 RMS / max | 重训模型 RMS / max | 结论 |
|---|---|---|---|
| NE | 0.0205 / 0.0981 | 0.0286 / **0.0690** | max 误差好 30%，RMS 略差 |
| PE | 0.0458 / 0.1632 | **0.0267 / 0.1032** | 两项均优 |

**放电电压精度（核心物理窗口 x_p≤0.95、x_n≥0.05 内，vs 参考解组装的电压）**

| 工况 | 原模型 RMS / max (mV) | 重训模型 RMS / max (mV) |
|---|---|---|
| 默认几何 1C | 49 / 176 | **45 / 104** |
| 默认几何 3C | inf / inf（窗口内即发散） | **25 / 81** |

原模型在 3C 曲线的物理窗口内就产生 NaN/inf（x_surf 提前越界 → 过电位爆炸）；重训模型 NaN 点更少且更晚。重训版 max 偏差出现在放电末端 x_p≈0.99 的 OCV 陡降段，属该模型类的固有难点。

**物理自检（推理模式）**：重训模型 1C/3C 处 PDE 相对残差 4–6%（原模型 7–18%），IC 误差 ≤0.06（原模型达 0.09–0.15）。

## 2. 环境（原生 Windows GPU 路径，已验证）

```bash
conda create -n pinn210 python=3.10 -y
conda install -n pinn210 -c conda-forge cudatoolkit=11.2 cudnn=8.1.0 -y
conda run -n pinn210 pip install tensorflow==2.10.1 numpy==1.23.5 protobuf==3.19.6 scipy matplotlib
```

- TF ≥2.11 无原生 Windows GPU 支持，2.10 是最后支持版本；RTX 3060（sm_86）+ 驱动 617.14 向下兼容 CUDA 11.2。
- GPU 验证：`conda run -n pinn210 python train/gpu_check.py`。本仓库网络极小（187 参数），eager 模式下 Python 开销主导（CPU 47ms/步 vs GPU 60ms/步），CPU/GPU 训练结果等价。
- 注意：直接用 `envs/pinn210/python.exe` 启动时 CUDA DLL 不在 PATH 上会回退 CPU；需要 GPU 时用 `conda run -n pinn210` 或先激活环境。

## 3. 复现流程（按序执行）

```bash
# 0) 原仓库回归基线（5 个 Notebook 示例 + 数值基线）
conda run -n pinn210 python train/regression_baseline.py
# 1) 独立数值参考解：自检（质量守恒/网格收敛）+ 生成真值数据集
conda run -n pinn210 python train/reference_spm.py selftest
conda run -n pinn210 python train/reference_spm.py gen
# 2) 对原模型探针（锁定训练约定：PDE 形式 / BC / t_max 角色 / 外推边界）
conda run -n pinn210 python train/probe_original.py
# 3) 训练（NE、PE 各一；步数可调，45k 步约 50 分钟/电极）
conda run -n pinn210 python train/train_spm.py ne 45000
conda run -n pinn210 python train/train_spm.py pe 45000
# 4) 三重验证：物理自检 + 参考解对比 + 端到端电压回归（原模型 vs 重训）
conda run -n pinn210 python train/validate.py
conda run -n pinn210 python train/voltage_ref_cmp.py   # 电压三方对比（含参考解）
```

重训模型输出在 `train/artifacts/models/SPM NE|PE`（与原版同格式的 TF SavedModel）。要让 Notebook 用重训模型，把该目录覆盖/替换根目录的 `SPM NE`/`SPM PE` 即可（`tf.keras.models.load_model` 直接兼容）。

## 4. 哪些是确定的（仓库证据 + 探针实证，1:1 重建）

1. **架构**（`keras_metadata.pb`）：Input(4)→BN→Dense(10,ELU)→Dense(10,ELU)→Dense(1)，float64，187 参数。
2. **输入输出约定**（`PINN_utils.py` 推理路径 + 探针实证）：输入 `[τ, ρ, bc_minmax, t_max_minmax]`；输出 x̄，x = x̄+x₀；归一化区间与 `Inputs_n/p` 的 min/max 精确一致（`reference_spm.BC_RANGE` 现按同式自动计算）。
3. **物理**（探针实证，原模型以 1–5% 精度满足）：无量纲球扩散 ∂x/∂τ = κ·∇²_ρx，κ=t_max·D_s/R_s²；表面 BC ∂x/∂ρ|₁=bc；r→0 对称；满电初值（x_n0=0.88 / x_p0=0.32，仅放电）。
4. **参数域**：厚度 50–200 μm、孔隙率 0.2–0.6、1C–3C（等效 bc 区间 ≈0.1C–3C）；t_max=3900/h。
5. **bc 端点溯源**：`bc_p_limit=0.176902` = 用 Q_pos/3600 算出的 1C bc_p（已数值验证）。

## 5. 哪些是不确定的（原作未公开，本复现的自定选择）

| 项 | 本复现的选择 |
|---|---|
| 损失构成/权重 | w_pde·(∇²_ρx−∂τx/κ)² + w_ic·x̄(0)² + w_bc·(∂ρx|₁−bc)² + w_sym·(∂ρx|ρmin)²；权重 1:4:1:1（PDE 残差按 κ 归一） |
| 配点策略 | bc/τ/ρ 独立均匀采样 2048/步 + BC 256 + 对称 256 + IC 1024，逐步重采样 |
| 优化 | Adam，1e-3 →(22.5k)→ 3e-4 →(45k)→ 1e-4，共 45k 步 |
| t_max 独立性 | 探针显示原模型对 t_max 有独立响应 → 训练时 t_max 由 bc 导出并加 ±5% 抖动 |
| 随机种子 | 0（非凸问题，只能复现物理等价解，无法逐权重复现） |

## 6. 实现要点与踩坑记录（复现者必读）

1. **BN 无二阶梯度**：Keras BN 的 fused 算子不支持二阶自动微分 → 两阶段训练：先 500 批标定 moving statistics，再冻结为推理模式（training=False）做物理训练。与部署行为一致。
2. **TF 2.10 eager tape 怪癖**：二阶梯度 `t2.gradient` 必须在 t2 上下文内部调用（外部调用返回 None），损失组装在 t2 内、权重梯度由外层 tape 承担（见 `train_spm.py` 注释）。
3. **归一化方向 bug（曾犯两次）**：t_max 的 min-max 归一化端点顺序在 NE/PE 间相反（NE 的 bc 区间下端对应大 |bc|，PE 相反），`norm_cols` 必须用方向无关的 min/max。训练内部自洽不代表部署约定正确——**最终验收必须以部署管线的评估为准**。
4. **训练中 `_infer` 的 tf.function 缓存**：类级 `@tf.function` 首次 trace 后绑定图，替换 `self.model` 不会生效；评估保存后的模型时直接调用。
5. 机器休眠会杀死后台训练进程；PE 曾在 30k/45k 步中断后重训。

## 7. 产物结构

```
train/
├── reference_spm.py        独立数值参考解（零依赖原仓库；质量守恒 ~1e-16，网格收敛 ~1e-4）
├── probe_original.py       原模型探针实验（约定实证）
├── train_spm.py            训练脚本（NE/PE）
├── validate.py             三重验证（物理自检 / 参考解对比 / 电压回归）
├── voltage_ref_cmp.py      电压三方对比（原模型 vs 重训 vs 参考解）
├── regression_baseline.py  原仓库回归基线
├── gpu_check.py / bench_device.py
├── README_train.md         本文件
└── artifacts/
    ├── baseline/           原模型基线图 + baseline.json
    ├── reference_dataset.npz   数值真值（9 bc × 100 t × 32 r / 电极）
    ├── models/SPM NE|PE    重训后的 SavedModel + train_report_{ne,pe}.json
    └── validation/         验证报告 + 电压对比图
```

## 8. 已知事实（复现过程中发现）

- 原 PINN 是个"松"PINN：PDE 相对残差 6–60%，IC 在 bc 区间角点处误差达 9–15%，PE 在 3C 端 ic_max 达 0.15–0.79。
- 原模型在 3C 放电的物理窗口内电压即发散（NaN/inf）；仓库 `plot_Battery` 对越界 bc 用硬边界直接拒绝（Out of bounds）。
- bc 越界 ±20% 外推：输出仍有限、单调，但精度无保证。

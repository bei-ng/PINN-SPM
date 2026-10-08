# 万能族 PINN（τ 参数化）—— 训练与使用说明

本目录是**独立于 `train/` 的第二套训练管线**，实现"路径 A"重训：把时间输入从
`t̂ = t/t_max` 换成 `τ̂ = τ/τ_ref`（τ = t·D_s/R_s²），使解族与颗粒半径 R_s、扩散系数
D_s **彻底解耦**。两套目录互不依赖、互不影响，只共用 `data/` 与 conda 环境 `pinn210`。

## 1. 数学设计

SPM 在无量纲坐标 (τ = t·D_s/R_s², ρ = r/R_s) 下：

```
∂x/∂τ = (1/ρ²)·∂/∂ρ(ρ²·∂x/∂ρ)，  表面 BC：∂x/∂ρ|₁ = bc，  初值：x(r,0) = x₀
```

**D_s 和 R_s 从方程中完全消失**——所有颗粒尺寸、扩散系数的解都是同一个万能族
F(τ̂, ρ; bc)。R_s、D_s 只通过两个外部解析映射进入：

```
τ̂ = τ/τ_ref，  τ_ref = 3900·D_s_ref/R_s_ref²     （每电极固定常数）
bc = ±R_s²·I / (D_s·3·ε_s·L·c_max·F·A)，  I = min(Q_n,Q_p)/3600·C_rate
```

物理时间带：放电自然终点满足 **τ_end·bc ≈ K = bc_ref·τ_ref**（每电极常数）。

**网络输入取时间带分数 s**（而非 τ̂ 本身）：`s = τ̂/τ̂_band = τ̂·h/1.1 ∈ [0,1]`，
其中 h = |bc|/|bc_ref|。原因：(s, ρ, bc) 与 (τ̂, ρ, bc) 信息等价（τ̂ = s·1.1/h 可逆
恢复），但所有输入特征尺度都是 O(1)——若直接喂 τ̂，其范围跨 [0, 55]，h=15 的
整个放电压缩在 τ̂∈[0,0.073] 内，10-10 小网络无法同时分辨两种尺度（实测 RMS 差
一个数量级）。PDE 残差在 s 坐标下为：

```
res = x_s − (1.1·τ_ref/h)·(x̂_ρρ + (2/ρ)·x̂_ρ) = 0
```

与原方程数学等价（按 h 重加权，使解误差在全域均匀）。BC 用**原始绝对残差**
`(∂ρx|₁ − bc)²`——通量误差对解的影响是绝对量级的，按 h 归一（曾实测）会允许
大 bc 处出现与 bc 同量级的绝对通量误差，污染中段精度。

| 常数 | NE | PE |
|---|---|---|
| bc_ref | −0.115617 | +0.176902 |
| τ_ref = 3900·D/R² | 2.3147 | 1.3677 |
| K = τ_ref·\|bc_ref\| | 0.26762 | 0.24194 |

## 2. 采样域

- **bc**：|bc|/|bc_ref| ∈ [0.02, 15]，log-uniform——覆盖 R_s ∈ [0.5,2]× ×
  C_rate ∈ [0.1,3] 的乘积域 [0.025, 12]×（bc ∝ R_s²·C_rate）并留余量
- **s（时间带分数）**：[0, 1] 均匀（对应每 bc 的时间带 τ̂ = s·1.1/h）
- **ρ**：[0.01, 1]（下限取 0.01，覆盖 R_s=2× 时的 ρ_min≈0.0122；比旧管线的
  0.0244 更靠近中心，对称条件更准确）
- 网络架构：Input(3)→BN→Dense(10,ELU)×2→Dense(1)，float64（同构缩维，173 参数）；
  BN 两阶段标定+冻结；TF 2.10 tape 怪癖的三层 tape 结构与 `train/` 相同

## 3. 推理映射（给反演/优化循环的接入公式）

给定设计点 `(L_j, ε_j, C_rate, R_s_j, D_s_j)` 与查询时间 t：

```python
# 1) 电流与 bc（每电极）
I_j = min(Q_n, Q_p)/3600 · C_rate          # I 两极相同，容量与 R_s 无关
bc_j = ±R_s_j²·I / (D_s_j·3·ε_s_j·L_j·c_max_j·F·A)
# 2) 时间映射
h_j = |bc_j|/|bc_ref_j|；  τ̂ = t·D_s_j/R_s_j²/τ_ref_j；  s = τ̂·h_j/1.1
# 3) 查网络（输入 [s, ρ, bc_norm]，bc_norm = (|bc|/|bc_ref| − 0.02)/14.98）
x̄_j(s, ρ, bc_norm) → x_j = x̄ + x₀_j → c_j = x_j·c_max_j
```

**R_s、D_s、I、L、ε、C-rate 全部是解析映射的自由参数**——不再需要为换颗粒尺寸重训。

## 3.1 现成推理接口（`predict_uni.py`）

输入任意设计参数组合得到放电/充电曲线（充电见 3.2），**未提供的参数自动用默认电芯参数**：

```python
from train_universal.predict_uni import plot_Battery_uni

r = plot_Battery_uni()                                    # 全默认（参考电芯 1C）
r = plot_Battery_uni(thickness_n=200e-6, porosity_p=0.4, C_rate=2,
                     R_s_n=1.64e-5, D_s_p=5e-15)          # 任意组合，缺省自动补
```

- 可调参数：`thickness_n/p`（m）、`porosity_n/p`、`C_rate`、**`R_s_n/p`（m）、
  `D_s_n/p`（m²/s）**——后两个是万能族新解锁的自由参数
- 返回 dict：`t, V, x_n_surface, x_p_surface, I, bc_n, bc_p, t_end_n, t_end_p,
  limiting, envelope_ok, violations, design`（生效参数回显）
- bc 超出训练包络 [0.02, 15]×bc_ref 时打印警告并计入 `violations`（仍计算，
  外推精度无保证）；`strict=True` 改为返回 None
- `experimental={'# Time [s]', 'Voltage [V]'}` 可叠加实验曲线；`plot=False` 纯计算

## 3.2 充电接口（`mode='charge'`，镜像法零重训）

SPM 的 PDE（常数 D_s）、常通量 BC、均匀初值全部线性 → **反接电流的充电解是放电解的
镜像**：网络输出偏移 u 取反，`x_ch = stoi0 − u`（放电 `x_dis = x0 + u`），共用同一
网络。时间带不变（充电满摆幅与放电等幅，τ_end·|bc| ≈ K 方向对称）；电流取负号进
Butler–Volmer（`V = OCV_pe − OCV_ne + eta_p − eta_n` 方向自动翻转）。

```python
r = plot_Battery_uni(mode='charge')                       # 默认电芯 1C 充电
r = plot_Battery_uni(mode='charge', C_rate=0.5,
                     R_s_n=1.5 * 8.21e-6, D_s_p=0.7e-14)  # R_s/D_s 同样自由
```

- 初值 = 0% SOC 均匀态 `stoi0`（params JSON 已有：NE 0.139 / PE 0.99，与放电用
  `stoi1` 起步对称）；OCV 分支恰好方向一致（NE 充电枝 ch_GrSi + PE 放电枝 dch_NMC811）
- 终止条件：`V ≥ V_cutoff`（默认 4.2 V，可调），返回曲线截断到越过点；
  dict 新增 `mode / V_cutoff / cutoff_reached / t_cutoff`；`I、bc_n、bc_p` 带
  充电符号（I 为负、bc_n 为正、bc_p 为负）
- **只有充电方向需要重训的情形**：引入破坏线性/对称性的物理（浓度相关 D_s、析锂、
  电解液/SEI、方向迟滞 OCV）时，镜像法失效，才需要按充电工况重训
- Notebook 演示：根目录 `PINN_uni notebook.ipynb`（`show_discharge_charge()` 助手
  把同一设计点的充放电曲线画到一张图，输出已内嵌）

**充电验证**（`test_charge.py`，vs 独立参考解 `solve(±bc, x_init=stoi0)`，4 工况）：

| 工况 | x_n RMS/max | x_p RMS/max | V RMS/max 全曲线 | V RMS/max t>0 | t_cutoff PINN/ref |
|---|---|---|---|---|---|
| 默认 1C | 0.034/0.044 | 0.031/0.037 | 53/254 mV | 46/73 mV | 3683/3553 s |
| R_s×1.5, 2C | 0.010/0.064 | 0.011/0.069 | 63/464 mV | 12/76 mV | 1192/1192 s |
| R_s×0.7, 0.5C | 0.045/0.056 | 0.050/0.061 | 73/160 mV | 73/160 mV | 8407/8060 s |
| R_s×2, 3C（包络角）| 0.049/0.160 | 0.070/0.199 | 134/622 mV | 68/172 mV | 404/448 s |

场误差与放电侧同源同量级（镜像不放大误差）。全曲线 max 集中在 **t=0 单点**：网络
IC 误差（~0.01–0.02）经 PE 低 SOC 端（x_p≈0.99）的陡 OCV 斜率放大为数百 mV 初值
偏移（0.5C 工况该处 OCV 平缓，仅 −6 mV）；t>0 后电压 RMS 12–73 mV，与放电侧电压
精度（80–136 mV）相当。截止时刻偏差 −10%～+4%。对比图
`artifacts/charge_cmp_1..4.png`。

## 4. 训练与验证

```bash
conda run -n pinn210 python train_universal/reference_spm_uni.py selftest  # 质量守恒 + 万能族数学验证
conda run -n pinn210 python train_universal/train_spm_uni.py ne 45000
conda run -n pinn210 python train_universal/train_spm_uni.py pe 45000
conda run -n pinn210 python train_universal/validate_uni.py                # 五重验证
```

五重验证内容：
① 物理自检（IC/表面BC/PDE残差/对称，推理模式）
② (R_s × C_rate) 网格 vs 参考解：R_s ∈ {0.5,0.7,1,1.5,2}× × C ∈ {0.2,1,3}，15 组/电极
③ 万能族一致性：不同 (R_s, C) 组合产生相同 bc（如 (1×,1.5C)、(1.22×,1C)、(1.73×,0.5C)）
   → 参考解互相重合（数学），网络分别匹配各自参考解（学习）
④ 与旧 4-input 模型在参考 R_s 处交叉对比（族的一致性 sanity）
⑤ 电压组装抽查：默认几何 1C、R_s ∈ {0.7, 1.0, 1.5}×，万能族电压 vs 参考解电压

## 4.1 本版验证结果（45k 步，30-30 网络）

**② (R_s × C_rate) 网格 vs 参考解**（每电极 15 组，共 30 组全部通过）：

- 场误差 RMS 0.003–0.054、max 0.028–0.199（x 单位）
- 规律：大 bc（大颗粒/高倍率）端更准（RMS 0.003–0.02），小 bc 角（R_s×0.5、C=0.2，
  bc≈0.25×bc_ref）最差（RMS ~0.05）；极端角（R_s×2、C=3，bc≈12×bc_ref）max 0.16–0.20
- **R_s 外推成立**：固定 C=1，R_s 从 0.5× 到 2×，精度无系统性劣化

**③ 万能族一致性**：不同 (R_s, C) 组合产生相同 bc → 参考解互相重合至 2.6e-4/5.7e-4
（数学验证 ✓）；网络对每组的偏差 ≤0.051/0.064（与 ② 一致）。

**④ 与旧 4-input 模型交叉**（参考 R_s）：PE 一致性极好（RMS 0.015）；NE RMS 0.062
（最大偏差 0.26 出现在旧模型已知的放电末端弱点，万能版在该处更贴近参考解）。

**① 物理自检**：PDE 残差 rms 0.044/0.046；IC 误差 0.080/0.095（比旧专用模型
0.02–0.07 偏大——宽 bc 域的代价，改进杠杆：调高 w_ic 或 N_IC 后重训）。

**⑤ 电压抽查**（默认几何 1C，物理窗口内 vs 参考解）：RMS 80–136 mV——约为旧
专用模型的 2 倍（IC 误差偏大 + 小 bc 角精度），换来的是 R_s/D_s 全域自由。

**训练成本**：45k 步/电极，~50 分钟 CPU（30-30 网络，配点 3584/步）。

## 5. 与旧管线（train/）的关系

| | train/（4 输入） | train_universal/（3 输入） |
|---|---|---|
| 时间输入 | t̂ = t/t_max（t_max 由 bc 导出） | τ̂ = t·D_s/R_s²/τ_ref |
| R_s / D_s | 固定材料参数 | **推理时自由** |
| 方程系数 | κ = t_max·D/R²（随 bc 变化 30×） | 无（残差对一切 bc 同形） |
| bc 域 | ~[0.1, 3]×（等效倍率） | ~[0.02, 15]×（log-uniform） |
| 产物兼容性 | PINN_utils / my_PINN_utils 直接加载 | 需按第 3 节映射调用 |

## 6. 已知边界

- τ̂ 超过 1.1/h（自然放电终点 10% 以外）为外推区
- bc 超出 [0.02, 15]×bc_ref 为外推区
- D_50 → R_s 的辨识建议用表面积加权有效半径 R_eff = 3Σr³n/Σr²n，而非简单 D_50/2
- SPM 层级限制不变：无电解液动力学、无粒径分布效应、恒温 298.15 K（原"仅放电"
  经 3.2 节镜像法已扩展到 CC 充电）
- 充电只覆盖 **CC 恒流段**（至 V_cutoff），不含 CV 恒压尾段；SPM 无析锂模型，
  x_n,surf→1 区物理上对应析锂窗口——以 V_cutoff 截断为准，快充倍率实际受析锂约束
- 充电曲线 t=0 电压含网络 IC 误差经陡 OCV 放大的初值偏移（可达数百 mV，见 3.2），
  首点之后精度与放电侧相当

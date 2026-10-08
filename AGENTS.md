# AGENTS.md — PINN-SPM-fast-prototyping

## 项目是什么
锂离子电池单粒子模型（SPM）的物理信息神经网络（PINN）快速原型库，NMC811 / Gr-Si 化学，仅放电。三套互不相同的模型资产：

1. **原版 CIDETEC 预训练模型**（仅推理、无官方训练代码）：根目录 `SPM NE/`、`SPM PE/`（TF SavedModel）；演示 `PINN notebook.ipynb`；推理工具 `PINN_utils.py`。
2. **重训版（4 输入）**：`train/` 是原模型约定的完整训练复现 + 独立数值参考解 + 验证；产物 `train/artifacts/models/SPM NE|PE`；演示 `my_PINN notebook.ipynb` + `my_PINN_utils.py`；结果记录 `my_PINN_results.md`。
3. **万能族版（3 输入，τ 参数化）**：`train_universal/` 是独立第二管线，把时间换成 τ̂ = τ/τ_ref，推理时解锁 R_s、D_s 自由参数；接口 `from train_universal.predict_uni import plot_Battery_uni`；演示 `PINN_uni notebook.ipynb`。

## 环境（Windows 原生 GPU 路径）
- conda 环境 `pinn210`：Python 3.10 + `tensorflow==2.10.1` + `numpy==1.23.5` + `protobuf==3.19.6` + scipy + matplotlib。无 requirements.txt，依赖以两个 README 为准。
- TF 2.10 是 Windows 原生 GPU 的最后版本。一律用 `conda run -n pinn210 python ...` 运行；直接调 `envs/pinn210/python.exe` 会因 CUDA DLL 不在 PATH 回退 CPU。
- GPU 自检：`conda run -n pinn210 python train/gpu_check.py`。网络极小（~180 参数、float64），CPU/GPU 训练结果等价。
- 非 git 仓库；机器休眠会杀死后台训练进程。

## 常用命令（均在仓库根目录执行）
```bash
# train/（重训管线）
conda run -n pinn210 python train/regression_baseline.py        # 原模型回归基线
conda run -n pinn210 python train/reference_spm.py selftest|gen # 数值参考解：自检 / 生成真值
conda run -n pinn210 python train/train_spm.py ne|pe 45000      # 训练（约 50 分钟/电极）
conda run -n pinn210 python train/validate.py                   # 三重验证
conda run -n pinn210 python train/voltage_ref_cmp.py            # 电压三方对比

# train_universal/（万能族管线）
conda run -n pinn210 python train_universal/reference_spm_uni.py selftest
conda run -n pinn210 python train_universal/train_spm_uni.py ne|pe 45000
conda run -n pinn210 python train_universal/validate_uni.py     # 五重验证
```
推理入口：`my_PINN_utils.plot_Battery(...)`（重训模型）、`train_universal.predict_uni.plot_Battery_uni(...)`（万能族，缺省参数自动补默认，`mode='charge'` 出充电曲线——镜像法共用放电网络，无需重训）。

## 架构边界
- `train/` 与 `train_universal/` 完全独立、互不 import；只共享 `data/` 与 conda 环境。不要在两者之间加依赖。
- `PINN_utils.py` 用相对 CWD 的路径加载模型 → 相关 Notebook/脚本必须在仓库根目录启动；`my_PINN_utils.py` 与 `train_universal/predict_uni.py` 按文件位置解析路径，任意 CWD 可用。
- 部署模型 = 根目录 `SPM NE|PE`；重训产物与其同格式（SavedModel），覆盖根目录即可被原 Notebook 使用。
- 两套输入约定不可混用：原/重训 4 输入 `[τ, ρ, bc_minmax, t_max_minmax]`；万能族 3 输入 `[s, ρ, bc_norm]`（bc 归一域 [0.02,15]×bc_ref）。
- 电池参数唯一来源：`data/params_GrSi_NMC811.json`；OCV 曲线 `data/pOCV_C20_*.txt`。

## 已知坑（复现记录，详见两份 README）
- Keras BN 不支持二阶梯度 → 两阶段训练：先标定 moving statistics，再冻结为推理模式（training=False）做物理训练。
- TF 2.10 eager tape 怪癖：二阶梯度 `t2.gradient` 必须在 t2 上下文内部调用（外部返回 None）；损失组装在 t2 内，权重梯度由外层 tape 承担。
- t_max 的 min-max 归一化端点方向在 NE/PE 间相反 → 必须用方向无关的 min/max；训练内部自洽 ≠ 部署约定正确，**验收必须走部署管线评估**。
- 类级 `@tf.function` 首次 trace 后绑定图，替换 `self.model` 不生效；评估已保存模型时直接调用。
- 原模型对越界 bc 硬拒绝（"Out of bounds"）且 3C 下物理窗口内即发散 NaN/inf；万能族对越界 bc 打印警告仍计算（外推精度无保证，`strict=True` 可拒绝）。
- 物理边界：SPM 层级（无电解液动力学、无粒径分布效应）、恒温 298.15 K、仅放电；τ̂ 超出自然放电终点 10% 为外推区。

## 改敏感区前先读
- `train/README_train.md`：架构/损失权重/采样策略/验证协议/训练坑全记录（中文）。
- `train_universal/README_universal.md`：万能族数学设计、推理映射公式、验证结果与已知边界（中文）。
- 一切精度对比的基准是独立数值参考解：`train/reference_spm.py`、`train_universal/reference_spm_uni.py`。

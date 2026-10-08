"""万能族 PINN 训练脚本（时间带分数 s 参数化，3 维输入）。在仓库根目录运行：

    conda run -n pinn210 python train_universal/train_spm_uni.py ne [steps]
    conda run -n pinn210 python train_spm_uni.py pe [steps]

与 train/train_spm.py（4 输入 t̂ 参数化）的区别：
  - 输入 (s, ρ, bc_norm)：s = τ̂/τ̂_band ∈ [0,1] 是"放电进度"（时间带分数），
    τ̂ = s·1.1/h，τ = τ̂·τ_ref，τ = t·D_s/R_s² —— (s, ρ, bc) 与 (τ, ρ, bc) 信息等价，
    解族与 R_s、D_s 无关：推理时 R_s、D_s、I、L、ε、C-rate 全部是解析映射的自由参数
  - 所有输入特征尺度 O(1)（旧 τ̂ 直接作输入会跨 3.4 个数量级，小网络无法分辨）
  - PDE 残差 = x_s − (1.1·τ_ref/h)·(x̂_ρρ + (2/ρ)x̂_ρ)（h = |bc|/|bc_ref| 按点缩放，
    与 ∂x/∂τ = ∇²ρx 数学等价且使解误差在全域均匀；BC 用原始绝对残差）
  - bc 在包络 [0.02, 15]×bc_ref 内 log-uniform 采样（覆盖 R_s×C_rate 乘积域）
  - ρ 采样下限 0.01（覆盖 R_s=2× 时的 ρ_min≈0.0122）

不确定项（自定超参，与上轮同骨架）：损失权重 1:4:1:1、45k 步、Adam 分段衰减、种子 0。
"""
import json
import os
import sys
import time

import numpy as np
import tensorflow as tf
from scipy.interpolate import CubicSpline

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from train_universal.reference_spm_uni import (ELEC, Electrode, load_params,
                                               BC_LO_MULT, BC_HI_MULT,
                                               BAND_MARGIN, R_MIN_ABS)

tf.keras.backend.set_floatx('float64')

# ---------------- 超参 ----------------
STEPS = 45000
LR_SCHEDULE = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
    [22500, 45000], [1e-3, 3e-4, 1e-4])
N_PDE, N_BC, N_SYM, N_IC = 2048, 256, 256, 1024
W_PDE, W_IC, W_BC, W_SYM = 1.0, 4.0, 1.0, 1.0
SEED = 0
LOG_EVERY, PROBE_EVERY = 500, 5000


def build_model():
    """Input(3)→BN→Dense(30,ELU)×2→Dense(1)。

    相对原架构（Dense 10）的偏离：bc 训练域宽 ~25×（[0.02,15]× vs [0.1,3]×），
    大 bc 处边界层陡峭，10 隐层单元装不下（实测 raw BC 残差卡在 0.25 量级），
    加宽到 30 后参数仍仅 ~1100。
    """
    inp = tf.keras.Input(shape=(3,), dtype=tf.float64)
    z = tf.keras.layers.BatchNormalization()(inp)
    z = tf.keras.layers.Dense(30, activation='elu')(z)
    z = tf.keras.layers.Dense(30, activation='elu')(z)
    out = tf.keras.layers.Dense(1, activation='linear')(z)
    return tf.keras.Model(inp, out)


class UniversalTrainer:
    def __init__(self, side, seed=SEED):
        self.side = side
        self.E = ELEC[side]                       # 参考参数电极（τ_ref/bc_ref/包络）
        self.P = load_params()
        tf.random.set_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.model = build_model()
        self.bc_ref_abs = abs(self.E.bc_ref())
        self.sign = self.E.sign
        self.tau_ref = self.E.tau_ref()
        self.rho_min = R_MIN_ABS / self.E.R_s      # 0.0244（NE）/0.0375（PE），与旧配方一致
        self.opt = tf.keras.optimizers.Adam(learning_rate=LR_SCHEDULE)
        # 在线校验用的参考解缓存
        self._ref_cache = {}

    # ---- 采样（log-uniform bc + 时间带内 τ̂） ----
    def sample_bc(self, n):
        u = np.exp(self.rng.uniform(np.log(BC_LO_MULT), np.log(BC_HI_MULT), n))
        return self.sign * u * self.bc_ref_abs

    def sample_s(self, n):
        """时间带分数 s ∈ [0,1]（τ̂ = s·1.1/h）。"""
        return self.rng.uniform(0.0, 1.0, n)

    def sample_rho(self, n):
        return self.rng.uniform(self.rho_min, 1.0, n)

    def pack(self, tau_hat, rho, bc):
        bc_n = np.abs(bc) / self.bc_ref_abs
        bc_n = (bc_n - BC_LO_MULT) / (BC_HI_MULT - BC_LO_MULT)
        return tf.stack([tau_hat, rho, bc_n], axis=1)

    # ---- BN 两阶段：先标定 moving statistics，再冻结为推理模式训练 ----
    def calibrate_bn(self, n_batches=500):
        for _ in range(n_batches):
            bc = self.sample_bc(N_PDE)
            X = self.pack(self.sample_s(N_PDE), self.sample_rho(N_PDE), bc)
            self.model(X, training=True)

    # ---- 损失 ----
    def train_step(self):
        bc_c = self.sample_bc(N_PDE)
        s_c = self.sample_s(N_PDE)
        rho_c = self.sample_rho(N_PDE)

        bc_b = self.sample_bc(N_BC)
        s_b = self.sample_s(N_BC)

        bc_s = self.sample_bc(N_SYM)
        s_s = self.sample_s(N_SYM)

        bc_i = self.sample_bc(N_IC)

        taus, rhos, bcs = [], [], []
        def add(tau, rho, bc):
            taus.append(tf.reshape(tf.constant(tau, tf.float64), (-1,)))
            rhos.append(tf.reshape(tf.constant(rho, tf.float64), (-1,)))
            bcs.append(tf.reshape(tf.constant(bc, tf.float64), (-1,)))
        add(s_c, rho_c, bc_c)
        add(s_b, np.ones(N_BC), bc_b)
        add(s_s, np.full(N_SYM, self.rho_min), bc_s)
        add(np.zeros(N_IC), self.sample_rho(N_IC), bc_i)

        tau_all = tf.concat(taus, 0); rho_all = tf.concat(rhos, 0)
        bc_all = tf.concat(bcs, 0)
        n_bc, n_sym = N_BC, N_SYM
        X = self.pack(tau_all, rho_all, bc_all)

        # 三层 tape（TF 2.10 eager：二阶梯度必须在 t2 上下文内调用）
        with tf.GradientTape(persistent=True) as t3:
            with tf.GradientTape(persistent=True) as t2:
                t2.watch(X)
                with tf.GradientTape(persistent=True) as t1:
                    t1.watch(X)
                    xbar = tf.reshape(self.model(X, training=False), (-1,))
                g = t1.gradient(xbar, X)
                dx_ds, dx_drho = g[:, 0], g[:, 1]
                d2x = t2.gradient(dx_drho, X)[:, 1]

                # 残差 = x_s − (1.1·τ_ref/h)·∇²ρx（h 按点缩放，与 ∂x/∂τ = ∇²ρx 等价）
                h_pt = tf.abs(bc_all) / self.bc_ref_abs
                lap = d2x + 2.0 * dx_drho / rho_all
                # 原始 τ-units 残差（与旧配方完全同形）：lap − x_τ，其中
                # x_τ = x_s·h/(1.1·τ_ref)。不加按点权重（res_s 形式的 (1.1·τ_ref/h)
                # 系数会在窄域内造成梯度失衡，实测 BC 卡在 5e-3 无法收敛）。
                res_pde = lap[:N_PDE] - dx_ds[:N_PDE] * h_pt[:N_PDE] / (BAND_MARGIN * self.tau_ref)

                loss_pde = tf.reduce_mean(tf.square(res_pde))
                # BC 用原始绝对残差（不按 h 归一）：通量误差对解的影响是绝对量级的，
                # 按 h 归一会允许大 bc 处出现与 bc 同量级的绝对通量误差（实测导致
                # 中段 h≈1 的解被污染到 RMS 0.47）
                loss_bc = tf.reduce_mean(tf.square(
                    dx_drho[N_PDE:N_PDE + n_bc] - bc_all[N_PDE:N_PDE + n_bc]))
                loss_sym = tf.reduce_mean(tf.square(dx_drho[N_PDE + n_bc:N_PDE + n_bc + n_sym]))
                loss_ic = tf.reduce_mean(tf.square(xbar[N_PDE + n_bc + n_sym:]))
                loss = W_PDE * loss_pde + W_IC * loss_ic + W_BC * loss_bc + W_SYM * loss_sym

            vars_ = self.model.trainable_variables
            grads = t3.gradient(loss, vars_)
        del t1, t2, t3
        self.opt.apply_gradients(zip(grads, vars_))
        return loss, loss_pde, loss_bc, loss_sym, loss_ic

    # ---- 推理 ----
    @tf.function(input_signature=[tf.TensorSpec([None, 3], tf.float64)])
    def _infer(self, X):
        return tf.reshape(self.model(X, training=False), (-1,))

    def xbar_on_grid(self, bc, tau_hat_grid, rho_grid):
        """给定 bc 的 (τ̂, ρ) 网格上的 x̄（推理模式）。"""
        T, R = np.meshgrid(tau_hat_grid, rho_grid, indexing='ij')
        X = self.pack(T.ravel(), R.ravel(), np.full(T.size, bc))
        return self._infer(tf.constant(X, tf.float64)).numpy().reshape(len(tau_hat_grid), len(rho_grid))

    # ---- 在线参考解校验（参考 R_s 下，τ̂ 网格 vs 参考解） ----
    def eval_against_reference(self, bc_mults=(0.5, 1.0, 2.0), n_cells=400):
        E = self.E
        errs = []
        for mult in bc_mults:
            bc = E.sign * mult * self.bc_ref_abs
            s_grid = np.linspace(0.0, 1.0, 100)
            rho = np.linspace(R_MIN_ABS / E.R_s, 1.0, 32)
            r_abs = rho * E.R_s
            # τ̂ = s·1.1/h；参考 R_s 下 t = τ̂·τ_ref·R_ref²/D_ref = s·1.1/h·3900
            t_eval = s_grid * BAND_MARGIN / mult * 3900.0
            r_c, x_shell, x_s = E.solve(bc, np.array(t_eval), n_cells=n_cells)
            knots_r = np.concatenate((r_c, [E.R_s]))
            x_ref = np.empty((100, 32))
            for k in range(100):
                knots_x = np.concatenate((x_shell[k], [x_s[k]]))
                x_ref[k] = CubicSpline(knots_r, knots_x)(r_abs)
            x_pred = self.xbar_on_grid(bc, s_grid, rho) + E.x0()
            errs.append((x_pred - x_ref).ravel())
        errs = np.concatenate(errs)
        return {'rms': float(np.sqrt(np.mean(errs ** 2))),
                'max': float(np.max(np.abs(errs)))}

    def physics_selfcheck(self):
        """推理模式下 IC / 表面BC / PDE 残差 / 对称性（随机探针）。"""
        rng = np.random.default_rng(123)
        out = {}
        for mult in (BC_LO_MULT ** 0.5, 1.0, BC_HI_MULT ** 0.5):   # 0.14×, 1×, 3.9×
            bc = self.sign * mult * self.bc_ref_abs
            s = self.sample_s(64)
            rho = self.sample_rho(64)
            Xt = tf.constant(self.pack(s, rho, np.full(64, bc)), tf.float64)
            with tf.GradientTape(persistent=True) as t2:
                t2.watch(Xt)
                with tf.GradientTape(persistent=True) as t1:
                    t1.watch(Xt)
                    xb = tf.reshape(self.model(Xt, training=False), (-1,))
                g = t1.gradient(xb, Xt)
                dx_ds, dx_drho = g[:, 0], g[:, 1]
                d2x = t2.gradient(dx_drho, Xt)[:, 1]
            del t1, t2
            h = mult
            res = dx_ds - (BAND_MARGIN * self.tau_ref / h) * (d2x + 2.0 * dx_drho / rho)
            X0 = tf.constant(self.pack(np.zeros(64), rho, np.full(64, bc)), tf.float64)
            ic = float(np.max(np.abs(self._infer(X0).numpy())))
            X1 = tf.constant(self.pack(np.ones(64), np.ones(64), np.full(64, bc)), tf.float64)
            with tf.GradientTape() as t1:
                t1.watch(X1)
                y1 = tf.reshape(self.model(X1, training=False), (-1,))
            bcg = t1.gradient(y1, X1)[:, 1].numpy()
            out[f'h={mult:.3f}'] = {
                'pde_rms': float(np.sqrt(np.mean(res ** 2))),
                'ic_max': ic,
                'bc_grad_err': float(np.max(np.abs(bcg - bc)) / abs(bc)),
            }
        return out


def main():
    side = sys.argv[1] if len(sys.argv) > 1 else 'ne'
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else STEPS
    assert side in ('ne', 'pe')

    tr = UniversalTrainer(side)
    print(f'=== training universal {side.upper()} PINN: steps={steps}, '
          f'inputs=(s, rho, bc_norm), N={N_PDE}+{N_BC}+{N_SYM}+{N_IC}/step, seed={SEED} ===')
    print(f'bc envelope = {BC_LO_MULT}..{BC_HI_MULT} x |bc_ref|={tr.bc_ref_abs:.5f}, '
          f'tau_ref={tr.tau_ref:.4f}, tau_hat_max={ELEC[side].tau_hat_max_global():.1f}')
    tr.calibrate_bn()
    print('BN moving statistics calibrated (500 batches).')

    t0 = time.time()
    hist = []
    for step in range(1, steps + 1):
        loss, lp, lb, ls, li = tr.train_step()
        if step % LOG_EVERY == 0 or step == 1:
            hist.append({'step': step, 'loss': float(loss), 'pde': float(lp),
                         'bc': float(lb), 'sym': float(ls), 'ic': float(li)})
            print(f'step {step:6d}  loss={loss:.3e} (pde {lp:.2e} bc {lb:.2e} '
                  f'sym {ls:.2e} ic {li:.2e})  [{time.time()-t0:.0f}s]', flush=True)
        if step % PROBE_EVERY == 0 or step == steps:
            m = tr.eval_against_reference()
            print(f'  [ref-check] x RMS={m["rms"]:.4f} max={m["max"]:.4f}', flush=True)

    print('\n--- physics self-check (inference mode) ---', flush=True)
    chk = tr.physics_selfcheck()
    for k, v in chk.items():
        print(f'{k}: ' + ', '.join(f'{kk}={vv:.3e}' for kk, vv in v.items()), flush=True)
    m = tr.eval_against_reference()
    print(f'final ref-check: {m}', flush=True)

    out_dir = os.path.join(HERE, 'artifacts', 'models')
    os.makedirs(out_dir, exist_ok=True)
    tag = 'UNI SPM NE' if side == 'ne' else 'UNI SPM PE'
    save_path = os.path.join(out_dir, tag)
    tf.keras.models.save_model(tr.model, save_path, save_format='tf')
    print('model saved ->', save_path, flush=True)

    report = {'side': side, 'steps': steps, 'seed': SEED, 'inputs': ['s', 'rho', 'bc_norm'],
              'weights': {'pde': W_PDE, 'ic': W_IC, 'bc': W_BC, 'sym': W_SYM},
              'collocation': {'n_pde': N_PDE, 'n_bc': N_BC, 'n_sym': N_SYM, 'n_ic': N_IC},
              'bc_envelope_mult': [BC_LO_MULT, BC_HI_MULT],
              'bc_sampling': 'log-uniform', 'band_margin': BAND_MARGIN,
              'hist_tail': hist[-10:], 'ref_check': m, 'physics_check': chk,
              'train_seconds': time.time() - t0}
    with open(os.path.join(out_dir, f'train_report_{side}.json'), 'w') as f:
        json.dump(report, f, indent=1)
    print('report saved.', flush=True)


if __name__ == '__main__':
    main()

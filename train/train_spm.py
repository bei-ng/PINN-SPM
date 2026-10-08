"""PINN-SPM 训练脚本（复现）。在仓库根目录运行：

    conda run -n pinn210 python train/train_spm.py ne [steps]
    conda run -n pinn210 python train/train_spm.py pe [steps]

复现约定（全部经 probe_original.py 实证锁定）：
  - 网络：Input(4) -> BN -> Dense(10,ELU) -> Dense(10,ELU) -> Dense(1)，float64（与原模型同构）
  - 输入 [τ, ρ, bc_minmax, t_max_minmax]；输出 x̄；x = x̄ + x0
  - 无量纲 PDE：∂x/∂τ = κ·(1/ρ²)∂/∂ρ(ρ²∂x/∂ρ)，κ = t_max·D_s/R_s²
  - 表面 BC：∂x/∂ρ|₁ = bc；对称：∂x/∂ρ|₀ = 0；IC：x̄(τ=0)=0
  - bc 在训练区间均匀采样；t_max 由 bc 通过 h=|bc/bc_ref| 决定，加 ±5% 抖动
    （探针显示原模型对 t_max 有独立响应，故保留轻微独立性）
  - 损失 = w_pde·(∇²ρx − ∂τx/κ)² + w_ic·x̄(0)² + w_bc·(∂ρx|₁−bc)² + w_sym·(∂ρx|ρmin)²
    （PDE 残差按 κ 归一，抵消 bc 跨两个量级导致的权重失衡）

不确定项（原作者未公开，此处为自定超参）：损失权重、配点规模、优化细节、随机种子。
"""
import json
import os
import sys
import time

import numpy as np
import tensorflow as tf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from train.reference_spm import (Electrode, load_params, BC_RANGE,
                                 R_MIN_ABS, T_MAX_BASE)

tf.keras.backend.set_floatx('float64')

# ---------------- 超参（不确定项，自定默认值） ----------------
STEPS = 45000
LR_SCHEDULE = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
    [22500, 45000], [1e-3, 3e-4, 1e-4])
BATCH_BC, BATCH_T, BATCH_R = 8, 16, 16      # 配点 2048/步
N_BC, N_SYM, N_IC = 256, 256, 1024          # 边界/对称/初值点每步数量
W_PDE, W_IC, W_BC, W_SYM = 1.0, 4.0, 1.0, 1.0
T_MAX_JITTER = 0.05
SEED = 0
LOG_EVERY, PROBE_EVERY = 500, 5000


def build_model():
    inp = tf.keras.Input(shape=(4,), dtype=tf.float64)
    z = tf.keras.layers.BatchNormalization()(inp)
    z = tf.keras.layers.Dense(10, activation='elu')(z)
    z = tf.keras.layers.Dense(10, activation='elu')(z)
    out = tf.keras.layers.Dense(1, activation='linear')(z)
    return tf.keras.Model(inp, out)


class SideTrainer:
    def __init__(self, side, seed=SEED):
        self.side = side
        self.e = Electrode(load_params(), side)
        self.rng = np.random.default_rng(seed)
        self.model = build_model()
        lo, hi = BC_RANGE[side]
        self.bc_lo, self.bc_hi = lo, hi
        self.bc_min, self.bc_max = min(lo, hi), max(lo, hi)   # 归一化用（方向无关）
        # t_max 归一化区间（方向无关）：|bc| 越大 t_max 越小
        self.tm_min = min(self.e.t_max(lo), self.e.t_max(hi))
        self.tm_max = max(self.e.t_max(lo), self.e.t_max(hi))
        self.rho_min = R_MIN_ABS / self.e.R_s
        self.opt = tf.keras.optimizers.Adam(learning_rate=LR_SCHEDULE)

    # ---- 输入归一化（与 normalize_Inputs_for_OCV 同式：min-max） ----
    def norm_cols(self, bc, t_max):
        bc_n = (bc - self.bc_min) / (self.bc_max - self.bc_min)
        tm_n = (t_max - self.tm_min) / (self.tm_max - self.tm_min)
        return bc_n, tm_n

    def pack(self, tau, rho, bc, t_max):
        bc_n, tm_n = self.norm_cols(bc, t_max)
        return tf.stack([tau, rho, bc_n, tm_n], axis=1)

    def sample_tmax(self, bc):
        h = np.abs(bc / self.e.bc_ref()) * (1.0 + self.rng.uniform(
            -T_MAX_JITTER, T_MAX_JITTER, size=bc.shape))
        return T_MAX_BASE / h

    # ---- BN 标定：原架构含 BatchNorm，其 fused 算子无二阶梯度，
    # 故先在采样分布上标定 moving statistics，再冻结为推理模式做物理训练 ----
    def calibrate_bn(self, n_batches=500, batch=2048):
        rng = self.rng
        for _ in range(n_batches):
            tau = rng.uniform(0.0, 1.0, batch)
            rho = rng.uniform(self.rho_min, 1.0, batch)
            bc = rng.uniform(self.bc_lo, self.bc_hi, batch)
            tm = self.sample_tmax(bc)
            X = self.pack(tau, rho, bc, tm)
            self.model(X, training=True)

    # ---- 损失 ----
    def train_step(self):
        rng = self.rng
        e = self.e
        n_pde = BATCH_BC * BATCH_T * BATCH_R
        # 配点：bc/τ/ρ 三元组独立均匀采样
        bc_c = rng.uniform(self.bc_lo, self.bc_hi, n_pde)
        tm_c = self.sample_tmax(bc_c)
        tau_c = rng.uniform(1e-3, 1.0, n_pde)
        rho_c = rng.uniform(self.rho_min, 1.0, n_pde)

        bc_b = rng.uniform(self.bc_lo, self.bc_hi, N_BC)
        tm_b = self.sample_tmax(bc_b)
        tau_b = rng.uniform(0.0, 1.0, N_BC)

        bc_s = rng.uniform(self.bc_lo, self.bc_hi, N_SYM)
        tm_s = self.sample_tmax(bc_s)
        tau_s = rng.uniform(0.0, 1.0, N_SYM)

        bc_i = rng.uniform(self.bc_lo, self.bc_hi, N_IC)
        tm_i = self.sample_tmax(bc_i)

        # 打包成一个大批次，行序 [pde, bc, sym, ic]
        taus, rhos, bcs, tms = [], [], [], []
        def add(tau, rho, bc, tm):
            taus.append(tf.reshape(tf.constant(tau, tf.float64), (-1,)))
            rhos.append(tf.reshape(tf.constant(rho, tf.float64), (-1,)))
            bcs.append(tf.reshape(tf.constant(bc, tf.float64), (-1,)))
            tms.append(tf.reshape(tf.constant(tm, tf.float64), (-1,)))
        add(tau_c, rho_c, bc_c, tm_c)
        add(tau_b, np.ones(N_BC), bc_b, tm_b)
        add(tau_s, np.full(N_SYM, self.rho_min), bc_s, tm_s)
        add(np.zeros(N_IC), rng.uniform(self.rho_min, 1.0, N_IC), bc_i, tm_i)

        tau_all = tf.concat(taus, 0); rho_all = tf.concat(rhos, 0)
        bc_all = tf.concat(bcs, 0); tm_all = tf.concat(tms, 0)
        n_bc = N_BC; n_sym = N_SYM
        X = self.pack(tau_all, rho_all, bc_all, tm_all)
        k_all = tm_all * e.D_s / e.R_s**2

        # 三层 tape。注意 TF 2.10 eager 的怪癖：二阶梯度 t2.gradient 必须在
        # t2 上下文内部调用（外部调用会返回 None），损失也在 t2 内组装，
        # 权重梯度由外层 t3 承担。
        with tf.GradientTape(persistent=True) as t3:
            with tf.GradientTape(persistent=True) as t2:
                t2.watch(X)
                with tf.GradientTape(persistent=True) as t1:
                    t1.watch(X)
                    xbar = tf.reshape(self.model(X, training=False), (-1,))
                g = t1.gradient(xbar, X)
                dx_dtau, dx_drho = g[:, 0], g[:, 1]
                d2x = t2.gradient(dx_drho, X)[:, 1]

                lap = d2x + 2.0 * dx_drho / rho_all
                res_pde = lap - dx_dtau / k_all

                loss_pde = tf.reduce_mean(tf.square(res_pde[:n_pde]))
                loss_bc = tf.reduce_mean(tf.square(
                    dx_drho[n_pde:n_pde + n_bc] - bc_all[n_pde:n_pde + n_bc]))
                loss_sym = tf.reduce_mean(tf.square(dx_drho[n_pde + n_bc:n_pde + n_bc + n_sym]))
                loss_ic = tf.reduce_mean(tf.square(xbar[n_pde + n_bc + n_sym:]))
                loss = W_PDE * loss_pde + W_IC * loss_ic + W_BC * loss_bc + W_SYM * loss_sym

            vars_ = self.model.trainable_variables
            grads = t3.gradient(loss, vars_)
        del t1, t2, t3
        self.opt.apply_gradients(zip(grads, vars_))
        return loss, loss_pde, loss_bc, loss_sym, loss_ic

    # ---- 推理模式评估（deployment 行为） ----
    @tf.function(input_signature=[tf.TensorSpec([None, 4], tf.float64)])
    def _infer(self, X):
        return tf.reshape(self.model(X, training=False), (-1,))

    def eval_against_reference(self, ref):
        """与参考解表面浓度序列对比（重训质量的核心指标）。"""
        e = self.e
        bcs = ref[f'{self.side}_bc']
        t_hat = ref[f'{self.side}_t_hat']
        r_hat = ref[f'{self.side}_r_hat']
        x_ref = ref[f'{self.side}_x']          # (n_bc, n_t, n_r)
        x0 = float(ref[f'{self.side}_x0'])
        errs, surf_errs = [], []
        for bi, bc in enumerate(bcs):
            t_max = e.t_max(bc)
            T, R = np.meshgrid(t_hat, r_hat, indexing='ij')
            tau_f, rho_f = T.ravel(), R.ravel()
            X = self.pack(tau_f, rho_f, np.full_like(tau_f, bc), np.full_like(tau_f, t_max))
            xb = self._infer(tf.constant(X, tf.float64)).numpy().reshape(len(t_hat), len(r_hat))
            x_pred = xb + x0
            errs.append(x_pred - x_ref[bi])
            surf_errs.append(x_pred[:, -1] - x_ref[bi][:, -1])
        errs = np.concatenate([e_.ravel() for e_ in errs])
        surf_errs = np.concatenate(surf_errs)
        return {'rms': float(np.sqrt(np.mean(errs**2))),
                'max': float(np.max(np.abs(errs))),
                'surf_rms': float(np.sqrt(np.mean(surf_errs**2))),
                'surf_max': float(np.max(np.abs(surf_errs)))}

    def physics_selfcheck(self):
        """推理模式下的 IC / 表面BC / PDE 残差（固定探针网格）。"""
        e = self.e
        rng = np.random.default_rng(123)
        bc_fix = np.array([self.bc_lo, e.bc_ref(), self.bc_hi])
        out = {}
        for bc in bc_fix:
            t_max = e.t_max(bc)
            tau = rng.uniform(0.0, 1.0, 64)
            rho = rng.uniform(self.rho_min, 1.0, 64)
            X = self.pack(tau, rho, np.full(64, bc), np.full(64, t_max))
            Xt = tf.constant(X, tf.float64)
            with tf.GradientTape(persistent=True) as t2:
                t2.watch(Xt)
                with tf.GradientTape(persistent=True) as t1:
                    t1.watch(Xt)
                    xb = tf.reshape(self.model(Xt, training=False), (-1,))
                g = t1.gradient(xb, Xt)
                dx_dtau, dx_drho = g[:, 0], g[:, 1]
                d2x = t2.gradient(dx_drho, Xt)[:, 1]
            del t1, t2
            lap = d2x + 2.0 * dx_drho / rho
            k = t_max * e.D_s / e.R_s**2
            res = lap - dx_dtau / k
            X0 = self.pack(np.zeros(64), rho, np.full(64, bc), np.full(64, t_max))
            ic = np.max(np.abs(self._infer(tf.constant(X0, tf.float64)).numpy()))
            X1 = self.pack(np.ones(64), np.ones(64), np.full(64, bc), np.full(64, t_max))
            X1t = tf.constant(X1, tf.float64)
            with tf.GradientTape() as t1:
                t1.watch(X1t)
                y1 = tf.reshape(self.model(X1t, training=False), (-1,))
            bcg = t1.gradient(y1, X1t)[:, 1].numpy()
            out[f'bc={bc:+.4f}'] = {
                'pde_rel': float(np.sqrt(np.mean(res**2)) / max(np.sqrt(np.mean((dx_dtau / k)**2)), 1e-30)),
                'ic_max': float(ic),
                'bc_grad_err': float(np.max(np.abs(bcg - bc)) / abs(bc)),
            }
        return out


def main():
    side = sys.argv[1] if len(sys.argv) > 1 else 'ne'
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else STEPS
    assert side in ('ne', 'pe')

    ref_path = os.path.join(ROOT, 'train', 'artifacts', 'reference_dataset.npz')
    ref = np.load(ref_path)
    tr = SideTrainer(side)

    print(f'=== training {side.upper()} PINN: steps={steps}, '
          f'collocation/step={BATCH_BC*BATCH_T*BATCH_R}, seed={SEED} ===')
    tr.calibrate_bn()
    print('BN moving statistics calibrated (500 batches, training-mode).')
    t0 = time.time()
    hist = []
    for step in range(1, steps + 1):
        loss, lp, lb, ls, li = tr.train_step()
        if step % LOG_EVERY == 0 or step == 1:
            hist.append({'step': step, 'loss': float(loss), 'pde': float(lp),
                         'bc': float(lb), 'sym': float(ls), 'ic': float(li)})
            print(f'step {step:6d}  loss={loss:.3e} (pde {lp:.2e} bc {lb:.2e} '
                  f'sym {ls:.2e} ic {li:.2e})  [{time.time()-t0:.0f}s]')
        if step % PROBE_EVERY == 0 or step == steps:
            m = tr.eval_against_reference(ref)
            print(f'  [ref-check] x RMS={m["rms"]:.4f} max={m["max"]:.4f} | '
                  f'surface RMS={m["surf_rms"]:.4f} max={m["surf_max"]:.4f}')

    print('\n--- physics self-check (inference mode) ---')
    chk = tr.physics_selfcheck()
    for k, v in chk.items():
        print(f'{k}: ' + ', '.join(f'{kk}={vv:.3e}' for kk, vv in v.items()))
    m = tr.eval_against_reference(ref)
    print(f'final ref-check: {m}')

    out_dir = os.path.join(ROOT, 'train', 'artifacts', 'models')
    os.makedirs(out_dir, exist_ok=True)
    tag = 'SPM NE' if side == 'ne' else 'SPM PE'
    save_path = os.path.join(out_dir, tag)
    tf.keras.models.save_model(tr.model, save_path, save_format='tf')
    print('model saved ->', save_path)

    report = {'side': side, 'steps': steps, 'seed': SEED,
              'weights': {'pde': W_PDE, 'ic': W_IC, 'bc': W_BC, 'sym': W_SYM},
              'collocation': {'bc': BATCH_BC, 't': BATCH_T, 'r': BATCH_R,
                              'n_bc': N_BC, 'n_sym': N_SYM, 'n_ic': N_IC},
              't_max_jitter': T_MAX_JITTER, 'hist_tail': hist[-10:],
              'ref_check': m, 'physics_check': chk,
              'train_seconds': time.time() - t0}
    with open(os.path.join(out_dir, f'train_report_{side}.json'), 'w') as f:
        json.dump(report, f, indent=1)
    print('report saved.')


if __name__ == '__main__':
    main()

"""Step 2: 对已发布的原 PINN 做探针实验，锁定训练约定（在仓库根目录运行）：

    conda run -n pinn210 python train/probe_original.py

探针内容：
  A. PDE 残差：检验 x̄ 是否满足无量纲球扩散方程  ∂x/∂τ = κ·(1/ρ²)∂/∂ρ(ρ²∂x/∂ρ)，
     κ = t_max·D_s/R_s²（τ=t/t_max, ρ=r/R_s）。
  B. 初始条件：x̄(τ=0) ≈ 0 ?
  C. 表面通量 BC：∂x/∂ρ|_{ρ=1} ≈ bc ?（若成立则整个输入/输出约定被证实）
  D. 中心对称：r→0 端梯度 ≈ 0 ?
  E. t_max 冗余性：固定 (τ, ρ, bc) 改变 t_max 输入，看输出是否变化。
  F. bc 外推行为：越界 5% / 20% 时的输出是否失效。
"""
import os
import sys

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PINN_utils import (U_model_n, U_model_p, Inputs_n, Inputs_p,
                        normalize_Inputs_for_OCV)
from train.reference_spm import Electrode, load_params, R_MIN_ABS, T_MAX_BASE

P = load_params()
ELEC = {'ne': Electrode(P, 'ne'), 'pe': Electrode(P, 'pe')}
MODEL = {'ne': U_model_n, 'pe': U_model_p}
LIM = {'ne': Inputs_n, 'pe': Inputs_p}


def make_input(side, tau, rho, bc, t_max):
    """构造与 PINN_utils 推理完全一致的归一化输入 (n,4)。"""
    tau = np.atleast_1d(np.asarray(tau, dtype=np.float64))
    rho = np.atleast_1d(np.asarray(rho, dtype=np.float64))
    bc = np.broadcast_to(np.asarray(bc, dtype=np.float64), tau.shape)
    t_max = np.broadcast_to(np.asarray(t_max, dtype=np.float64), tau.shape)
    raw = np.stack([tau, rho, bc, t_max], axis=1)
    lim = LIM[side]
    bc_n = (raw[:, 2] - lim[:, 2].min()) / (lim[:, 2].max() - lim[:, 2].min())
    tm_n = (raw[:, 3] - lim[:, 3].min()) / (lim[:, 3].max() - lim[:, 3].min())
    out = raw.copy()
    out[:, 2] = bc_n
    out[:, 3] = tm_n
    return tf.constant(out, dtype=tf.float64)


def model_xyz(side, X):
    return tf.reshape(tf.cast(MODEL[side](X), tf.float64), (-1,))


def probe_pde_ic_bc(side, bc_list):
    e = ELEC[side]
    print(f'\n===== probe {side.upper()} =====')
    tau_1d = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
    rho_1d = np.linspace(0.05, 1.0, 40)
    T, R = np.meshgrid(tau_1d, rho_1d, indexing='ij')
    tau_f, rho_f = T.ravel(), R.ravel()

    for bc in bc_list:
        t_max = e.t_max(bc)
        X = make_input(side, tau_f, rho_f, bc, t_max)
        with tf.GradientTape(persistent=True) as t2:
            t2.watch(X)
            with tf.GradientTape(persistent=True) as t1:
                t1.watch(X)
                y = model_xyz(side, X)
            g = t1.gradient(y, X)                      # (n,4): dy/dτ, dy/dρ, ...
            dy_dtau = g[:, 0]
            dy_drho = g[:, 1]
            d2y_drho2 = t2.gradient(dy_drho, X)[:, 1]
        del t1, t2

        y = y.numpy(); dy_dtau = dy_dtau.numpy(); dy_drho = dy_drho.numpy()
        d2y_drho2 = d2y_drho2.numpy()
        kappa = t_max * e.D_s / e.R_s**2
        rho = rho_f
        lap = d2y_drho2 + 2.0 * dy_drho / rho          # (1/ρ²)∂/∂ρ(ρ²∂x/∂ρ)
        resid = dy_dtau - kappa * lap
        rel_pde = np.linalg.norm(resid) / max(np.linalg.norm(dy_dtau), 1e-30)

        # IC: τ=0
        X0 = make_input(side, np.zeros(40), rho_1d, bc, t_max)
        y0 = model_xyz(side, X0).numpy()
        # 表面 BC: ∂x/∂ρ|₁ vs bc
        surf_grad = dy_drho[rho_f == 1.0]
        bc_err = np.max(np.abs(surf_grad - bc)) / abs(bc)
        # 中心对称：在模型定义域最内点 ρ_min 处的梯度
        Xin = make_input(side, tau_f, np.full_like(tau_f, R_MIN_ABS / e.R_s), bc, t_max)
        with tf.GradientTape() as t1:
            t1.watch(Xin)
            yi = model_xyz(side, Xin)
        grad_in = t1.gradient(yi, Xin)[:, 1].numpy()
        sym = np.max(np.abs(grad_in)) / max(abs(bc), 1e-30)

        print(f'bc={bc:+.4f} κ={kappa:9.3f}: '
              f'PDE rel.resid={rel_pde:.3e} | IC max|x̄(0)|={np.max(np.abs(y0)):.3e} | '
              f'BC err |∂ρx|₁-bc|/|bc|={bc_err:.3e} | sym grad@ρmin={sym:.3e}')


def probe_tmax_redundancy(side):
    e = ELEC[side]
    lo, hi = (-0.346851, -0.011509128654707082) if side == 'ne' else (0.0164088, 0.546388)
    print(f'\n===== t_max redundancy ({side.upper()}) =====')
    for bc in (lo, e.bc_ref(), hi):
        t_max_slave = e.t_max(bc)
        tau = np.random.default_rng(0).uniform(0.01, 0.99, 50)
        rho = np.random.default_rng(1).uniform(0.03, 1.0, 50)
        base = model_xyz(side, make_input(side, tau, rho, bc, t_max_slave)).numpy()
        deltas = []
        for fac in (0.8, 1.2, 2.0):
            Xv = make_input(side, tau, rho, bc, t_max_slave * fac)
            v = model_xyz(side, Xv).numpy()
            deltas.append(np.max(np.abs(v - base)))
        print(f'bc={bc:+.4f}: max Δx̄ when t_max ×0.8/1.2/2.0 = '
              + ' / '.join(f'{d:.3e}' for d in deltas))


def probe_bc_extrapolation(side):
    e = ELEC[side]
    lo, hi = (-0.346851, -0.011509128654707082) if side == 'ne' else (0.0164088, 0.546388)
    print(f'\n===== bc extrapolation ({side.upper()}) =====')
    tau = np.linspace(0.0, 1.0, 100)
    rho = np.full_like(tau, 1.0)  # 表面浓度序列
    for bc in (lo, hi, lo * 1.05, hi * 1.05, lo * 1.2, hi * 1.2):
        if side == 'pe' and bc < 0:
            continue
        t_max = e.t_max(bc)
        y = model_xyz(side, make_input(side, tau, rho, bc, t_max)).numpy()
        print(f'bc={bc:+.4f}: x_surf range [{(y+e.x0()).min():.4f}, {(y+e.x0()).max():.4f}], '
              f'finite={np.all(np.isfinite(y))}, monotone_dep={np.all(np.diff(y) != 0)}')


if __name__ == '__main__':
    np.random.seed(0)
    for side in ('ne', 'pe'):
        e = ELEC[side]
        lo, hi = (-0.346851, -0.011509128654707082) if side == 'ne' else (0.0164088, 0.546388)
        bc_list = [e.bc_ref(), lo, hi]
        probe_pde_ic_bc(side, bc_list)
    probe_tmax_redundancy('ne')
    probe_tmax_redundancy('pe')
    probe_bc_extrapolation('ne')
    probe_bc_extrapolation('pe')

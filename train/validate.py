"""Step 4: 三重验证（在仓库根目录运行）：

    conda run -n pinn210 python train/validate.py

① 物理自检（IC / 表面BC / PDE 残差，推理模式）—— 重训 vs 原模型
② 与独立数值参考解对比（最终正确性标准）—— 重训 vs 原模型
③ 端到端电压回归：用重训模型替换 PINN_utils 的模型后，复算 5 个 Notebook
   示例的电压曲线，与原模型逐点对比并绘图。

产物：train/artifacts/validation/（图 + validation_report.json）
"""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import PINN_utils as pu
from train.reference_spm import Electrode, load_params, BC_RANGE
from train.train_spm import SideTrainer

tf.keras.backend.set_floatx('float64')

OUT = os.path.join('train', 'artifacts', 'validation')
os.makedirs(OUT, exist_ok=True)

ELEC = {'ne': Electrode(load_params(), 'ne'), 'pe': Electrode(load_params(), 'pe')}
LIM = {'ne': pu.Inputs_n, 'pe': pu.Inputs_p}


def load_models(side):
    tag = 'SPM NE' if side == 'ne' else 'SPM PE'
    orig = tf.keras.models.load_model(tag)
    re = tf.keras.models.load_model(os.path.join('train', 'artifacts', 'models', tag))
    return orig, re


def probe_grid(side, bc):
    """(t̂, r̂) 全评估网格输入（与 PINN_utils 完全同式归一化）。"""
    e = ELEC[side]
    t_hat = np.linspace(0.0, 1.0, 100)
    r_hat = np.linspace(0.2e-6 / e.R_s, 1.0, 32)
    T, R = np.meshgrid(t_hat, r_hat, indexing='ij')
    tau_f, rho_f = T.ravel(), R.ravel()
    t_max = e.t_max(bc)
    raw = np.stack([tau_f, rho_f, np.full_like(tau_f, bc), np.full_like(tau_f, t_max)], 1)
    lim = LIM[side]
    raw[:, 2] = (raw[:, 2] - lim[:, 2].min()) / (lim[:, 2].max() - lim[:, 2].min())
    raw[:, 3] = (raw[:, 3] - lim[:, 3].min()) / (lim[:, 3].max() - lim[:, 3].min())
    return raw, t_hat, r_hat, t_max


def eval_model(model, side, X):
    return np.asarray(tf.reshape(model(tf.constant(X, tf.float64)), (-1,)))


def vs_reference(model, side, ref):
    """模型 vs 数值参考解：全 (bc × t × r) 网格上的 x 误差。"""
    e = ELEC[side]
    x0 = float(ref[f'{side}_x0'])
    errs, surf = [], []
    for bi, bc in enumerate(ref[f'{side}_bc']):
        X, t_hat, r_hat, _ = probe_grid(side, bc)
        x = eval_model(model, side, X).reshape(100, 32) + x0
        errs.append(x - ref[f'{side}_x'][bi])
        surf.append(x[:, -1] - ref[f'{side}_x'][bi][:, -1])
    errs = np.concatenate([e_.ravel() for e_ in errs])
    surf = np.concatenate(surf)
    return {'rms': float(np.sqrt(np.mean(errs**2))), 'max': float(np.max(np.abs(errs))),
            'surf_rms': float(np.sqrt(np.mean(surf**2))), 'surf_max': float(np.max(np.abs(surf)))}


def physics_check(model, side):
    """推理模式下 IC / 表面BC / 归一化PDE残差（固定随机探针）。"""
    e = ELEC[side]
    rng = np.random.default_rng(123)
    tr_helper = SideTrainer(side)          # 仅借用 pack/norm 逻辑
    tr_helper.model = model
    chk = {}
    for bc in (BC_RANGE[side][0], e.bc_ref(), BC_RANGE[side][1]):
        t_max = e.t_max(bc)
        tau = rng.uniform(0, 1, 64)
        rho = rng.uniform(0.2e-6 / e.R_s, 1.0, 64)
        Xt = tf.constant(tr_helper.pack(tau, rho, np.full(64, bc), np.full(64, t_max)), tf.float64)
        with tf.GradientTape(persistent=True) as t2:
            t2.watch(Xt)
            with tf.GradientTape(persistent=True) as t1:
                t1.watch(Xt)
                xb = tf.reshape(model(Xt, training=False), (-1,))
            g = t1.gradient(xb, Xt)
            dx_dtau, dx_drho = g[:, 0], g[:, 1]
            d2x = t2.gradient(dx_drho, Xt)[:, 1]
        del t1, t2
        k = t_max * e.D_s / e.R_s**2
        res = (d2x + 2 * dx_drho / rho) - dx_dtau / k
        X0 = tf.constant(tr_helper.pack(np.zeros(64), rho, np.full(64, bc), np.full(64, t_max)), tf.float64)
        ic = float(np.max(np.abs(eval_model(model, side, X0.numpy()))))
        X1 = tf.constant(tr_helper.pack(np.ones(64), np.ones(64), np.full(64, bc), np.full(64, t_max)), tf.float64)
        with tf.GradientTape() as t1:
            t1.watch(X1)
            y1 = tf.reshape(model(X1, training=False), (-1,))
        bcg = t1.gradient(y1, X1)[:, 1].numpy()
        chk[f'bc={bc:+.4f}'] = {
            'pde_rel': float(np.sqrt(np.mean(res**2)) / max(np.sqrt(np.mean((dx_dtau / k)**2)), 1e-30)),
            'ic_max': ic,
            'bc_grad_err': float(np.max(np.abs(bcg - bc)) / abs(bc)),
        }
    return chk


def voltage_case(side_models, case):
    """按 plot_Battery 同一管线计算一组几何/倍率的电压曲线（传入模型替换）。"""
    tn, tp, pn, pp, cr = case
    pu.U_model_n, pu.U_model_p = side_models['ne'], side_models['pe']
    bc_n_o, bc_p_o, I = pu.obtain_bcs(tn, tp, pn, pp, cr)
    in_n = pu.create_input_from_bc('ne', bc_n_o)
    in_p = pu.create_input_from_bc('pe', bc_p_o)
    t_max_n, t_max_p = in_n[0, 3], in_p[0, 3]
    rn = pu.obtain_x_n('input', 0, pu.normalize_Inputs_for_OCV(in_n, pu.Inputs_n), t_max_n)
    rp = pu.obtain_x_p('input', 0, pu.normalize_Inputs_for_OCV(in_p, pu.Inputs_p), t_max_p)
    plt.close('all')
    xns, xps = np.asarray(rn[1]), np.asarray(rp[1])
    t_arr = np.linspace(0, min(t_max_n, t_max_p), 100)
    xns_i = np.interp(t_arr, np.linspace(0, t_max_n, 100), xns)
    xps_i = np.interp(t_arr, np.linspace(0, t_max_p, 100), xps)
    eps_s_n = 1 - pu.fracInMat_n - pn
    eps_s_p = 1 - pu.fracInMat_p - pp
    eta_n = 2 * pu.R * pu.T / pu.F * np.arcsinh(
        pu.R_s_n * I / (2 * pu.k_0_n * np.sqrt(pu.c_e_0) * 3 * eps_s_n * pu.L_n
                        * pu.c_s_max_n * pu.F * pu.A_n * np.sqrt(np.maximum(xns_i * (1 - xns_i), 0))))
    eta_p = 2 * pu.R * pu.T / pu.F * np.arcsinh(
        -pu.R_s_p * I / (2 * pu.k_0_p * np.sqrt(pu.c_e_0) * 3 * eps_s_p * pu.L_p
                         * pu.c_s_max_p * pu.F * pu.A_p * np.sqrt(np.maximum(xps_i * (1 - xps_i), 0))))
    V = np.asarray(pu.inter_cathode(xps_i) - pu.inter_anode(xns_i) + eta_p - eta_n)
    return t_arr, V, xns_i, xps_i


def main():
    ref = np.load(os.path.join('train', 'artifacts', 'reference_dataset.npz'))
    report = {'reference': {}, 'physics': {}, 'agreement': {}, 'voltage': {}}
    cases = [
        ('default_1C', pu.L_n, pu.L_p, pu.epsilon_n, pu.epsilon_p, 1),
        ('thick_n_2e4_1C', 2e-4, 5e-5, 0.2, 0.2, 1),
        ('thick_n_2e4_por60_2C', 2e-4, 5e-5, 0.6, 0.6, 2),
        ('thin_5e5_1C', 5e-5, 5e-5, 0.2, 0.2, 1),
        ('default_3C', pu.L_n, pu.L_p, pu.epsilon_n, pu.epsilon_p, 3),
    ]

    models = {}
    for side in ('ne', 'pe'):
        orig, re = load_models(side)
        models[side] = {'orig': orig, 're': re}
        print(f'\n########## {side.upper()} ##########')
        for name, m in (('original', orig), ('retrained', re)):
            r = vs_reference(m, side, ref)
            c = physics_check(m, side)
            report['reference'][f'{side}_{name}'] = r
            report['physics'][f'{side}_{name}'] = c
            print(f'[{name}] vs 参考解: x RMS={r["rms"]:.4f} max={r["max"]:.4f} | '
                  f'surface RMS={r["surf_rms"]:.4f} max={r["surf_max"]:.4f}')
            for k, v in c.items():
                print(f'   {k}: pde_rel={v["pde_rel"]:.3e} ic_max={v["ic_max"]:.3e} '
                      f'bc_err={v["bc_grad_err"]:.3e}')

        # 重训 vs 原模型一致性
        diffs = []
        for bc in ref[f'{side}_bc']:
            X, _, _, _ = probe_grid(side, bc)
            diffs.append(eval_model(re, side, X) - eval_model(orig, side, X))
        diffs = np.concatenate([d.ravel() for d in diffs])
        report['agreement'][side] = {'rms': float(np.sqrt(np.mean(diffs**2))),
                                     'max': float(np.max(np.abs(diffs)))}
        print(f'[agreement] x̄(retrained)-x̄(original): RMS={report["agreement"][side]["rms"]:.4f} '
              f'max={report["agreement"][side]["max"]:.4f}')

    # ---- 端到端电压回归 ----
    print('\n########## voltage end-to-end ##########')
    orig_pair = {'ne': models['ne']['orig'], 'pe': models['pe']['orig']}
    re_pair = {'ne': models['ne']['re'], 'pe': models['pe']['re']}
    for name, *case in cases:
        t_o, V_o, xn_o, xp_o = voltage_case(orig_pair, case)
        t_r, V_r, xn_r, xp_r = voltage_case(re_pair, case)
        valid = np.isfinite(V_o) & np.isfinite(V_r)
        dv = np.max(np.abs(V_r[valid] - V_o[valid]))
        dxn = np.max(np.abs(xn_r[valid] - xn_o[valid]))
        dxp = np.max(np.abs(xp_r[valid] - xp_o[valid]))
        report['voltage'][name] = {
            'max_dV': float(dv), 'max_dx_surf_ne': float(dxn), 'max_dx_surf_pe': float(dxp),
            'V_end_original': float(V_o[np.isfinite(V_o)][-1]) if np.any(np.isfinite(V_o)) else None,
            'V_end_retrained': float(V_r[np.isfinite(V_r)][-1]) if np.any(np.isfinite(V_r)) else None,
            'n_nan_original': int(np.sum(~np.isfinite(V_o))), 'n_nan_retrained': int(np.sum(~np.isfinite(V_r))),
        }
        print(f'{name}: max|ΔV|={dv*1000:.1f} mV | Δx_surf NE/PE={dxn:.4f}/{dxp:.4f} | '
              f'NaN pts orig/re = {report["voltage"][name]["n_nan_original"]}/{report["voltage"][name]["n_nan_retrained"]}')
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(t_o, V_o, 'r-', label='original PINN', lw=1.5)
        ax.plot(t_r, V_r, 'b--', label='retrained PINN', lw=1.5)
        ax.set_xlabel('t [s]'); ax.set_ylabel('V [V]')
        ax.set_title(name)
        ax.legend()
        fig.savefig(os.path.join(OUT, f'voltage_{name}.png'), dpi=120, bbox_inches='tight')
        plt.close(fig)

    with open(os.path.join(OUT, 'validation_report.json'), 'w') as f:
        json.dump(report, f, indent=1)
    print('\nreport saved ->', OUT)


if __name__ == '__main__':
    main()

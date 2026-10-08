"""电压曲线三方对比：original PINN vs retrained PINN vs 数值参考解。

在仓库根目录运行：  conda run -n pinn210 python train/voltage_ref_cmp.py
判据：在物理窗口内（参考解表面嵌锂度保持 0.02 ≤ x ≤ 0.99）比较 RMS/max |ΔV|。
"""
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
from train.reference_spm import Electrode, load_params
from train.validate import voltage_case, load_models

tf.keras.backend.set_floatx('float64')
P = load_params()
E_NE = Electrode(P, 'ne')
E_PE = Electrode(P, 'pe')

cases = [
    ('default_1C', pu.L_n, pu.L_p, pu.epsilon_n, pu.epsilon_p, 1),
    ('thick_n_2e4_1C', 2e-4, 5e-5, 0.2, 0.2, 1),
    ('thick_n_2e4_por60_2C', 2e-4, 5e-5, 0.6, 0.6, 2),
    ('thin_5e5_1C', 5e-5, 5e-5, 0.2, 0.2, 1),
    ('default_3C', pu.L_n, pu.L_p, pu.epsilon_n, pu.epsilon_p, 3),
]


def reference_voltage(case):
    """用参考解的表面浓度 + 同一组 pOCV/Butler-Volmer 公式组装电压曲线。"""
    tn, tp, pn, pp, cr = case
    bc_n_o, bc_p_o, I = pu.obtain_bcs(tn, tp, pn, pp, cr)
    t_total = min(E_NE.t_max(bc_n_o), E_PE.t_max(bc_p_o))
    t_arr = np.linspace(0, t_total, 100)
    _, _, xns = E_NE.solve(bc_n_o, t_arr, n_cells=400)
    _, _, xps = E_PE.solve(bc_p_o, t_arr, n_cells=400)
    eps_s_n = 1 - E_NE.frac_inert - pn
    eps_s_p = 1 - E_PE.frac_inert - pp
    eta_n = 2 * pu.R * pu.T / pu.F * np.arcsinh(
        pu.R_s_n * I / (2 * pu.k_0_n * np.sqrt(pu.c_e_0) * 3 * eps_s_n * pu.L_n
                        * pu.c_s_max_n * pu.F * pu.A_n * np.sqrt(np.maximum(xns * (1 - xns), 0))))
    eta_p = 2 * pu.R * pu.T / pu.F * np.arcsinh(
        -pu.R_s_p * I / (2 * pu.k_0_p * np.sqrt(pu.c_e_0) * 3 * eps_s_p * pu.L_p
                         * pu.c_s_max_p * pu.F * pu.A_p * np.sqrt(np.maximum(xps * (1 - xps), 0))))
    V = np.asarray(pu.inter_cathode(xps) - pu.inter_anode(xns) + eta_p - eta_n)
    return t_arr, V, xns, xps


def main():
    orig_pair, re_pair = load_models_pair()
    print(f'{"case":24s} {"orig RMS":>9s} {"orig max":>9s} {"retr RMS":>9s} {"retr max":>9s}  (mV, 物理窗口内)')
    for name, *case in cases:
        t_r, V_ref, xns_ref, xps_ref = reference_voltage(case)
        # 物理窗口：两极表面嵌锂度都未越界
        phys = (xns_ref >= 0.02) & (xps_ref <= 0.99) & np.isfinite(V_ref)
        _, V_o, _, _ = voltage_case(orig_pair, case)
        _, V_re, _, _ = voltage_case(re_pair, case)
        d_o = V_o[phys] - V_ref[phys]
        d_r = V_re[phys] - V_ref[phys]
        print(f'{name:24s} {np.sqrt(np.mean(d_o**2))*1000:9.1f} {np.max(np.abs(d_o))*1000:9.1f} '
              f'{np.sqrt(np.mean(d_r**2))*1000:9.1f} {np.max(np.abs(d_r))*1000:9.1f}   n_phys={phys.sum()}')
        fig, ax = plt.subplots(figsize=(6.5, 4))
        ax.plot(t_r, V_ref, 'k-', lw=2.0, label='reference solver')
        ax.plot(t_r, V_o, 'r--', lw=1.3, label='original PINN')
        ax.plot(t_r, V_re, 'b-.', lw=1.3, label='retrained PINN')
        ax.axvspan(t_r[phys][0] if phys.any() else 0, t_r[phys][-1] if phys.any() else 0,
                   color='gray', alpha=0.08)
        ax.set_xlabel('t [s]'); ax.set_ylabel('V [V]'); ax.set_title(name); ax.legend()
        fig.savefig(os.path.join('train', 'artifacts', 'validation', f'v3_voltage_{name}.png'),
                    dpi=120, bbox_inches='tight')
        plt.close(fig)


def load_models_pair():
    orig = {}; re = {}
    for side in ('ne', 'pe'):
        o, r = load_models(side)
        orig[side] = o
        re[side] = r
    return orig, re


if __name__ == '__main__':
    main()

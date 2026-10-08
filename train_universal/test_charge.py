"""充电模式验证：PINN 镜像法充电曲线 vs 独立数值参考解。

镜像法（predict_uni.mode='charge'）不重训：x_ch = stoi0 + x0 − x̄(s,ρ;|bc|)，
共用放电网络。参考解独立积分同一充电问题：solve(bc 充电符号, x_init=stoi0)，
电压组装与 predict_uni 同式（电流带负号）。

对比量：表面化学计量比 x_n/x_p、组装电压 V（至 PINN 截止电压截断点）。

用法：conda run -n pinn210 python train_universal/test_charge.py
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from train_universal.predict_uni import plot_Battery_uni, _OCV, _R, _T, _F
from train_universal.reference_spm_uni import Electrode, load_params

_P = load_params()

CASES = [
    ('default 1C', {}),
    ('Rs x1.5, 2C', dict(C_rate=2, R_s_n=1.5 * 8.21e-6, R_s_p=1.5 * 5.34e-6)),
    ('Rs x0.7, 0.5C', dict(C_rate=0.5, R_s_n=0.7 * 8.21e-6, R_s_p=0.7 * 5.34e-6)),
    ('Rs x2, 3C (envelope corner)', dict(C_rate=3, R_s_n=2 * 8.21e-6, R_s_p=2 * 5.34e-6)),
]


def eta_of(E, xs, I_signed, porosity, sign):
    eps_s = E.eps_s(porosity)
    with np.errstate(divide='ignore', invalid='ignore'):
        return 2 * _R * _T / _F * np.arcsinh(
            sign * E.R_s * I_signed / (2 * E.k0 * np.sqrt(E.c_e0) * 3 * eps_s * E.L0 * E.c_max *
                                       _F * E.A * np.sqrt(np.maximum(xs * (1 - xs), 0))))


def reference_charge(design, I_signed, t_full):
    """独立数值参考解的充电：bc 充电符号（bc_value 直接吃带符号电流）、初值 stoi0。"""
    tn, tp = design['thickness_n'], design['thickness_p']
    pn, pp = design['porosity_n'], design['porosity_p']
    EN = Electrode(_P, 'ne', R_s=design['R_s_n'], D_s=design['D_s_n'])
    EP = Electrode(_P, 'pe', R_s=design['R_s_p'], D_s=design['D_s_p'])
    out = {}
    for side, E, L, por in (('ne', EN, tn, pn), ('pe', EP, tp, pp)):
        bc = E.bc_value(I_signed, L, por)          # 充电符号（与 predict_uni 的取反结果一致）
        _, _, x_surf = E.solve(bc, t_full, x_init=E.stoi0)
        out[side] = x_surf
    V = (_OCV['pe'](out['pe']) - _OCV['ne'](out['ne'])
         + eta_of(EP, out['pe'], I_signed, pp, -1.0)
         - eta_of(EN, out['ne'], I_signed, pn, +1.0))
    return out['ne'], out['pe'], np.asarray(V)  # 全带长（调用方自行截断对齐）


def main():
    out_dir = os.path.join(HERE, 'artifacts')
    os.makedirs(out_dir, exist_ok=True)
    print('case                          | x_n RMS/max      | x_p RMS/max      '
          '| V RMS/max (mV)   | t_cutoff PINN/ref')
    print('-' * 118)
    for i, (name, kwargs) in enumerate(CASES, 1):
        r = plot_Battery_uni(mode='charge', plot=False, **kwargs)
        t_pinn, V_pinn = r['t'], r['V']
        n = len(t_pinn)                          # PINN 曲线已截到 V_cutoff 越过点
        t_full = np.linspace(0.0, min(r['t_end_n'], r['t_end_p']), 100)

        # 参考解在同一设计点（R_s/D_s 覆盖）上积分充电问题
        xn_full, xp_full, V_ref_full = reference_charge(r['design'], r['I'], t_full)
        xn_ref, xp_ref, V_ref = xn_full[:n], xp_full[:n], V_ref_full[:n]

        # 参考解自身到达截止电压的位置（供对比终止时刻）
        hit = np.where(np.isfinite(V_ref_full) & (V_ref_full >= r['V_cutoff']))[0]
        t_cut_ref = float(t_full[hit[0]]) if hit.size else None

        m = np.isfinite(V_pinn) & np.isfinite(V_ref)
        e_n = r['x_n_surface'][m] - xn_ref[m]
        e_p = r['x_p_surface'][m] - xp_ref[m]
        eV = (V_pinn[m] - V_ref[m]) * 1e3
        t_cut_pinn = r['t_cutoff']

        print('%-29s | %8.4f/%6.4f | %8.4f/%6.4f | %9.1f/%7.1f | %s / %s' % (
            name,
            np.sqrt(np.mean(e_n ** 2)), np.max(np.abs(e_n)),
            np.sqrt(np.mean(e_p ** 2)), np.max(np.abs(e_p)),
            np.sqrt(np.mean(eV ** 2)), np.max(np.abs(eV)),
            ('%.0f s' % t_cut_pinn) if r['cutoff_reached'] else 'n/r',
            ('%.0f s' % t_cut_ref) if t_cut_ref is not None else 'n/r'))

        # 逐工况对比图
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
        ax1.plot(t_pinn, V_pinn, 'b-', lw=1.5, label='PINN mirror (charge)')
        ax1.plot(t_full[:n], V_ref, 'r--', lw=1.2, label='numeric reference')
        ax1.axhline(r['V_cutoff'], color='gray', ls=':', lw=0.8)
        ax1.set_ylabel('V [V]')
        ax1.set_title('charge, %s' % name, fontsize=10)
        ax1.legend(fontsize=8)
        ax2.plot(t_pinn, r['x_n_surface'], 'b-', lw=1.5, label=r'$x_{n,surf}$ PINN')
        ax2.plot(t_full[:n], xn_ref, 'r--', lw=1.2, label=r'$x_{n,surf}$ ref')
        ax2.plot(t_pinn, r['x_p_surface'], 'b-', lw=1.5, label=r'$x_{p,surf}$ PINN')
        ax2.plot(t_full[:n], xp_ref, 'r--', lw=1.2, label=r'$x_{p,surf}$ ref')
        ax2.set_xlabel('t [s]')
        ax2.set_ylabel('surface stoichiometry')
        ax2.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, 'charge_cmp_%d.png' % i), dpi=110)
        plt.close(fig)

    print('\nfigures: train_universal/artifacts/charge_cmp_1..4.png')


if __name__ == '__main__':
    main()

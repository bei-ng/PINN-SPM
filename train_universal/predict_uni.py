"""万能 PINN 推理接口：输入设计参数 → 放电/充电曲线。未提供的参数用默认值（NMC811/G-Si 参考电芯）。

用法（在仓库根目录，或任意目录配合 sys.path）：

    from train_universal.predict_uni import plot_Battery_uni

    # 全默认：110/70 µm、孔隙率 0.56/0.31、1C、参考 R_s/D_s
    r = plot_Battery_uni()

    # 任意组合：未提供的自动用默认值
    r = plot_Battery_uni(thickness_n=200e-6, porosity_p=0.4, C_rate=2,
                         R_s_n=1.5e-6 * 2, D_s_p=5e-15)

可调参数（全部可选）：
    thickness_n/p  电极厚度 [m]，默认 110e-6 / 70e-6
    porosity_n/p   孔隙率 [-]，默认 0.56 / 0.31
    C_rate         倍率 [-]，默认 1
    R_s_n/p        颗粒半径 [m]，默认 8.21e-6 / 5.34e-6（可任意取，模型已解耦）
    D_s_n/p        固相扩散系数 [m²/s]，默认 4e-14 / 1e-14（可任意取）

返回 dict：t, V, x_n_surface, x_p_surface, I, bc_n, bc_p, t_end_n, t_end_p,
          envelope_ok, violations, design（生效参数回显）。
bc 超出训练包络 [0.02, 15]×bc_ref 时打印警告并在 violations 中标注（仍计算，
外推精度无保证）；strict=True 时改为返回 None。

充电：mode='charge'（默认 'discharge'）。利用 PDE/BC/初值的线性性，充电解 =
放电解的镜像：网络输出偏移 u 取反，x = stoi0 − u（放电为 x = x0 + u），共用
同一网络、无需重训；初值取 0% SOC 均匀态 stoi0，电流取负号进 Butler–Volmer，
曲线在 V ≥ V_cutoff（默认 4.2 V）处截断。返回 dict 额外含 mode / V_cutoff /
cutoff_reached / t_cutoff。
"""
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as si
import tensorflow as tf

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
os.chdir(ROOT)
tf.keras.backend.set_floatx('float64')

from train_universal.reference_spm_uni import ELEC, Electrode, load_params, \
    BC_LO_MULT, BC_HI_MULT

MODELS = {}
for _side, _tag in (('ne', 'UNI SPM NE'), ('pe', 'UNI SPM PE')):
    MODELS[_side] = tf.keras.models.load_model(os.path.join(HERE, 'artifacts', 'models', _tag))

with open(os.path.join(ROOT, 'data', 'params_GrSi_NMC811.json')) as f:
    _P = json.load(f)
_R = _P['constants']['R']['value']
_F = _P['constants']['F']['value']
_T = 298.15

_OCV = {}
for _side in ('ne', 'pe'):
    _key = 'negativeElectrode' if _side == 'ne' else 'positiveElectrode'
    _d = np.loadtxt(os.path.join(ROOT, 'data', _P[_key]['active_materials'][0]['openCircuitPotential']['value']))
    _OCV[_side] = si.interp1d(_d[:, 0], _d[:, 1], fill_value='extrapolate')


def _uni_xbar(side, bc, s, rho):
    """万能族网络前向：输入 (s, ρ, bc_norm)。"""
    E = ELEC[side]
    T, R = np.meshgrid(np.atleast_1d(s), np.atleast_1d(rho), indexing='ij')
    u = abs(bc) / abs(E.bc_ref())
    bc_n = (u - BC_LO_MULT) / (BC_HI_MULT - BC_LO_MULT)
    X = np.stack([T.ravel(), R.ravel(), np.full(T.size, bc_n)], 1)
    y = tf.reshape(MODELS[side](tf.constant(X, tf.float64), training=False), (-1,)).numpy()
    return y.reshape(T.shape)


def plot_Battery_uni(thickness_n=None, thickness_p=None, porosity_n=None, porosity_p=None,
                     C_rate=None, R_s_n=None, R_s_p=None, D_s_n=None, D_s_p=None,
                     experimental=None, strict=False, plot=True,
                     mode='discharge', V_cutoff=4.2):
    """输入设计参数（未提供者用默认值），返回并绘制放电/充电曲线。"""
    if mode not in ('discharge', 'charge'):
        raise ValueError(f"mode must be 'discharge' or 'charge', got {mode!r}")
    charge = (mode == 'charge')
    # ---- 生效设计（None → 默认） ----
    tn = thickness_n if thickness_n is not None else ELEC['ne'].L0
    tp = thickness_p if thickness_p is not None else ELEC['pe'].L0
    pn = porosity_n if porosity_n is not None else ELEC['ne'].eps0
    pp = porosity_p if porosity_p is not None else ELEC['pe'].eps0
    cr = C_rate if C_rate is not None else 1.0

    EL = Electrode(_P, 'ne', R_s=R_s_n, D_s=D_s_n)
    EP = Electrode(_P, 'pe', R_s=R_s_p, D_s=D_s_p)
    design = {'thickness_n': tn, 'thickness_p': tp, 'porosity_n': pn, 'porosity_p': pp,
              'C_rate': cr, 'R_s_n': EL.R_s, 'R_s_p': EP.R_s,
              'D_s_n': EL.D_s, 'D_s_p': EP.D_s}

    eps_s_n = EL.eps_s(pn)
    eps_s_p = EP.eps_s(pp)
    if eps_s_n <= 0 or eps_s_p <= 0:
        print(f'Invalid porosity: active volume fraction <= 0 (eps_s_n={eps_s_n:.3f}, eps_s_p={eps_s_p:.3f})')
        return None

    # ---- 容量、电流、bc（容量与 R_s/D_s 无关） ----
    Q_n = EL._capacity(tn, eps_s_n)
    Q_p = EP._capacity(tp, eps_s_p)
    I = min(Q_n, Q_p) / 3600.0 * cr
    limiting = 'ne' if Q_n <= Q_p else 'pe'
    bc = {'ne': EL.bc_value(I, tn, pn), 'pe': EP.bc_value(I, tp, pp)}
    if charge:
        # 充电电流反向 → 表面通量 bc 变号（网络只用 |bc|，符号仅进输出与 eta）
        bc = {k: -v for k, v in bc.items()}
        I_signed = -I
    else:
        I_signed = I

    # ---- 训练包络检查（外推警告）----
    # 注意：必须对照参考电极的 bc_ref（固定常数）。若用覆盖后电极自己的 bc_ref，
    # R_s²/D_s 会在比值中约掉，R_s/D_s 的影响将完全逃过检查。
    violations = []
    for side, E in (('ne', EL), ('pe', EP)):
        u = abs(bc[side]) / abs(ELEC[side].bc_ref())
        if not (BC_LO_MULT <= u <= BC_HI_MULT):
            violations.append(f'{side}: |bc|/|bc_ref| = {u:.4g} outside training envelope '
                              f'[{BC_LO_MULT}, {BC_HI_MULT}] (extrapolation, accuracy not guaranteed)')
    if violations:
        print('WARNING out of training envelope:\n  ' + '\n  '.join(violations))
        if strict:
            return None

    # ---- 每电极时间带与公共时间轴 ----
    surf = {}
    t_ends = {}
    for side, E in (('ne', EL), ('pe', EP)):
        h = abs(bc[side]) / abs(ELEC[side].bc_ref())
        s = np.linspace(0.0, 1.0, 100)
        # t = τ̂·τ_ref·R²/D = s·1.1/h·τ_ref·R²/D（充电摆幅与放电等幅，K 常数方向对称）
        t_side = s * 1.1 / h * ELEC[side].tau_ref() * E.R_s ** 2 / E.D_s
        xbar = _uni_xbar(side, bc[side], s, np.array([1.0]))[:, 0]
        # 网络输出 xbar = u（对初值 x0 的偏移，放电 x = x0 + u）。
        # 充电镜像：电流反号 → u → −u，x = stoi0 − u（stoi0 = 0% SOC 均匀初值）
        xs = (E.stoi0 - xbar) if charge else (xbar + E.x0())
        surf[side] = (t_side, xs)
        t_ends[side] = t_side[-1]

    t = np.linspace(0.0, min(t_ends.values()), 100)
    xs_n = np.interp(t, surf['ne'][0], surf['ne'][1])
    xs_p = np.interp(t, surf['pe'][0], surf['pe'][1])

    # ---- 电压：pOCV + Butler–Volmer（用设计的 R_s/k_0；充电电流带负号） ----
    def eta_of(E, xs, sign):
        eps_s = E.eps_s(pp if E.side == 'pe' else pn)
        with np.errstate(divide='ignore', invalid='ignore'):
            return 2 * _R * _T / _F * np.arcsinh(
                sign * E.R_s * I_signed / (2 * E.k0 * np.sqrt(E.c_e0) * 3 * eps_s * E.L0 * E.c_max *
                                           _F * E.A * np.sqrt(np.maximum(xs * (1 - xs), 0))))

    eta_n = eta_of(EL, xs_n, +1.0)
    eta_p = eta_of(EP, xs_p, -1.0)
    V = np.asarray(_OCV['pe'](xs_p) - _OCV['ne'](xs_n) + eta_p - eta_n)
    valid = (xs_n > 0) & (xs_n < 1) & (xs_p > 0) & (xs_p < 1)
    V = np.where(valid, V, np.nan)

    # ---- 充电终止：上截止电压（越过点截断；恒流框架不覆盖 CV 段） ----
    cutoff_reached, t_cutoff = False, None
    if charge:
        hit = np.where(np.isfinite(V) & (V >= V_cutoff))[0]
        if hit.size:
            k = hit[0]
            cutoff_reached, t_cutoff = True, float(t[k])
            t, V, xs_n, xs_p = t[:k + 1], V[:k + 1], xs_n[:k + 1], xs_p[:k + 1]

    # ---- 绘图 ----
    if plot:
        plt.figure()
        plt.plot(t, V, '-', lw=1.5, color='blue',
                 label='universal PINN SPM (%s)' % mode)
        if experimental is not None:
            plt.plot(experimental['# Time [s]'], experimental['Voltage [V]'], 'r',
                     label='experiment', alpha=1)
        plt.xlabel(r'$t$ [s]')
        plt.ylabel(r'$Voltage$ [V]')
        labels = [
            r'$L_{NE}=%.0f\,\mu$m' % (tn * 1e6), r'$L_{PE}=%.0f\,\mu$m' % (tp * 1e6),
            r'$\varepsilon_{NE}=%.2f$' % pn, r'$\varepsilon_{PE}=%.2f$' % pp,
            r'$C=%.1f$' % cr,
            r'$R_{s,NE}=%.1f\,\mu$m' % (EL.R_s * 1e6), r'$R_{s,PE}=%.1f\,\mu$m' % (EP.R_s * 1e6),
            r'$D_{s,NE}=%.1e$' % EL.D_s, r'$D_{s,PE}=%.1e$' % EP.D_s,
            ('limiting: ' + limiting),
        ]
        for lb in labels:
            plt.plot([], [], ' ', label=lb)
        plt.legend(bbox_to_anchor=(1, 0.5), fontsize=8)
        if violations:
            plt.title('WARNING: out of training envelope (see violations)', color='red', fontsize=10)
        if not np.any(np.isfinite(V)):
            plt.text(0.5, 0.5, 'No valid curve:\nsurface stoichiometry left (0,1)\n'
                     '— parameters far outside training envelope',
                     transform=plt.gca().transAxes, ha='center', color='red', fontsize=11)

    return {'t': t, 'V': V, 'x_n_surface': xs_n, 'x_p_surface': xs_p,
            'I': float(I_signed), 'bc_n': float(bc['ne']), 'bc_p': float(bc['pe']),
            't_end_n': float(t_ends['ne']), 't_end_p': float(t_ends['pe']),
            'limiting': limiting, 'envelope_ok': not violations,
            'violations': violations, 'design': design,
            'mode': mode, 'V_cutoff': V_cutoff,
            'cutoff_reached': cutoff_reached, 't_cutoff': t_cutoff}


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    out_dir = os.path.join(HERE, 'artifacts')
    os.makedirs(out_dir, exist_ok=True)

    # 1) 全默认 1C
    r1 = plot_Battery_uni()
    print('default 1C: V[0]=%.4f  V_end=%.4f  limiting=%s  envelope_ok=%s' % (
        r1['V'][0], r1['V'][~np.isnan(r1['V'])][-1], r1['limiting'], r1['envelope_ok']))
    plt.savefig(os.path.join(out_dir, 'predict_demo_default.png'), dpi=110, bbox_inches='tight')

    # 2) 大颗粒 + 高倍率（R_s/D_s 自由参数演示；u_n = 4·2·2 = 16→改 D_s 压回包络内）
    r2 = plot_Battery_uni(C_rate=2, R_s_n=2 * 8.21e-6, R_s_p=1.5 * 5.34e-6, D_s_n=3e-14)
    print('Rs x2/1.5, Ds x0.75, 2C: V[0]=%.4f  envelope_ok=%s' % (r2['V'][0], r2['envelope_ok']))
    plt.savefig(os.path.join(out_dir, 'predict_demo_custom.png'), dpi=110, bbox_inches='tight')

    # 3) 越包络外推警告演示（u = 0.01 < 包络下限 0.02）
    r3 = plot_Battery_uni(C_rate=0.01)
    print('0.01C: returned?', r3 is not None, '| violations:', r3['violations'] if r3 else None)

    # 4) 充电（镜像法，共用放电网络）：默认电芯 1C，截到上截止电压
    r4 = plot_Battery_uni(mode='charge')
    fin = np.isfinite(r4['V'])
    print('default 1C charge: V[0]=%.4f  V_end=%.4f  t_cutoff=%s  cutoff_reached=%s' % (
        r4['V'][0], r4['V'][fin][-1],
        '%.0f s' % r4['t_cutoff'] if r4['cutoff_reached'] else 'not reached',
        r4['cutoff_reached']))
    plt.savefig(os.path.join(out_dir, 'predict_demo_charge.png'), dpi=110, bbox_inches='tight')

    # 5) 充电 + 自由 R_s/D_s：0.5C（充电常用倍率段）
    r5 = plot_Battery_uni(mode='charge', C_rate=0.5, R_s_n=1.5 * 8.21e-6, D_s_p=0.7e-14)
    print('charge 0.5C Rs1.5: V[0]=%.4f  cutoff_reached=%s  envelope_ok=%s' % (
        r5['V'][0], r5['cutoff_reached'], r5['envelope_ok']))
    plt.close('all')

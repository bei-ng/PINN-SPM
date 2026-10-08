"""万能族 PINN 五重验证（在仓库根目录运行）：

    conda run -n pinn210 python train_universal/validate_uni.py

① 物理自检（推理模式：IC / 表面BC / PDE 残差 / 对称性）
② (R_s × C_rate) 网格 vs 参考解：R_s ∈ {0.5,0.7,1,1.5,2}× × C ∈ {0.2,1,3}，15 组/电极
③ 万能族一致性：不同 (R_s, C) 组合产生相同 bc → 参考解互相重合（数学验证），
   网络对每组都应匹配各自参考解（学习验证）
④ 与旧 4-input 模型（train/artifacts/models）在参考 R_s 处交叉对比
⑤ 电压组装抽查：默认几何 1C，R_s ∈ {0.7, 1.0, 1.5}×（两极同倍），vs 参考解电压

产物：train_universal/artifacts/validation/（JSON 报告 + 图）
"""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy.interpolate import CubicSpline

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from train_universal.reference_spm_uni import (ELEC, Electrode, load_params,
                                               BC_LO_MULT, BC_HI_MULT, R_MIN_ABS)

tf.keras.backend.set_floatx('float64')
P = load_params()
OUT = os.path.join(HERE, 'artifacts', 'validation')
os.makedirs(OUT, exist_ok=True)

MODELS = {}
for side in ('ne', 'pe'):
    tag = 'UNI SPM NE' if side == 'ne' else 'UNI SPM PE'
    MODELS[side] = tf.keras.models.load_model(os.path.join(HERE, 'artifacts', 'models', tag))

# 旧 4-input 模型（仅 ④ 用；其输入归一化常数按原仓库约定硬编码）
OLD_MODELS = {}
OLD_LIM = {
    'ne': {'bc': (-0.3468502645833334, -0.011509128654707082), 'tmax': (1300.0, 39178.06095372121)},
    'pe': {'bc': (0.016408826202450197, 0.5464139125415916), 'tmax': (1262.6309179993384, 42045.609569377964)},
}
for side, tag in (('ne', 'SPM NE'), ('pe', 'SPM PE')):
    p = os.path.join(ROOT, 'train', 'artifacts', 'models', tag)
    if os.path.isdir(p):
        OLD_MODELS[side] = tf.keras.models.load_model(p)


def uni_xbar(side, bc, s, rho):
    """万能族网络前向：输入 (s, ρ, bc_norm)，s 为时间带分数。"""
    E = ELEC[side]
    T, R = np.meshgrid(s, rho, indexing='ij')
    u = np.abs(bc) / abs(E.bc_ref())
    bc_n = (u - BC_LO_MULT) / (BC_HI_MULT - BC_LO_MULT)
    X = np.stack([T.ravel(), R.ravel(), np.full(T.size, bc_n)], 1)
    y = tf.reshape(MODELS[side](tf.constant(X, tf.float64), training=False), (-1,)).numpy()
    return y.reshape(T.shape)


def old_xbar(side, bc, t_hat, rho, t_max):
    if side not in OLD_MODELS:
        return None
    lim = OLD_LIM[side]
    T, R = np.meshgrid(t_hat, rho, indexing='ij')
    bc_n = (np.abs(bc) - lim['bc'][0]) / (lim['bc'][1] - lim['bc'][0])
    tm_n = (t_max - lim['tmax'][0]) / (lim['tmax'][1] - lim['tmax'][0])
    X = np.stack([T.ravel(), R.ravel(), np.full(T.size, bc_n), np.full(T.size, tm_n)], 1)
    y = tf.reshape(OLD_MODELS[side](tf.constant(X, tf.float64), training=False), (-1,)).numpy()
    return y.reshape(T.shape)


def ref_field(side, bc, R_s_mult, s_grid):
    """参考解：给定 bc 与 R_s 倍数，在时间带分数 s 网格上求 x（ρ 网格 32 点）。
    τ̂ = s·1.1/h；τ = τ̂·τ_ref；t = τ·R_s²/D_s（τ_ref 为固定归一化常数）。"""
    E0 = ELEC[side]
    E = Electrode(P, side, R_s=R_s_mult * E0.R_s) if R_s_mult != 1.0 else E0
    rho = np.linspace(R_MIN_ABS / E.R_s, 1.0, 32)
    r_abs = rho * E.R_s
    h = abs(bc) / abs(E0.bc_ref())
    tau_hat = s_grid * 1.1 / h
    tau = tau_hat * E0.tau_ref()
    t_eval = tau * E.R_s ** 2 / E.D_s
    r_c, x_shell, x_s = E.solve(bc, np.array(t_eval))
    out = np.empty((len(s_grid), 32))
    for k in range(len(s_grid)):
        out[k] = CubicSpline(np.concatenate((r_c, [E.R_s])),
                             np.concatenate((x_shell[k], [x_s[k]])))(r_abs)
    return out, rho


def design_bc(side, c_rate, R_s_mult=1.0):
    """默认几何下给定倍率与 R_s 倍数的 bc（I 与 R_s 无关）。"""
    E0 = ELEC[side]
    E = Electrode(P, side, R_s=R_s_mult * E0.R_s)
    I = min(ELEC['ne'].Q0, ELEC['pe'].Q0) / 3600.0 * c_rate
    return E.bc_value(I, E.L0, E.eps0)


report = {'physics': {}, 'grid_rs_c': {}, 'universal_family': {}, 'old_model_cross': {}, 'voltage': {}}

# ---------- ① 物理自检 ----------
print('===== ① physics self-check =====', flush=True)
for side in ('ne', 'pe'):
    E = ELEC[side]
    res_all = {}
    for mult in (0.14, 1.0, 3.9):
        bc_center = E.sign * mult * abs(E.bc_ref())
        rng = np.random.default_rng(7)
        u = mult * 10.0 ** rng.uniform(-0.5, 0.5, 64)      # 该倍率附近的对数均匀探针
        bcq = E.sign * u * abs(E.bc_ref())
        s = rng.uniform(0, 1, 64)
        rho = rng.uniform(R_MIN_ABS / E.R_s, 1.0, 64)
        T, R, B = s, rho, bcq
        Xt = tf.constant(np.stack([T, R, (np.abs(B) / abs(E.bc_ref()) - BC_LO_MULT) /
                                   (BC_HI_MULT - BC_LO_MULT)], 1), tf.float64)
        with tf.GradientTape(persistent=True) as t2:
            t2.watch(Xt)
            with tf.GradientTape(persistent=True) as t1:
                t1.watch(Xt)
                xb = tf.reshape(MODELS[side](Xt, training=False), (-1,))
            g = t1.gradient(xb, Xt)
            ds, drho = g[:, 0], g[:, 1]
            d2 = t2.gradient(drho, Xt)[:, 1]
        del t1, t2
        # 原始 τ-units 残差：x_τ = ds·u/(1.1·τ_ref)
        res = (d2 + 2 * drho / rho) - ds * u / (1.1 * E.tau_ref())
        X0 = tf.constant(np.stack([np.zeros(64), rho, (np.abs(B) / abs(E.bc_ref()) - BC_LO_MULT) /
                                   (BC_HI_MULT - BC_LO_MULT)], 1), tf.float64)
        ic = float(np.max(np.abs(tf.reshape(MODELS[side](X0, training=False), (-1,)).numpy())))
        X1 = tf.constant(np.stack([np.ones(64), np.ones(64), (np.abs(B) / abs(E.bc_ref()) - BC_LO_MULT) /
                                   (BC_HI_MULT - BC_LO_MULT)], 1), tf.float64)
        with tf.GradientTape() as t1:
            t1.watch(X1)
            y1 = tf.reshape(MODELS[side](X1, training=False), (-1,))
        bcg = t1.gradient(y1, X1)[:, 1].numpy()
        res_all[f'h~{mult:.2f}'] = {'pde_rms': float(np.sqrt(np.mean(res ** 2))),
                                    'ic_max': ic,
                                    'bc_err': float(np.max(np.abs(bcg - bcq) / np.abs(bcq)))}
    report['physics'][side] = res_all
    print(f'{side}: ' + json.dumps(res_all), flush=True)

# ---------- ② (R_s × C_rate) 网格 vs 参考解 ----------
print('\n===== ② R_s × C_rate grid vs reference =====', flush=True)
RS_MULTS = [0.5, 0.7, 1.0, 1.5, 2.0]
C_RATES = [0.2, 1.0, 3.0]
for side in ('ne', 'pe'):
    E = ELEC[side]
    rows = []
    for rm in RS_MULTS:
        for c in C_RATES:
            bc = design_bc(side, c, rm)
            s_grid = np.linspace(0.0, 1.0, 60)
            x_ref, rho = ref_field(side, bc, rm, s_grid)
            x_pred = uni_xbar(side, bc, s_grid, rho) + E.x0()
            err = (x_pred - x_ref).ravel()
            rows.append({'R_s_mult': rm, 'C_rate': c, 'bc': float(bc),
                         'rms': float(np.sqrt(np.mean(err ** 2))), 'max': float(np.max(np.abs(err)))})
            print(f'{side} R_s×{rm:<4} C={c:<4}: bc={bc:+.4f}  RMS={rows[-1]["rms"]:.4f}  max={rows[-1]["max"]:.4f}', flush=True)
    report['grid_rs_c'][side] = rows

# ---------- ③ 万能族一致性 ----------
print('\n===== ③ universal-family consistency =====', flush=True)
for side in ('ne', 'pe'):
    E = ELEC[side]
    target_mult = 1.5
    combos = [(1.0, 1.5), (np.sqrt(1.5), 1.0), (np.sqrt(3.0), 0.5)]   # bc 均为 1.5×ref（bc ∝ R_s²·C）
    s_grid = np.linspace(0.05, 0.95, 40)
    ref_sol, rho = [], None
    net_err = []
    for rm, c in combos:
        bc = design_bc(side, c, rm)
        xr, rho = ref_field(side, bc, rm, s_grid)
        ref_sol.append(xr)
        xn = uni_xbar(side, bc, s_grid, rho) + E.x0()
        net_err.append((xn - xr).ravel())
    inter_ref = max(np.max(np.abs(ref_sol[0] - ref_sol[1])), np.max(np.abs(ref_sol[1] - ref_sol[2])))
    net_max = max(float(np.max(np.abs(e))) for e in net_err)
    report['universal_family'][side] = {
        'combos': [{'R_s_mult': rm, 'C_rate': c} for rm, c in combos],
        'inter_reference_spread': float(inter_ref),
        'net_vs_ref_max': net_max}
    print(f'{side}: inter-reference spread={inter_ref:.2e} (数学) | net-vs-ref max={net_max:.4f} (学习)', flush=True)

# ---------- ④ 与旧 4-input 模型交叉（参考 R_s 处） ----------
print('\n===== ④ cross-check vs old 4-input models (R_s = reference) =====', flush=True)
for side in ('ne', 'pe'):
    if side not in OLD_MODELS:
        print(f'{side}: old model not found, skipped')
        continue
    E = ELEC[side]
    errs = []
    for mult in (0.5, 1.0, 2.0):
        bc = E.sign * mult * abs(E.bc_ref())
        h = mult
        t_max = 3900.0 / h
        t_hat = np.linspace(0.0, 1.0, 60)
        rho = np.linspace(R_MIN_ABS / E.R_s, 1.0, 32)
        # 参考 R_s 下 τ̂ = t/3900 = t̂·t_max/3900；s = τ̂·h/1.1
        s_grid = t_hat * (t_max / 3900.0) * h / 1.1
        x_old = old_xbar(side, bc, t_hat, rho, t_max)
        if x_old is None:
            continue
        x_new = uni_xbar(side, bc, s_grid, rho)
        errs.append((x_new - x_old).ravel())
    errs = np.concatenate(errs)
    report['old_model_cross'][side] = {'rms': float(np.sqrt(np.mean(errs ** 2))),
                                       'max': float(np.max(np.abs(errs)))}
    print(f'{side}: x̄(uni) − x̄(old): RMS={report["old_model_cross"][side]["rms"]:.4f} '
          f'max={report["old_model_cross"][side]["max"]:.4f}', flush=True)

# ---------- ⑤ 电压组装抽查 ----------
print('\n===== ⑤ voltage assembly spot-check =====', flush=True)
pu_consts = {'F': P['constants']['F']['value'], 'R': P['constants']['R']['value'], 'T': 298.15}


def voltage_curve(r_s_mult, c_rate=1.0):
    """默认几何、两极同 R_s 倍数：万能族组装的电压 vs 参考解电压。"""
    import scipy.interpolate as si
    def ocv(side):
        key = 'negativeElectrode' if side == 'ne' else 'positiveElectrode'
        f = P[key]['active_materials'][0]['openCircuitPotential']['value']
        d = np.loadtxt(os.path.join(ROOT, 'data', f))
        return si.interp1d(d[:, 0], d[:, 1], fill_value='extrapolate')
    ua, uc = ocv('ne'), ocv('pe')
    I = min(ELEC['ne'].Q0, ELEC['pe'].Q0) / 3600.0 * c_rate
    t_end = []
    surf = {}
    for side in ('ne', 'pe'):
        E0 = ELEC[side]
        E = Electrode(P, side, R_s=r_s_mult * E0.R_s)
        bc = E.bc_value(I, E.L0, E.eps0)
        s = np.linspace(0.0, 1.0, 100)
        # 物理时间：t = s·1.1·K/|bc|·R_s²/D_s（τ̂ = s·1.1/h，τ = τ̂·τ_ref）
        t_side = s * 1.1 * E0.K() / abs(bc) * E.R_s ** 2 / E.D_s
        xs = uni_xbar(side, bc, s, np.array([1.0]))[:, 0] + E.x0()
        surf[side] = (t_side, xs)
        t_end.append(t_side[-1])
    t_common = np.linspace(0.0, min(t_end), 100)
    # 公共物理时间轴上查询两极表面嵌锂度
    xs_n = np.interp(t_common, surf['ne'][0], surf['ne'][1])
    xs_p = np.interp(t_common, surf['pe'][0], surf['pe'][1])
    eta = {}
    for side, E0 in (('ne', ELEC['ne']), ('pe', ELEC['pe'])):
        E = Electrode(P, side, R_s=r_s_mult * E0.R_s)
        xs = xs_n if side == 'ne' else xs_p
        sgn = 1.0 if side == 'ne' else -1.0
        eps_s = E.eps_s(E.eps0)
        eta[side] = 2 * pu_consts['R'] * pu_consts['T'] / pu_consts['F'] * np.arcsinh(
            sgn * E.R_s * I / (2 * E.k0 * np.sqrt(E.c_e0) * 3 * eps_s * E.L0 * E.c_max *
                                pu_consts['F'] * E.A * np.sqrt(np.maximum(xs * (1 - xs), 0))))
    V = uc(xs_p) - ua(xs_n) + eta['pe'] - eta['ne']
    return t_common, np.asarray(V), xs_n, xs_p


def voltage_ref(r_s_mult, c_rate=1.0):
    I = min(ELEC['ne'].Q0, ELEC['pe'].Q0) / 3600.0 * c_rate
    curves = {}
    for side in ('ne', 'pe'):
        E0 = ELEC[side]
        E = Electrode(P, side, R_s=r_s_mult * E0.R_s)
        bc = E.bc_value(I, E.L0, E.eps0)
        tau_end = E0.K() / abs(bc)
        t_eval = np.linspace(0.0, tau_end * E.R_s ** 2 / E.D_s, 100)
        _, _, xs = E.solve(bc, t_eval)
        curves[side] = (t_eval, xs)
    t_end = min(curves['ne'][0][-1], curves['pe'][0][-1])
    t = np.linspace(0.0, t_end, 100)
    xs_n = np.interp(t, curves['ne'][0], curves['ne'][1])
    xs_p = np.interp(t, curves['pe'][0], curves['pe'][1])
    eta = {}
    for side in ('ne', 'pe'):
        E0 = ELEC[side]
        E = Electrode(P, side, R_s=r_s_mult * E0.R_s)
        xs = xs_n if side == 'ne' else xs_p
        sgn = 1.0 if side == 'ne' else -1.0
        eps_s = E.eps_s(E.eps0)
        eta[side] = 2 * pu_consts['R'] * pu_consts['T'] / pu_consts['F'] * np.arcsinh(
            sgn * E.R_s * I / (2 * E.k0 * np.sqrt(E.c_e0) * 3 * eps_s * E.L0 * E.c_max *
                                pu_consts['F'] * E.A * np.sqrt(np.maximum(xs * (1 - xs), 0))))
    import scipy.interpolate as si
    def ocv(side):
        key = 'negativeElectrode' if side == 'ne' else 'positiveElectrode'
        f = P[key]['active_materials'][0]['openCircuitPotential']['value']
        d = np.loadtxt(os.path.join(ROOT, 'data', f))
        return si.interp1d(d[:, 0], d[:, 1], fill_value='extrapolate')
    V = ocv('pe')(xs_p) - ocv('ne')(xs_n) + eta['pe'] - eta['ne']
    return t, np.asarray(V), xs_n, xs_p


for rm in (0.7, 1.0, 1.5):
    t_n, V_n, xn_n, xp_n = voltage_curve(rm)
    t_r, V_r, xn_r, xp_r = voltage_ref(rm)
    n = min(len(t_n), len(t_r))
    # 物理窗口：参考解表面嵌锂度未越界（过放电尾段两侧都不物理，不参与指标）
    phys = (xn_r >= 0.02) & (xp_r <= 0.99) & np.isfinite(V_n[:n]) & np.isfinite(V_r[:n])
    dv = np.abs(V_n[:n][phys] - V_r[:n][phys])
    report['voltage'][f'R_s_mult={rm}'] = {'rms_mV': float(np.sqrt(np.mean(dv ** 2)) * 1000),
                                           'max_mV': float(np.max(dv) * 1000), 'n_phys': int(phys.sum())}
    print(f'R_s×{rm}: V(net) vs V(ref): RMS={report["voltage"][f"R_s_mult={rm}"]["rms_mV"]:.1f} mV, '
          f'max={report["voltage"][f"R_s_mult={rm}"]["max_mV"]:.1f} mV (n_phys={phys.sum()}/{n})', flush=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(t_r[:n], V_r[:n], 'k-', lw=2, label='reference solver')
    ax.plot(t_n[:n], V_n[:n], 'b--', lw=1.5, label=f'universal PINN (R_s×{rm})')
    ax.set_xlabel('t [s]'); ax.set_ylabel('V [V]'); ax.legend(); ax.set_title(f'universal PINN voltage, R_s×{rm}')
    fig.savefig(os.path.join(OUT, f'uni_voltage_Rs{rm}.png'), dpi=120, bbox_inches='tight')
    plt.close(fig)

with open(os.path.join(OUT, 'validation_report.json'), 'w') as f:
    json.dump(report, f, indent=1)
print('\nreport saved ->', OUT, flush=True)

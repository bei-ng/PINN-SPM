"""独立 SPM 数值参考解（与原仓库零依赖，只用 numpy/scipy + 参数 JSON）。

物理：单粒子模型球坐标扩散，嵌锂度 x = c/c_max 满足
    ∂x/∂t = D_s (1/r²) ∂/∂r (r² ∂x/∂r),  r ∈ (0, R_s]
    表面 BC:  ∂x/∂r|_{R_s} = bc / R_s          （bc = 无量纲表面通量参数，与 PINN_utils.py 同式）
    对称 BC:  ∂x/∂r|_0 = 0
    初始:     x(r, 0) = x0 （z=1 满电，负极 0.88 / 正极 0.32）

bc 与几何/倍率的换算（同 PINN_utils.obtain_bcs）：
    bc = ±R_s²·I / (D_s·3·ε_s·L·c_max·F·A),  I = min(Q_n,Q_p)/3600·C_rate
    t_max = 3900/h,  h = |bc/bc_ref|（NE 基准 bc_n(1C)，PE 基准 bc_p_limit）

用法：
    python train/reference_spm.py gen       # 生成参考数据集 npz
    python train/reference_spm.py selftest  # 质量守恒 + 网格收敛自检
"""
import json
import os
import sys

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 与 PINN_utils.py 相同的评估网格约定
N_TIME = 100
N_R = 32
R_MIN_ABS = 0.2e-6       # r̂ 网格起点（绝对值），避开 r=0
T_MAX_BASE = 3900.0      # 1C 满放电时长基准 [s]


def load_params():
    with open(os.path.join(ROOT, 'data', 'params_GrSi_NMC811.json')) as f:
        return json.load(f)


class Electrode:
    """单个电极的几何/材料常数 + 参考求解器。"""

    def __init__(self, p, side):
        key = 'negativeElectrode' if side == 'ne' else 'positiveElectrode'
        e = p[key]
        am = e['active_materials'][0]
        self.side = side
        self.F = p['constants']['F']['value']
        self.L0 = e['thickness']['value']
        self.A = e['area']['value']
        self.eps0 = e['porosity']['value']
        self.eps_s0 = am['volFrac_active']['value']
        self.frac_inert = 1 - self.eps0 - self.eps_s0
        self.R_s = am['particleRadius']['value']
        self.c_max = am['maximumConcentration']['value']
        self.D_s = am['diffusionConstant']['value']
        self.stoi1 = am['stoichiometry1']['value']
        self.stoi0 = am['stoichiometry0']['value']
        self.k0 = am['kineticConstant']['value']
        self.c_e0 = p['electrolyte']['initialConcentration']['value']
        self.Q0 = self._capacity(self.L0, self.eps_s0)  # 默认几何 1C 容量 [C]

    def eps_s(self, porosity):
        return 1 - self.frac_inert - porosity

    def _capacity(self, L, eps_s):
        return self.A * self.F * L * eps_s * self.c_max * abs(self.stoi1 - self.stoi0)

    def current(self, thickness_n, thickness_p, porosity_n, porosity_p, c_rate,
                Q_n, Q_p):
        """与 obtain_bcs 相同：I 由 min(Q_n,Q_p) 决定。Q_n/Q_p 由调用方按两极几何算好传入。"""
        Q = min(Q_n, Q_p)
        return Q / 3600.0 * c_rate

    def bc_value(self, I, thickness, porosity):
        eps_s = self.eps_s(porosity)
        sign = -1.0 if self.side == 'ne' else 1.0
        return sign * self.R_s**2 * I / (self.D_s * 3 * eps_s * thickness
                                         * self.c_max * self.F * self.A)

    def bc_ref(self):
        """bc 基准：NE 用默认几何 1C；PE 用 Q_pos/3600（即仓库硬编码 bc_p_limit 的来源）。"""
        I = self.Q0 / 3600.0
        return self.bc_value(I, self.L0, self.eps0)

    def t_max(self, bc):
        h = abs(bc / self.bc_ref())
        return T_MAX_BASE / h

    def x0(self):
        """z=1 满电初始嵌锂度。"""
        return self.stoi1

    # ---------------- 数值求解器 ----------------

    def solve(self, bc, t_eval, n_cells=200, rtol=1e-8, atol=1e-10):
        """球坐标有限体积 + BDF。返回 (shell 中心 r, shell 平均 x(t,:), 表面 x(t))。"""
        bc = float(bc)                                  # 兼容 TF 张量输入
        R_s, D = self.R_s, self.D_s
        dr = R_s / n_cells
        r_c = (np.arange(n_cells) + 0.5) * dr          # 壳中心
        i = np.arange(n_cells)
        r_out = (i + 1) * dr                            # r_{i+1/2}
        r_in = i * dr                                   # r_{i-1/2}
        vol = r_out**3 - r_in**3                        # ∝ 壳体积（4π/3 约去）
        bc_surf_grad = bc / R_s                         # ∂x/∂r|_R

        def rhs(t, y):
            dx = np.zeros_like(y)
            # 内界面通量项（i=0 处为零：对称）
            grad_in = (y[1:] - y[:-1]) / dr
            flux_in = r_in[1:]**2 * grad_in             # 作用于 i=1..N-1 的内界面
            # 外界面：i=0..N-2 用中心差分，i=N-1 用 BC
            grad_out = np.empty(n_cells)
            grad_out[:-1] = (y[1:] - y[:-1]) / dr
            grad_out[-1] = bc_surf_grad
            flux_out = r_out**2 * grad_out
            dx[:] = 3.0 * D * (flux_out - np.concatenate(([0.0], flux_in))) / vol
            return dx

        y0 = np.full(n_cells, self.x0())
        sol = solve_ivp(rhs, (t_eval[0], t_eval[-1]), y0, method='BDF',
                        t_eval=t_eval, rtol=rtol, atol=atol)
        if not sol.success:
            raise RuntimeError(f'solve_ivp failed: {sol.message}')
        x_shell = sol.y.T                               # (n_t, n_cells)
        # 表面值：外推 x(R) = x_{N-1} + (R - r_{N-1})·∂x/∂r|_R
        x_surf = x_shell[:, -1] + (R_s - r_c[-1]) * bc_surf_grad
        return r_c, x_shell, x_surf

    def x_at(self, bc, t_eval, r_eval, n_cells=200):
        """在任意 (t, r) 评估点插值参考解。返回 (n_t, n_r)。"""
        r_c, x_shell, x_surf = self.solve(bc, t_eval, n_cells)
        # 壳中心 + 表面点作为样条节点（保证 r=R_s 处精确）
        knots_r = np.concatenate((r_c, [self.R_s]))
        out = np.empty((len(t_eval), len(r_eval)))
        for k, x_row in enumerate(x_shell):
            knots_x = np.concatenate((x_row, [x_surf[k]]))
            sp = CubicSpline(knots_r, knots_x)
            out[k] = sp(r_eval)
        return out

    def mass_check(self, bc, t_final, n_cells=200):
        """质量守恒自检：Δ(总锂量) 应等于 表面通量×面积×时间。"""
        R_s, D = self.R_s, self.D_s
        dr = R_s / n_cells
        r_c, x_shell, _ = self.solve(bc, np.array([0.0, t_final]), n_cells)
        # 壳体积：r ∈ [i·dr, (i+1)·dr]
        i = np.arange(n_cells)
        vol = 4.0 / 3.0 * np.pi * (((i + 1) * dr)**3 - (i * dr)**3)
        dM = np.sum((x_shell[1] - x_shell[0]) * vol) * self.c_max   # mol
        j = -bc * self.c_max * D / R_s                              # 表面锂通量（出为正）
        A_tot = 4.0 * np.pi * R_s**2
        M_flux = -j * A_tot * t_final                               # 流入为正 → 与 ΔM 同号
        return dM, M_flux, abs(dM - M_flux) / max(abs(M_flux), 1e-30)


# 训练/归一化 bc 区间：与 PINN_utils.py 的 bc_n_space/bc_p_space 精确同式
#   NE: linspace(-0.011509128654707082, 3*bc_n, 2)   （下端点为原仓库硬编码字面量）
#   PE: linspace(0.1*bc_p, 3.33*bc_p, 2)             （两端均由默认几何 1C 电流算出）
_P = load_params()
_NE = Electrode(_P, 'ne')
_PE = Electrode(_P, 'pe')
_I_cell_1c = min(_NE.Q0, _PE.Q0) / 3600.0
BC_RANGE = {
    'ne': (-0.011509128654707082,
           3.0 * _NE.bc_value(_I_cell_1c, _NE.L0, _NE.eps0)),
    'pe': (0.1 * _PE.bc_value(_I_cell_1c, _PE.L0, _PE.eps0),
           3.33 * _PE.bc_value(_I_cell_1c, _PE.L0, _PE.eps0)),
}


def main_gen():
    """生成参考数据集：bc 网格 × (t̂, r̂) 评估网格。"""
    p = load_params()
    out = {}
    for side in ('ne', 'pe'):
        e = Electrode(p, side)
        lo, hi = BC_RANGE[side]
        bcs = np.linspace(lo, hi, 9)
        r_hat = np.linspace(R_MIN_ABS / e.R_s, 1.0, N_R)
        r_abs = r_hat * e.R_s
        xs, ts, surf = [], [], []
        for bc in bcs:
            t_max = e.t_max(bc)
            t_hat = np.linspace(0.0, 1.0, N_TIME)
            t_eval = t_hat * t_max
            r_c, x_shell, x_s = e.solve(bc, t_eval, n_cells=400)
            # 壳中心 → r̂ 网格插值
            knots_r = np.concatenate((r_c, [e.R_s]))
            xg = np.empty((N_TIME, N_R))
            for k in range(N_TIME):
                knots_x = np.concatenate((x_shell[k], [x_s[k]]))
                xg[k] = CubicSpline(knots_r, knots_x)(r_abs)
            xs.append(xg)
            ts.append(t_eval)
            surf.append(x_s)
        out[f'{side}_bc'] = bcs
        out[f'{side}_r_hat'] = r_hat
        out[f'{side}_t_hat'] = np.linspace(0.0, 1.0, N_TIME)
        out[f'{side}_x'] = np.array(xs)          # (n_bc, n_t, n_r)
        out[f'{side}_x_surface'] = np.array(surf)  # (n_bc, n_t)
        out[f'{side}_t_eval'] = np.array(ts)
        out[f'{side}_x0'] = e.x0()
        out[f'{side}_R_s'] = e.R_s
        out[f'{side}_bc_ref'] = e.bc_ref()
        print(f'{side}: {len(bcs)} bc values, bc_ref={e.bc_ref():.6f}, x0={e.x0()}')
    path = os.path.join(ROOT, 'train', 'artifacts', 'reference_dataset.npz')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **out)
    print('saved ->', path)


def main_selftest():
    p = load_params()
    for side in ('ne', 'pe'):
        e = Electrode(p, side)
        for bc in (BC_RANGE[side][0], e.bc_ref(), BC_RANGE[side][1]):
            t_max = e.t_max(bc)
            dM, M_flux, rel = e.mass_check(bc, t_max)
            # 网格收敛：N=200 vs N=400 的表面浓度最大差
            t_eval = np.linspace(0, t_max, 21)
            _, _, x1 = e.solve(bc, t_eval, n_cells=200)
            _, _, x2 = e.solve(bc, t_eval, n_cells=400)
            conv = np.max(np.abs(x1 - x2))
            print(f'{side} bc={bc:+.4f}: mass rel.err={rel:.2e}, '
                  f'grid conv (N200 vs N400)={conv:.2e}, x_surf final={x1[-1]:.4f}')


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'selftest'
    if mode == 'gen':
        main_gen()
    elif mode == 'selftest':
        main_selftest()
    else:
        print(__doc__)

"""万能族 SPM 参考解（自包含，与 train/ 目录零代码依赖）。

数学设计（τ 参数化）：
  无量纲坐标 τ = t·D_s/R_s²，ρ = r/R_s 下，SPM 解族与 D_s、R_s 完全无关：

      ∂x/∂τ = (1/ρ²) ∂/∂ρ (ρ² ∂x/∂ρ)，  表面 BC：∂x/∂ρ|₁ = bc，  初值 x(r,0)=x0

  即存在万能族 F(τ̂, ρ; bc)。R_s、D_s 只通过两个外部映射进入：
      τ̂ = τ/τ_ref（τ_ref = 3900·D_s_ref/R_s_ref²，默认电极 1C 窗口）
      bc = ±R_s²·I / (D_s·3·ε_s·L·c_max·F·A)

  时间带：放电自然终点满足 τ_end·bc ≈ K（每电极常数，K = bc_ref·τ_ref），
  训练/评估在 τ̂ = s·1.1/h 内采样，h = |bc|/|bc_ref|（1.1 为过放电余量）。

  bc 训练包络：|bc|/|bc_ref| ∈ [BC_LO_MULT, BC_HI_MULT] = [0.02, 15]，
  覆盖 R_s ∈ [0.5,2]× × C_rate ∈ [0.1,3] 的乘积域 [0.025, 12]×。

用法：
    python train_universal/reference_spm_uni.py selftest
"""
import json
import os
import sys

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

N_R = 32
R_MIN_ABS = 0.2e-6        # 绝对半径下限（评估网格起点，避开 ρ→0 奇点）
T_WINDOW_1C = 3900.0      # 默认电极 1C 放电窗口 [s]（与原仓库约定一致）
BC_LO_MULT, BC_HI_MULT = 0.02, 15.0   # bc 训练包络（×bc_ref）
BAND_MARGIN = 1.1         # τ 采样带 = 1.1 × K/|bc|（10% 过放电余量）
RHO_MIN_TRAIN = 0.01      # 训练 ρ 下限（覆盖 R_s=2× 时的 ρ_min=0.0122）


def load_params():
    with open(os.path.join(ROOT, 'data', 'params_GrSi_NMC811.json')) as f:
        return json.load(f)


class Electrode:
    """单电极几何/材料常数 + 量纲化参考求解器。R_s/D_s 可覆盖（万能族验证用）。"""

    def __init__(self, p, side, R_s=None, D_s=None):
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
        self.R_s = float(R_s) if R_s is not None else am['particleRadius']['value']
        self.c_max = am['maximumConcentration']['value']
        self.D_s = float(D_s) if D_s is not None else am['diffusionConstant']['value']
        self.stoi1 = am['stoichiometry1']['value']
        self.stoi0 = am['stoichiometry0']['value']
        self.k0 = am['kineticConstant']['value']
        self.c_e0 = p['electrolyte']['initialConcentration']['value']
        self.sign = -1.0 if side == 'ne' else 1.0
        self.Q0 = self._capacity(self.L0, self.eps_s0)   # 默认几何 1C 容量 [C]（与 R_s 无关）

    def eps_s(self, porosity):
        return 1 - self.frac_inert - porosity

    def _capacity(self, L, eps_s):
        return self.A * self.F * L * eps_s * self.c_max * abs(self.stoi1 - self.stoi0)

    def bc_value(self, I, thickness, porosity):
        """无量纲边界通量（含本电极 R_s 覆盖）。"""
        eps_s = self.eps_s(porosity)
        return self.sign * self.R_s ** 2 * I / (self.D_s * 3 * eps_s * thickness
                                                * self.c_max * self.F * self.A)

    def bc_ref(self):
        """bc 基准（×1）：NE 用默认几何 1C 电流；PE 用 Q_pos/3600（原仓库 bc_p_limit 的来源）。"""
        I = self.Q0 / 3600.0
        return self.bc_value(I, self.L0, self.eps0)

    def x0(self):
        return self.stoi1

    # ---- 无量纲常数 ----
    def tau_ref(self):
        return T_WINDOW_1C * self.D_s / self.R_s ** 2      # 用参考参数构造的电极上 = 3900·D/R²

    def K(self):
        """τ_end·bc ≈ K = bc_ref·τ_ref（每电极常数）。"""
        return abs(self.bc_ref()) * self.tau_ref()

    def h_of(self, bc):
        return abs(bc) / abs(self.bc_ref())

    def bc_norm(self, bc):
        """bc → [0,1]（按包络 min-max，符号每电极固定，只归一化幅值）。"""
        u = abs(bc) / abs(self.bc_ref())
        return (u - BC_LO_MULT) / (BC_HI_MULT - BC_LO_MULT)

    def bc_denorm(self, bc_n):
        return self.sign * (BC_LO_MULT + bc_n * (BC_HI_MULT - BC_LO_MULT)) * abs(self.bc_ref())

    def tau_hat_band(self, bc):
        """该 bc 的 τ̂ 采样带上限 = 1.1/h（τ̂ = τ/τ_ref）。"""
        return BAND_MARGIN / self.h_of(bc)

    def tau_hat_max_global(self):
        return BAND_MARGIN / BC_LO_MULT

    def t_of_tau(self, tau, R_s=None, D_s=None):
        """物理时间 t = τ·R_s²/D_s（默认用本电极参数）。"""
        R_s = self.R_s if R_s is None else R_s
        D_s = self.D_s if D_s is None else D_s
        return tau * R_s ** 2 / D_s

    # ---- 量纲化求解器（有限体积 + BDF，支持 R_s/D_s 覆盖） ----
    def solve(self, bc, t_eval, n_cells=400, rtol=1e-8, atol=1e-10, x_init=None):
        bc = float(bc)
        R_s, D = self.R_s, self.D_s
        dr = R_s / n_cells
        r_c = (np.arange(n_cells) + 0.5) * dr
        i = np.arange(n_cells)
        r_out = (i + 1) * dr
        r_in = i * dr
        vol = r_out ** 3 - r_in ** 3
        bc_surf_grad = bc / R_s

        def rhs(t, y):
            grad_in = (y[1:] - y[:-1]) / dr
            flux_in = r_in[1:] ** 2 * grad_in
            grad_out = np.empty(n_cells)
            grad_out[:-1] = (y[1:] - y[:-1]) / dr
            grad_out[-1] = bc_surf_grad
            flux_out = r_out ** 2 * grad_out
            return 3.0 * D * (flux_out - np.concatenate(([0.0], flux_in))) / vol

        y0 = np.full(n_cells, self.x0() if x_init is None else float(x_init))
        sol = solve_ivp(rhs, (t_eval[0], t_eval[-1]), y0, method='BDF',
                        t_eval=t_eval, rtol=rtol, atol=atol)
        if not sol.success:
            raise RuntimeError(f'solve_ivp failed: {sol.message}')
        x_shell = sol.y.T
        x_surf = x_shell[:, -1] + (R_s - r_c[-1]) * bc_surf_grad
        return r_c, x_shell, x_surf

    def x_at(self, bc, t_eval, r_eval, n_cells=400):
        """任意 (t, r) 点插值（ρ 网格随 R_s 变化）。返回 (n_t, n_r)。"""
        r_c, x_shell, x_surf = self.solve(bc, t_eval, n_cells)
        knots_r = np.concatenate((r_c, [self.R_s]))
        out = np.empty((len(t_eval), len(r_eval)))
        for k in range(len(t_eval)):
            knots_x = np.concatenate((x_shell[k], [x_surf[k]]))
            out[k] = CubicSpline(knots_r, knots_x)(r_eval)
        return out

    def mass_check(self, bc, t_final, n_cells=400):
        dr = self.R_s / n_cells
        r_c, x_shell, _ = self.solve(bc, np.array([0.0, t_final]), n_cells)
        i = np.arange(n_cells)
        vol = 4.0 / 3.0 * np.pi * (((i + 1) * dr) ** 3 - (i * dr) ** 3)
        dM = np.sum((x_shell[1] - x_shell[0]) * vol) * self.c_max
        j = -bc * self.c_max * self.D_s / self.R_s
        M_flux = -j * 4.0 * np.pi * self.R_s ** 2 * t_final
        return dM, M_flux, abs(dM - M_flux) / max(abs(M_flux), 1e-30)


# 模块级单例（参考参数）
_P = load_params()
ELEC = {'ne': Electrode(_P, 'ne'), 'pe': Electrode(_P, 'pe')}
I_CELL_1C = min(ELEC['ne'].Q0, ELEC['pe'].Q0) / 3600.0


def bc_envelope(side):
    """bc 训练包络（带符号）。"""
    E = ELEC[side]
    return E.sign * BC_LO_MULT * abs(E.bc_ref()), E.sign * BC_HI_MULT * abs(E.bc_ref())


def main_selftest():
    for side in ('ne', 'pe'):
        E = ELEC[side]
        print(f'===== {side.upper()} =====')
        print(f'bc_ref={E.bc_ref():+.6f}  tau_ref={E.tau_ref():.4f}  K={E.K():.5f}  '
              f'tau_hat_max={E.tau_hat_max_global():.1f}')
        print(f'bc envelope: [{bc_envelope(side)[0]:+.5f}, {bc_envelope(side)[1]:+.5f}]')
        # 质量守恒（参考 R_s）
        for mult in (0.5, 1.0, 2.0):
            bc = E.sign * mult * abs(E.bc_ref())
            t_end = E.t_of_tau(E.K() / abs(bc))
            dM, M_flux, rel = E.mass_check(bc, t_end)
            print(f'  R_s×{mult}: bc={bc:+.4f}  mass rel.err={rel:.2e}')
        # 万能族数学验证：不同 R_s、相同 bc 的参考解在 (τ̂, ρ) 坐标下应重合
        #   τ = τ̂·τ_ref（τ_ref 是固定归一化常数）；t = τ·R_s²/D_s
        bc = E.sign * 1.5 * abs(E.bc_ref())
        tau_hat = np.linspace(0.05, 1.0, 8) * E.tau_hat_band(bc)
        xs = []
        for mult in (0.6, 1.0, 1.7):
            E2 = Electrode(_P, side, R_s=mult * ELEC[side].R_s)
            t_eval = [E2.t_of_tau(th * E.tau_ref()) for th in tau_hat]
            _, _, x_s = E2.solve(bc, np.array(t_eval))
            xs.append(x_s)
        spread = np.max(np.abs(xs[0] - xs[1]))
        spread2 = np.max(np.abs(xs[1] - xs[2]))
        print(f'  universal-family check (bc=1.5×ref, R_s×{{0.6,1,1.7}}): '
              f'max spread = {max(spread, spread2):.2e} (应 ~求解器容差)')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'selftest':
        main_selftest()
    else:
        print(__doc__)

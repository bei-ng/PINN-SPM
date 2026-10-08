"""my_PINN_utils.py —— 仿照 PINN_utils.py，改用重训模型、只画放电曲线。

与原 PINN_utils.py 的差异：
  1. 模型加载自 train/artifacts/models/SPM NE|PE（重训产物，与原版同格式 SavedModel）；
  2. plot_Battery 不再画浓度云图（obtain_x_n/p 不绘图，只返回数组），只画放电曲线；
  3. 修复原版瑕疵：过电位公式改用用户传入的 thickness_n/thickness_p（原版固定用 L_n/L_p）；
  4. 放电末端 x_surf 越出 (0,1) 的点置 NaN（原版会产生 NaN/inf 电压）；
  5. plot_Battery 返回结果字典（原版无返回值）；
  6. 路径基于本文件位置解析，不依赖当前工作目录。

用法与原版一致（在任意目录）：
    from my_PINN_utils import *
    plot_Battery(thickness_n=L_n, thickness_p=L_p,
                 porosity_n=epsilon_n, porosity_p=epsilon_p, C_rate=1)
换模型目录：改下面的 MODEL_DIR。
"""
import json
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import scipy
import tensorflow as tf

warnings.filterwarnings('ignore')

_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(_ROOT, 'train', 'artifacts', 'models')   # 重训模型目录

# ---------------- 模型与参数（对应原版 10-71 行） ----------------
U_model_n = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'SPM NE'))
U_model_p = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'SPM PE'))

with open(os.path.join(_ROOT, 'data', 'params_GrSi_NMC811.json')) as f:
    parameters = json.loads(f.read())

R = tf.constant(parameters['constants']['R']['value'], dtype=tf.float64)
F = tf.constant(parameters['constants']['F']['value'], dtype=tf.float64)
alpha = tf.constant(parameters['constants']['alpha']['value'], dtype=tf.float64)
# negative electrode
L_n = parameters['negativeElectrode']['thickness']['value']
A_n = parameters['negativeElectrode']['area']['value']
epsilon_n = tf.constant(parameters['negativeElectrode']['porosity']['value'], dtype=tf.float64)
sigma_n = tf.constant(parameters['negativeElectrode']['electronicConductivity']['value'], dtype=tf.float64)
k_0_n = tf.constant(parameters['negativeElectrode']['active_materials'][0]['kineticConstant']['value'], dtype=tf.float64)
c_s_max_n = tf.constant(parameters['negativeElectrode']['active_materials'][0]['maximumConcentration']['value'], dtype=tf.float64)
R_s_n = tf.constant(parameters['negativeElectrode']['active_materials'][0]['particleRadius']['value'], dtype=tf.float64)
epsilon_s_n = tf.constant(parameters['negativeElectrode']['active_materials'][0]['volFrac_active']['value'], dtype=tf.float64)
D_s_n = tf.constant(parameters['negativeElectrode']['active_materials'][0]['diffusionConstant']['value'], dtype=tf.float64)
# separator
L_s = parameters['separator']['thickness']['value']
epsilon_sep = tf.constant(parameters['separator']['porosity']['value'], dtype=tf.float64)
# positive electrode
L_p = parameters['positiveElectrode']['thickness']['value']
A_p = parameters['positiveElectrode']['area']['value']
epsilon_p = tf.constant(parameters['positiveElectrode']['porosity']['value'], dtype=tf.float64)
sigma_p = tf.constant(parameters['positiveElectrode']['electronicConductivity']['value'], dtype=tf.float64)
k_0_p = tf.constant(parameters['positiveElectrode']['active_materials'][0]['kineticConstant']['value'], dtype=tf.float64)
c_s_max_p = tf.constant(parameters['positiveElectrode']['active_materials'][0]['maximumConcentration']['value'], dtype=tf.float64)
R_s_p = tf.constant(parameters['positiveElectrode']['active_materials'][0]['particleRadius']['value'], dtype=tf.float64)
epsilon_s_p = tf.constant(parameters['positiveElectrode']['active_materials'][0]['volFrac_active']['value'], dtype=tf.float64)
D_s_p = tf.constant(parameters['positiveElectrode']['active_materials'][0]['diffusionConstant']['value'], dtype=tf.float64)

c_e_0 = tf.constant(parameters['electrolyte']['initialConcentration']['value'], dtype=tf.float64)

stoi_1_n = parameters['negativeElectrode']['active_materials'][0]['stoichiometry1']['value']
stoi_0_n = parameters['negativeElectrode']['active_materials'][0]['stoichiometry0']['value']
stoi_1_p = parameters['positiveElectrode']['active_materials'][0]['stoichiometry1']['value']
stoi_0_p = parameters['positiveElectrode']['active_materials'][0]['stoichiometry0']['value']

z = 1   # 初始 SoC：模型只在满电初始条件下模拟放电
c_s_n_0 = tf.constant(c_s_max_n * (z * (stoi_1_n - stoi_0_n) + stoi_0_n), dtype=tf.float64)
c_s_p_0 = tf.constant(c_s_max_p * (z * (stoi_1_p - stoi_0_p) + stoi_0_p), dtype=tf.float64)
x_n_0 = c_s_n_0 / c_s_max_n
x_p_0 = c_s_p_0 / c_s_max_p

OCP_anode = np.loadtxt(os.path.join(_ROOT, 'data', parameters['negativeElectrode']['active_materials'][0]['openCircuitPotential']['value']))
inter_anode = scipy.interpolate.interp1d(OCP_anode[:, 0], OCP_anode[:, 1], fill_value='extrapolate')
OCP_cathode = np.loadtxt(os.path.join(_ROOT, 'data', parameters['positiveElectrode']['active_materials'][0]['openCircuitPotential']['value']))
inter_cathode = scipy.interpolate.interp1d(OCP_cathode[:, 0], OCP_cathode[:, 1], fill_value='extrapolate')

fracInMat_n = 1 - epsilon_n - epsilon_s_n
fracInMat_p = 1 - epsilon_p - epsilon_s_p

Q_neg = A_n * F * L_n * epsilon_s_n * c_s_max_n * abs(stoi_1_n - stoi_0_n)
Q_pos = A_p * F * L_p * epsilon_s_p * c_s_max_p * abs(stoi_1_p - stoi_0_p)
Q = min(Q_neg, Q_pos)
C_rate = 1
switch_ch_disch = 1
I = Q / 3600 * C_rate * switch_ch_disch
L_0 = L_n + L_s + L_p
T = 298.15

bc_n = -R_s_n ** 2 * I / (D_s_n * 3 * epsilon_s_n * L_n * c_s_max_n * F * A_n)
bc_p = R_s_p ** 2 * I / (D_s_p * 3 * epsilon_s_p * L_p * c_s_max_p * F * A_p)
bc_p_limit = 0.176902333333332934

# 训练区间输入数组（同时充当归一化 lim，与原版同式，保证归一化区间精确一致）
bc_n_space = np.linspace(-0.011509128654707082, 3 * bc_n, 2)
bc_p_space = np.linspace(bc_p * 1e-1, 3.33 * bc_p, 2)


def obtain_bcs(thickness_n, thickness_p, porosity_n, porosity_p, C_rate,
               obtain_real_parameters=0, ne_or_pe_for_real_parameter='ne',
               min_value_bc=-0.012, max_value_bc=-0.010):
    """设计参数 → (bc_n, bc_p, I)。筛选模式与原版相同。"""
    epsilon_s_n_new = 1 - fracInMat_n - porosity_n
    epsilon_s_p_new = 1 - fracInMat_p - porosity_p

    Q_neg_new = A_n * F * thickness_n * epsilon_s_n_new * c_s_max_n * abs(stoi_1_n - stoi_0_n)
    Q_pos_new = A_p * F * thickness_p * epsilon_s_p_new * c_s_max_p * abs(stoi_1_p - stoi_0_p)

    Q_new = min(Q_neg_new, Q_pos_new)
    I_new = Q_new / 3600 * C_rate

    bc_n_new = -R_s_n ** 2 * I_new / (D_s_n * 3 * epsilon_s_n_new * thickness_n * c_s_max_n * F * A_n)
    bc_p_new = R_s_p ** 2 * I_new / (D_s_p * 3 * epsilon_s_p_new * thickness_p * c_s_max_p * F * A_p)

    if obtain_real_parameters:
        if ne_or_pe_for_real_parameter == 'ne':
            if min_value_bc < bc_n_new < max_value_bc:
                return thickness_n, thickness_p, porosity_n, porosity_p, C_rate
        if ne_or_pe_for_real_parameter == 'pe':
            if min_value_bc < bc_p_new < max_value_bc:
                return thickness_n, thickness_p, porosity_n, porosity_p, C_rate
    else:
        return bc_n_new, bc_p_new, I_new


def normalize_Inputs_for_OCV(Inputs, Inputs_lim):
    """bc 与 t_max 两列按训练区间 min-max 归一化（t̂、r̂ 已是归一化量，原样保留）。"""
    t_norm = Inputs[:, 0]
    bc_norm = (Inputs[:, 2] - min(Inputs_lim[:, 2])) / (max(Inputs_lim[:, 2]) - min(Inputs_lim[:, 2]))
    t_max_arr = (Inputs[:, 3] - min(Inputs_lim[:, 3])) / (max(Inputs_lim[:, 3]) - min(Inputs_lim[:, 3]))
    return np.stack((t_norm, Inputs[:, 1], bc_norm, t_max_arr), axis=1)


def create_input_from_bc(ne_or_pe, bc):
    """单电极评估网格：100 时刻 × 32 半径点，行 = [t̂, r̂, bc, t_max]（未归一化）。"""
    bc = float(bc)
    if ne_or_pe == 'ne':
        h = abs(bc / bc_n)
        R_s = R_s_n
    else:
        h = abs(bc / bc_p_limit)
        R_s = R_s_p
    time_steps, r_steps = 100, 32
    t_max = 3900 / h
    time = np.linspace(0, t_max, time_steps) / t_max
    r_arr = np.linspace(0.2e-6, float(R_s), r_steps) / float(R_s)
    return np.array([[t, r, bc, t_max] for t in time for r in r_arr])


# 训练区间输入数组（保留原版构造方式；用于越界检查与归一化 lim）
Inputs_n = create_input_from_bc('ne', bc_n_space[0])
for bc_elem in bc_n_space:
    Inputs_n = np.concatenate((Inputs_n, create_input_from_bc('ne', bc_elem)), 0)
Inputs_n = np.delete(Inputs_n, np.s_[0:len(create_input_from_bc('ne', bc_n_space[0]))], 0)

Inputs_p = create_input_from_bc('pe', bc_p_space[0])
for bc_elem in bc_p_space:
    Inputs_p = np.concatenate((Inputs_p, create_input_from_bc('pe', bc_elem)), 0)
Inputs_p = np.delete(Inputs_p, np.s_[0:len(create_input_from_bc('pe', bc_p_space[0]))], 0)


def obtain_position(value_norm, variable, Input):
    """在输入数组中查找某归一化值所在的所有行号（'time' 查第 0 列，'space' 查第 1 列）。"""
    col = 0 if variable == 'time' else 1
    return [pos for pos, elem in enumerate(Input[:, col]) if elem == value_norm]


def obtain_x_n(select_or_input, n, input, t_max_n_input=0):
    """负极前向：返回 (完整浓度场 x_n, 表面浓度序列, 时间轴, t_max)，不绘图。"""
    Input = input
    t_max = t_max_n_input
    position_frontier_r_max = obtain_position(Input[:, 1][-1], 'space', Input)
    x_barra_n = tf.squeeze(tf.cast(U_model_n(Input), tf.float64))
    x_n = x_barra_n + x_n_0
    x_n_surface = np.array([x_n[pos] for pos in position_frontier_r_max])
    return x_n, x_n_surface, np.linspace(0, 1, 100), t_max


def obtain_x_p(select_or_input, n, input, t_max_p_input=0):
    """正极前向：返回 (完整浓度场 x_p, 表面浓度序列, 时间轴, t_max)，不绘图。"""
    Input = input
    t_max = t_max_p_input
    position_frontier_r_max = obtain_position(Input[:, 1][-1], 'space', Input)
    x_barra_p = tf.squeeze(tf.cast(U_model_p(Input), tf.float64))
    x_p = x_barra_p + x_p_0
    x_p_surface = np.array([x_p[pos] for pos in position_frontier_r_max])
    return x_p, x_p_surface, np.linspace(0, 1, 100), t_max


def plot_Battery(thickness_n=L_n, thickness_p=L_p, porosity_n=epsilon_n,
                 porosity_p=epsilon_p, C_rate=1, experimental_t_or_f=False, exp=0):
    """给定设计参数与倍率，绘制放电电压曲线（不画浓度云图）。

    返回 dict：t, V, x_n_surface, x_p_surface, bc_n, bc_p, I,
               t_max_n, t_max_p, in_bounds；越界时返回 None 并打印提示。
    """
    bc_n_OCV, bc_p_OCV, I_new = obtain_bcs(thickness_n, thickness_p, porosity_n, porosity_p, C_rate)

    out_of_bounds = []
    if abs(bc_n_OCV) < abs(Inputs_n[:, 2][0]):
        out_of_bounds.append('NE')
    if abs(bc_p_OCV) < abs(Inputs_p[:, 2][0]):
        out_of_bounds.append('PE')
    if out_of_bounds:
        print('Out of bounds ' + '/'.join(out_of_bounds))
        return None

    epsilon_s_n_ocv = 1 - fracInMat_n - porosity_n
    epsilon_s_p_ocv = 1 - fracInMat_p - porosity_p

    Input_n = normalize_Inputs_for_OCV(create_input_from_bc('ne', bc_n_OCV), Inputs_n)
    Input_p = normalize_Inputs_for_OCV(create_input_from_bc('pe', bc_p_OCV), Inputs_p)
    t_max_n = create_input_from_bc('ne', bc_n_OCV)[0, 3]
    t_max_p = create_input_from_bc('pe', bc_p_OCV)[0, 3]

    _, x_n_surface, _, _ = obtain_x_n('input', 0, Input_n, t_max_n)
    _, x_p_surface, _, _ = obtain_x_p('input', 0, Input_p, t_max_p)

    t_total = min(t_max_n, t_max_p)
    t_arr = np.linspace(0, t_total, 100)
    x_n_surf = np.interp(t_arr, np.linspace(0, t_max_n, 100), x_n_surface)
    x_p_surf = np.interp(t_arr, np.linspace(0, t_max_p, 100), x_p_surface)

    # Butler–Volmer 过电位（修复：厚度用请求值而非默认 L_n/L_p）
    eta_n = 2 * R * T / F * np.arcsinh(
        R_s_n * I_new / (2 * k_0_n * np.sqrt(c_e_0) * 3 * epsilon_s_n_ocv * thickness_n
                         * c_s_max_n * F * A_n * np.sqrt(np.maximum(x_n_surf * (1 - x_n_surf), 0))))
    eta_p = 2 * R * T / F * np.arcsinh(
        -R_s_p * I_new / (2 * k_0_p * np.sqrt(c_e_0) * 3 * epsilon_s_p_ocv * thickness_p
                          * c_s_max_p * F * A_p * np.sqrt(np.maximum(x_p_surf * (1 - x_p_surf), 0))))

    V = np.asarray(inter_cathode(x_p_surf) - inter_anode(x_n_surf) + eta_p - eta_n)
    # 放电末端 x_surf 越出 (0,1) 的时刻置 NaN（物理上已过放电）
    valid = (x_n_surf > 0) & (x_n_surf < 1) & (x_p_surf > 0) & (x_p_surf < 1)
    V = np.where(valid, V, np.nan)

    plt.figure()
    plt.plot(t_arr, V, '-', linewidth=1.5, color='blue', label='my PINN SPM')
    if experimental_t_or_f:
        plt.plot(exp['# Time [s]'], exp['Voltage [V]'], 'r', label='FEM SPM', alpha=1)
    plt.xlabel(r'$t$ [s]')
    plt.ylabel(r'$Voltage$ [V]')
    for label in (r'$thickness_{NE} = %0.2f$ µm' % (thickness_n * 1e6),
                  r'$thickness_{PE} = %0.2f$ µm' % (thickness_p * 1e6),
                  r'$porosity_{NE} = %0.3f$' % porosity_n,
                  r'$porosity_{PE} = %0.3f$' % porosity_p,
                  r'$C_{Rate} = %0.3f$' % C_rate):
        plt.plot([], [], ' ', label=label)   # 图例占位（无可见线）
    plt.legend(bbox_to_anchor=(1, 0.5))

    return {'t': t_arr, 'V': V, 'x_n_surface': x_n_surf, 'x_p_surface': x_p_surf,
            'bc_n': float(bc_n_OCV), 'bc_p': float(bc_p_OCV), 'I': float(I_new),
            't_max_n': float(t_max_n), 't_max_p': float(t_max_p),
            'in_bounds': True}


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    out = plot_Battery(thickness_n=L_n, thickness_p=L_p,
                       porosity_n=epsilon_n, porosity_p=epsilon_p, C_rate=1)
    print('default 1C: V[0]=%.4f, V_end(valid)=%.4f, points=%d' % (
        out['V'][0], out['V'][~np.isnan(out['V'])][-1], np.sum(~np.isnan(out['V']))))
    plt.savefig(os.path.join(_ROOT, 'train', 'artifacts', 'my_pinnt_demo_1C.png'),
                dpi=120, bbox_inches='tight')

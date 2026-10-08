"""Step 0b: 用 pinn210 环境对原仓库做回归基准测试。

在仓库根目录运行：  conda run -n pinn210 python train/regression_baseline.py
产物：train/artifacts/baseline/ 下的图片 + 数值基线 JSON。
"""
import os
import sys
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tensorflow as tf
from PINN_utils import (plot_Battery, obtain_bcs, create_input_from_bc,
                        normalize_Inputs_for_OCV, obtain_x_n, obtain_x_p,
                        Inputs_n, Inputs_p, L_n, L_p, epsilon_n, epsilon_p,
                        inter_anode, inter_cathode, R, T, F, k_0_n, k_0_p,
                        c_s_max_n, c_s_max_p, c_e_0, R_s_n, R_s_p, D_s_n, D_s_p,
                        A_n, A_p, fracInMat_n, fracInMat_p, stoi_1_n, stoi_0_n,
                        stoi_1_p, stoi_0_p)

OUT = os.path.join('train', 'artifacts', 'baseline')
os.makedirs(OUT, exist_ok=True)

print('TF:', tf.__version__, '| GPUs:', tf.config.list_physical_devices('GPU'))

cases = [
    ('default_1C',  L_n,   L_p,   epsilon_n, epsilon_p, 1),
    ('thick_n_2e4_1C', 2e-4, 5e-5, 0.2, 0.2, 1),
    ('thick_n_2e4_por60_2C', 2e-4, 5e-5, 0.6, 0.6, 2),
    ('thin_5e5_1C', 5e-5, 5e-5, 0.2, 0.2, 1),
    ('default_3C', L_n,   L_p,   epsilon_n, epsilon_p, 3),
]

baseline = {}
for name, tn, tp, pn, pp, cr in cases:
    print(f'--- case {name}: L_n={tn}, L_p={tp}, eps_n={pn}, eps_p={pp}, C={cr}')
    plot_Battery(thickness_n=tn, thickness_p=tp, porosity_n=pn, porosity_p=pp, C_rate=cr)
    figs = plt.get_fignums()
    for k, fnum in enumerate(figs):
        plt.figure(fnum).savefig(os.path.join(OUT, f'{name}_fig{k}.png'), dpi=120, bbox_inches='tight')
    plt.close('all')

    # 数值基线：bc、电流、t_max 与末端电压
    bc_n_o, bc_p_o, I = obtain_bcs(tn, tp, pn, pp, cr)
    in_n = create_input_from_bc('ne', bc_n_o)
    in_p = create_input_from_bc('pe', bc_p_o)
    baseline[name] = {
        'bc_n': float(bc_n_o), 'bc_p': float(bc_p_o), 'I': float(I),
        't_max_n': float(in_n[0, 3]), 't_max_p': float(in_p[0, 3]),
    }

# 表面浓度与电压曲线的数值基线（default_1C 一例，供重训后对比）
tn, tp, pn, pp, cr = L_n, L_p, epsilon_n, epsilon_p, 1
bc_n_o, bc_p_o, I = obtain_bcs(tn, tp, pn, pp, cr)
in_n = create_input_from_bc('ne', bc_n_o)
in_p = create_input_from_bc('pe', bc_p_o)
t_max_n, t_max_p = in_n[0, 3], in_p[0, 3]
xn, xns, tarr_n, _ = obtain_x_n('input', 0, normalize_Inputs_for_OCV(in_n, Inputs_n), t_max_n)
xp, xps, tarr_p, _ = obtain_x_p('input', 0, normalize_Inputs_for_OCV(in_p, Inputs_p), t_max_p)
t_total = min(t_max_n, t_max_p)
t_arr = np.linspace(0, t_total, 100)
xns_i = np.asarray(np.interp(t_arr, np.linspace(0, t_max_n, 100), xns))
xps_i = np.asarray(np.interp(t_arr, np.linspace(0, t_max_p, 100), xps))
eps_s_n = 1 - fracInMat_n - pn
eps_s_p = 1 - fracInMat_p - pp
eta_n = 2*R*T/F*np.arcsinh(R_s_n*I/(2*k_0_n*np.sqrt(c_e_0)*3*eps_s_n*L_n*c_s_max_n*F*A_n*np.sqrt(xns_i*(1-xns_i))))
eta_p = 2*R*T/F*np.arcsinh(-R_s_p*I/(2*k_0_p*np.sqrt(c_e_0)*3*eps_s_p*L_p*c_s_max_p*F*A_p*np.sqrt(xps_i*(1-xps_i))))
V = np.asarray(inter_cathode(xps_i) - inter_anode(xns_i) + eta_p - eta_n)
baseline['default_1C_voltage_curve'] = {
    't': t_arr.tolist(), 'V': V.tolist(),
    'x_n_surface': xns_i.tolist(), 'x_p_surface': xps_i.tolist(),
}

with open(os.path.join(OUT, 'baseline.json'), 'w') as f:
    json.dump(baseline, f, indent=1)
print('baseline saved to', OUT)

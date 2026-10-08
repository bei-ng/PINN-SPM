"""诊断：窄域模型的误差在哪里？损失小而解错 = 公式链条有 bug。"""
import os
import sys
import numpy as np
import tensorflow as tf
from scipy.interpolate import CubicSpline

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
os.chdir(ROOT)
tf.keras.backend.set_floatx('float64')

from train_universal.reference_spm_uni import ELEC, R_MIN_ABS, BC_LO_MULT, BC_HI_MULT

side = 'ne'
E = ELEC[side]
model = tf.keras.models.load_model(os.path.join(HERE, 'artifacts', 'models', 'UNI SPM NE'))

bc = E.sign * 1.0 * abs(E.bc_ref())       # h = 1
h = 1.0
s_grid = np.linspace(0, 1, 41)
rho = np.linspace(R_MIN_ABS / E.R_s, 1.0, 32)
r_abs = rho * E.R_s

# 参考解
t_eval = s_grid * 1.1 / h * 3900.0
r_c, x_shell, x_surf = E.solve(bc, t_eval)
x_ref = np.empty((len(s_grid), 32))
for k in range(len(s_grid)):
    x_ref[k] = CubicSpline(np.concatenate((r_c, [E.R_s])),
                           np.concatenate((x_shell[k], [x_surf[k]])))(r_abs)

# 网络
def net_xbar(s, r):
    T, R = np.meshgrid(s, r, indexing='ij')
    u = abs(bc) / abs(E.bc_ref())
    bn = (u - 0.5) / 1.5          # test_narrow 的归一化
    X = np.stack([T.ravel(), R.ravel(), np.full(T.size, bn)], 1)
    return tf.reshape(model(tf.constant(X, tf.float64), training=False), (-1,)).numpy().reshape(T.shape)

xbar_net = net_xbar(s_grid, rho)
err = xbar_net + E.x0() - x_ref
print('error: rms=%.4f max=%.4f' % (np.sqrt(np.mean(err**2)), np.max(np.abs(err))))
print('err(s) profile (mean over rho):', np.round(err.mean(1)[::8], 3))
print('err(rho) profile (mean over s):', np.round(err.mean(0)[::4], 3))

# 网格化 PDE 残差（数值微分验证，替代 tape）
eps = 1e-4
def net_at(s, r):
    return net_xbar(np.atleast_1d(s), np.atleast_1d(r))
res_map = np.zeros_like(err)
for i, sv in enumerate(s_grid):
    for j, rv in enumerate(rho):
        rv_ = max(rv, 0.02)
        xs = np.array([max(sv - eps, 0), sv, sv + eps])
        rs = np.array([max(rv_ - eps, 0.005), rv_, min(rv_ + eps, 1.0)])
        f = net_xbar(xs, rs)[:, :, None] if False else None
    break
# 用 tape 一次性算（简单起见对抽样点）
idx = [(i, j) for i in range(0, 41, 8) for j in range(0, 32, 4)]
pts_s = np.array([s_grid[i] for i, j in idx])
pts_r = np.array([rho[j] for i, j in idx])
u = abs(bc) / abs(E.bc_ref())
bn = (u - 0.5) / 1.5
Xt = tf.constant(np.stack([pts_s, pts_r, np.full(len(pts_s), bn)], 1), tf.float64)
with tf.GradientTape(persistent=True) as t2:
    t2.watch(Xt)
    with tf.GradientTape(persistent=True) as t1:
        t1.watch(Xt)
        xb = tf.reshape(model(Xt, training=False), (-1,))
    g = t1.gradient(xb, Xt)
    dx_ds, dx_drho = g[:, 0], g[:, 1]
    d2 = t2.gradient(dx_drho, Xt)[:, 1]
del t1, t2
res = dx_ds.numpy() - 1.1 * E.tau_ref() * (d2.numpy() + 2 * dx_drho.numpy() / pts_r)
print('net PDE residual on sampled grid: rms=%.4f max=%.4f' % (np.sqrt(np.mean(res**2)), np.max(np.abs(res))))
print('sample (s, rho, x_net, x_ref, err, res):')
for k in range(0, len(pts_s), 6):
    i, j = idx[k]
    print('  (%.2f, %.2f): net=%.4f ref=%.4f err=%+.4f res=%+.4f' %
          (pts_s[k], pts_r[k], xbar_net[i, j], x_ref[i, j] - E.x0(), err[i, j], res[k]))

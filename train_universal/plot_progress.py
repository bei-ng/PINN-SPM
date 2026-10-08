"""训练进度面板：解析后台日志，绘制损失曲线与 ref-check 轨迹。

    conda run -n pinn210 python train_universal/plot_progress.py [日志路径]

产物：train_universal/artifacts/progress.png
"""
import json
import os
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

DEFAULT_LOG = (r'C:\Users\lenovo\.zcode\cli\exec\sess_52e87722-5b66-4cac-8c5f-'
               r'62ec8d5ad89f\call_a0b601a918ea466cbdfa9db2-stdout.log')
log_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_LOG

text = open(log_path, encoding='utf-8', errors='ignore').read()
# 按"=== training universal"分节（NE / PE）
chunks = re.split(r'=== training universal (\w+) PINN.*?===', text)
# chunks: [前导, side1, body1, side2, body2, ...]
runs = []
for i in range(1, len(chunks) - 1, 2):
    runs.append((chunks[i], chunks[i + 1]))

step_re = re.compile(r'step\s+(\d+)\s+loss=([\d.eE+-]+)\s+\(pde ([\d.eE+-]+) bc ([\d.eE+-]+) '
                     r'sym ([\d.eE+-]+) ic ([\d.eE+-]+)\)\s+\[(\d+)s\]')
ref_re = re.compile(r'\[ref-check\] x RMS=([\d.eE+-]+) max=([\d.eE+-]+)')

fig, axes = plt.subplots(1, 2 if len(runs) else 1, figsize=(13, 4.6), squeeze=False)
colors = {'pde': 'tab:blue', 'bc': 'tab:orange', 'sym': 'tab:green', 'ic': 'tab:red'}

n_done = 0
for ax_idx, (side, body) in enumerate(runs):
    steps, comps = [], {'pde': [], 'bc': [], 'sym': [], 'ic': []}
    for m in step_re.finditer(body):
        steps.append(int(m.group(1)))
        comps['pde'].append(float(m.group(3)))
        comps['bc'].append(float(m.group(4)))
        comps['sym'].append(float(m.group(5)))
        comps['ic'].append(float(m.group(6)))
    ref = [(int(m.start()), float(m.group(1)), float(m.group(2))) for m in ref_re.finditer(body)]
    # ref-check 的 step 号：按出现顺序对应 LOG/PROBE_EVERY 网格
    ref_steps = [s for s in steps if s % 5000 == 0][:len(ref)] if steps else []
    ax = axes[0][ax_idx] if len(runs) > 1 else axes[0][0]
    for k in comps:
        ax.plot(steps, comps[k], label=k, color=colors[k], lw=1.2)
    ax.set_yscale('log')
    ax.set_xlabel('step'); ax.set_ylabel('loss')
    total = 45000
    last_step = steps[-1] if steps else 0
    status = 'done' if 'model saved' in body else f'running {last_step}/{total}'
    ax.set_title(f'{side.upper()}: {status}')
    ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    if ref and ref_steps and len(ref_steps) == len(ref):
        ax2.plot(ref_steps, [r[1] for r in ref], 'k-o', ms=4, lw=1.5, label='ref RMS')
        ax2.plot(ref_steps, [r[2] for r in ref], 'k--s', ms=3, lw=1.0, label='ref max')
        ax2.set_ylabel('ref-check error', color='k')
    else:  # ref-check step 号对不上时按序号画
        ax2.plot(range(len(ref)), [r[1] for r in ref], 'k-o', ms=4, lw=1.5, label='ref RMS')
    if 'final ref-check' in body:
        m = re.search(r'final ref-check: \{.rms.: ([\d.eE+-]+), .max.: ([\d.eE+-]+)', body)
        if m:
            ax2.axhline(float(m.group(1)), color='purple', ls=':', lw=1)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc='upper right')
    if 'model saved' in body:
        n_done += 1

fig.suptitle(f'Universal PINN training progress  ({n_done}/{len(runs) if runs else 0} models saved)', y=1.02)
fig.tight_layout()
out = os.path.join(HERE, 'artifacts', 'progress.png')
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=110, bbox_inches='tight')
print('saved ->', out)

"""诊断实验：窄 bc 域（h∈[0.5,2]）短训，对照旧管线轨迹。

    conda run -n pinn210 python train_universal/test_narrow.py ne 10000
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import train_universal.train_spm_uni as tu

tu.BC_LO_MULT, tu.BC_HI_MULT = 0.5, 2.0        # 窄域（对分实验）
side = sys.argv[1] if len(sys.argv) > 1 else 'ne'
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 10000
sys.argv = ['test_narrow', side, str(steps)]
tu.main()

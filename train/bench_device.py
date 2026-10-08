import os, sys, time
import numpy as np
import tensorflow as tf
tf.keras.backend.set_floatx('float64')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); os.chdir(ROOT)
from train.train_spm import SideTrainer

tr = SideTrainer('ne')
tr.calibrate_bn(50)
for _ in range(20):
    tr.train_step()          # warm up
t0 = time.time()
N = 50
for _ in range(N):
    tr.train_step()
dt = (time.time() - t0) / N
print(f'{N} steps avg {dt*1000:.1f} ms/step -> 30k steps ~ {dt*30000/60:.1f} min')

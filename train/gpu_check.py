import time
import tensorflow as tf

print('TF version:', tf.__version__)
print('Physical devices:', tf.config.list_physical_devices())
gpus = tf.config.list_physical_devices('GPU')
print('GPUs:', gpus)
if gpus:
    with tf.device('/GPU:0'):
        a = tf.random.normal([2000, 2000], dtype=tf.float32)
        b = tf.random.normal([2000, 2000], dtype=tf.float32)
        t0 = time.time()
        c = tf.matmul(a, b)
        _ = c.numpy()
        dt = time.time() - t0
    print(f'GPU matmul 2000x2000 OK in {dt*1000:.0f} ms')
else:
    print('NO GPU DETECTED')

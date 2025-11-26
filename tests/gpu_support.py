import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
print("GPUs available:", gpus)

import tensorflow as tf

print("TensorFlow version:", tf.__version__)
print("Built with CUDA:", tf.sysconfig.get_build_info()['cuda_version'])
print("Built with cuDNN:", tf.sysconfig.get_build_info()['cudnn_version'])


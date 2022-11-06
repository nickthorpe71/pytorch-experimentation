import tensorflow as tf
from tensorflow import keras

t2 = tf.constant(tf.range(1.0, 5), shape=(1, 2, 2, 1))
tf_filter = tf.constant(tf.range(1.0, 5), shape=[2, 2, 1, 1])

tf_transcconv_layer = keras.layers.Conv2DTranspose(
  input_shape=(2, 2, 1),
  filters=1,
  kernel_size=2,
  strides=1,
  use_bias=False,
  weights=[tf_filter],
)

t_transconv = tf_transcconv_layer(t2)
print(tf.reshape(t_transconv, [3,3])) # reshape so it prints nicely
  
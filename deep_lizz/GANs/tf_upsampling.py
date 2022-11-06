import tensorflow as tf

t2 = tf.constant(tf.range(1, 5), shape=(1, 2, 2, 1))
print(tf.reshape(t2, (2, 2)))

t2_resize_nearest = tf.image.resize(t2, size=[4, 4], method='nearest')
t2_resize_nearest = tf.reshape(t2_resize_nearest, shape=[4, 4])
print(t2_resize_nearest)


t2_resize_bilinear = tf.image.resize(t2, size=[4, 4], method='bilinear')
t2_resize_bilinear = tf.reshape(t2_resize_bilinear, shape=[4, 4])
print(t2_resize_bilinear)

# Keras Upsampling2D
upsampling_layer_nearest = tf.keras.layers.UpSampling2D(
    size=(2, 2), interpolation='nearest')

t2_keras_nearest = upsampling_layer_nearest(t2)
t2_keras_nearest = tf.reshape(t2_keras_nearest, shape=[4, 4])
print(t2_keras_nearest)

upsampling_layer_bilinear = tf.keras.layers.UpSampling2D(
    size=(2, 2), interpolation='bilinear')
t2_keras_bilinear = tf.reshape(upsampling_layer_bilinear(t2), shape=[4, 4])
print(t2_keras_bilinear)

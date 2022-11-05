import tensorflow as tf

t2 = tf.constant(tf.range(1,5), shape=(1,2,2,1))
# print(t2)
print(tf.reshape(t2, (2,2)))

t2_resize_nearest = tf.image.resize(t2, size=[4,4], method='nearest')
# print(t2_resize_nearest)

t2_resize_nearest = tf.reshape(t2_resize_nearest, shape=[4,4])
print(t2_resize_nearest)


t2_resize_bilinear = tf.image.resize(t2, size=[4,4], method='bilinear')
# print(t2_resize_bilinear)

t2_resize_bilinear = tf.reshape(t2_resize_bilinear, shape=[4,4])
print(t2_resize_bilinear)
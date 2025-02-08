import tensorflow as tf

a = tf.Variable(0.499, dtype=tf.float32)
b = tf.Variable(0.500, dtype=tf.float32)

print("round(0.499) = ", tf.round(a))
print("round(0.501) = ", tf.round(b))
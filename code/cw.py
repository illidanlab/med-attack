import tensorflow as tf


def cw(model, x, lamb=0, optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01), min_prob=0):

    xshape = x.get_shape().as_list()
    noise = tf.get_variable('noise', xshape, tf.float32, initializer=tf.initializers.zeros)

    # ISTA
    cond1 = tf.cast(tf.greater(noise, lamb), tf.float32)
    cond2 = tf.cast(tf.less_equal(tf.abs(noise), lamb), tf.float32)
    cond3 = tf.cast(tf.less(noise, tf.negative(lamb)), tf.float32)

    assign_noise = tf.multiply(cond1,tf.subtract(noise,lamb)) + tf.multiply(cond2, tf.constant(0.0)) + tf.multiply(cond3, tf.add(noise,lamb))
    setter = tf.assign(noise, assign_noise)

    # Adversarial
    xadv = x + noise
    ybar, logits = model(xadv, logits=True)

    ydim = ybar.get_shape().as_list()[1]
    y = tf.argmin(ybar, axis=1, output_type=tf.int32)

    mask = tf.one_hot(y, ydim, on_value=0.0, off_value=float('inf'))
    yt = tf.reduce_max(logits - mask, axis=1)
    yo = tf.reduce_max(logits, axis=1)

    loss0 = tf.nn.relu(yo - yt + min_prob)

    axis = list(range(1, len(xshape)))
    loss1 = tf.reduce_sum((tf.abs(xadv - x)), axis=axis)

    loss = loss0 + lamb * loss1
    train_op = optimizer.minimize(loss, var_list=[noise])

    return train_op, xadv, noise, setter

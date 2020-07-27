import tensorflow as tf


def tf_accuracy(y_true, y_pred):
    with tf.name_scope('accuracy'):
        y_pred = tf.argmax(y_pred, axis=-1)
        y_true = tf.cast(y_true, y_pred.dtype)
        return tf.reduce_mean(tf.cast(tf.equal(y_pred, y_true), tf.float32))


def tf_perplexity(y_true, y_pred):
    with tf.name_scope('perplexity'):
        loss = - y_pred * tf.math.log(y_pred + 0.0000001)

        # loss *= mask
        loss = tf.reduce_sum(loss, axis=-1)
        loss = tf.reduce_mean(loss)

        _perplexity = tf.pow(2., loss)
        return _perplexity

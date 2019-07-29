import numpy as np

import tensorflow as tf

# Todo: remove data_format

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5
_BLOCK_EXPANSION = 1


def output_layer(inputs, output_shape):
    with tf.variable_scope("output_layer"):
        flattened_output_shape = int(np.prod(output_shape))
        h = tf.layers.dense(inputs=inputs, units=flattened_output_shape)
        if len(output_shape) > 1:
            h = tf.reshape(h, [-1] + list(output_shape))
        h = tf.nn.log_softmax(h, axis=-1)
        return h


def batch_norm(inputs, training, data_format, name=None):
    return tf.layers.batch_normalization(
        inputs=inputs, axis=1 if data_format == 'channels_first' else 3,
        momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
        scale=True, training=training, fused=True, name=name,
    )


def conv2d_fixed_padding(inputs, filters, kernel_size, strides, data_format, name=None):
    return tf.layers.conv2d(
        inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
        # padding=('SAME' if strides == 1 else 'VALID'), use_bias=False,
        padding='SAME', use_bias=False,
        kernel_initializer=tf.variance_scaling_initializer(),
        data_format=data_format,
        name=name,
    )


def conv3x3(inputs, planes, data_format, strides, name=None):
    return conv2d_fixed_padding(
        inputs=inputs,
        filters=planes,
        kernel_size=3,
        strides=strides,
        data_format=data_format,
        name=name,
    )


def conv1x1(inputs, planes, data_format, strides, name=None):
    return conv2d_fixed_padding(
        inputs=inputs,
        filters=planes,
        kernel_size=1,
        strides=strides,
        data_format=data_format,
        name=name,
    )


def basic_block_v2(inputs, planes, training, data_format, strides, downsample=None):
    with tf.variable_scope("basic_block"):
        residual = inputs

        # Phase 1
        ##print(0, inputs.shape)
        out = batch_norm(inputs, training, data_format, name="bn1")
        out = tf.nn.relu(out)
        if downsample:
            """
            residual = conv1x1(x, planes * BLOCK_EXPANSION, data_format, 
                               strides=strides,
                               name="downsample")
            """
            residual = tf.layers.conv2d(
                inputs=out,
                filters=planes * _BLOCK_EXPANSION,
                kernel_size=1,
                strides=strides,
                padding='VALID',
                use_bias=False,
                kernel_initializer=tf.variance_scaling_initializer(),
                data_format=data_format,
                name="downsample",
            )
            ##print(x.shape, residual.shape, strides)
        out = conv3x3(out, planes, data_format, strides=strides, name="conv1")
        ##print(1, out.shape, strides)

        # Phase 2
        out = batch_norm(out, training, data_format, name="bn2")
        out = tf.nn.relu(out)
        out = conv3x3(out, planes, data_format, strides=1, name="conv2")
        ##print(2, out.shape)

        out = out + residual
        ##print(3, out.shape)
        return out


def gaussian_noise_layer(inputs, std, training):
    with tf.variable_scope("gaussian_noise_layer"):
        if training:
            noise = tf.random_normal(tf.shape(inputs), mean=0.0, stddev=std, dtype=tf.float32)
            output = tf.add_n([inputs, noise])
            return output
        else:
            return tf.identity(inputs)


def avg_pool_layer(inputs):
    return tf.reduce_mean(inputs, axis=(2, 3), name="avg_pool")

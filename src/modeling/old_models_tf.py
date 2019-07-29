import tensorflow as tf

from . import resnet_tf as resnet


def resnet_cnn(inputs, parameters, is_train_holder, scope_name, reuse, debug=False, count_unit=None, growth_factor=2):
    """
    Function that constructs a ResNet CNN
    --- First conv sequence ---
    conv[num_filters=16, first_kerneal_size=7*7, first_stride=2, padding='VALID' , no bias]
    maxpool(first_pool_size=3, first_pool_stride=2)
    --- Res Block ---
    we only use normal building block
    Each of the Res Layer will have # of blocks in blocks_per_layer_list
    filters = num_filters * 2 ^ {current_depth which is zero indexed}
    --- Res layer  ---
    projection_shortcut = conv(filters, 1*1, strides = stride, padding='SAME' if strides == 1 else 'VALID', no bias)
    projection_shortcut only appears in the first Res Block per layer
    forward_res = BN
                  -> ReLU
                  -> conv(filters, 3*3, strides=1, padding="same", no bias)
                  -> BN
                  -> ReLU
                  -> conv(filters, 3*3, stride=1, padding="same", no bias)
    short_cut = input
    output = forward_res + short_cut
    --- Final BN ---
    BN

    """
    # TODO: Let's say we don't need to worry about the channel things
    # if self.data_format == 'channels_first':
    #   # Convert the inputs from channels_last (NHWC) to channels_first (NCHW).
    #   # This provides a large performance boost on GPU. See
    #   # https://www.tensorflow.org/performance/performance_guide#data_formats
    #   inputs = tf.transpose(inputs, [0, 3, 1, 2])

    # building blocks differ between v1 and v2
    block_fn_map_v1 = {"normal": resnet._building_block_v1, "bottleneck": resnet._bottleneck_block_v1}
    block_fn_map_v2 = {"normal": resnet._building_block_v2, "bottleneck": resnet._bottleneck_block_v2}
    version_map = {1:block_fn_map_v1, 2:block_fn_map_v2}

    # retrieve parameters
    num_filters = parameters["num_filters"]
    first_layer_kernel_size = parameters["first_layer_kernel_size"]
    first_layer_conv_stride = parameters["first_layer_conv_stride"]
    # blocks_per_layer_list: a list of int. how many blocks per layer.
    blocks_per_layer_list = parameters["blocks_per_layer_list"]
    # block_strides_list: a list of int. The stride to use for the first convolution of the layer.
    block_strides_list = parameters["block_strides_list"]
    resnet_version = 2 if "resnet_version" not in parameters else parameters["resnet_version"]
    block_fn = version_map[resnet_version][parameters["block_fn"]]
    first_pool_size = None if "first_pool_size" not in parameters else parameters["first_pool_size"]
    first_pool_stride = None if "first_pool_stride" not in parameters else parameters["first_pool_stride"]
    data_format = "channels_last"

    with tf.variable_scope(scope_name, reuse=reuse):
        if debug:
            print("Input to cnn, size = {0}".format(inputs.get_shape()))

        # first convolution
        inputs = resnet.conv2d_fixed_padding(
            inputs=inputs, filters=num_filters, kernel_size=first_layer_kernel_size,
            strides=first_layer_conv_stride, data_format=data_format)
        inputs = tf.identity(inputs, 'initial_conv')

        if debug:
            print("After first conv = {0}".format(inputs.get_shape()))
        if count_unit is not None:
            count_unit.add_units(inputs.get_shape().as_list(), "{0}_first_conv".format(scope_name))

        # first version has BN and ReLU right after the first conv
        if resnet_version == 1:
            inputs = resnet.batch_norm(inputs, is_train_holder, data_format)
            inputs = tf.nn.relu(inputs)

        # first pooling
        if first_pool_size:
          inputs = tf.layers.max_pooling2d(
              inputs=inputs, pool_size=first_pool_size,
              strides=first_pool_stride, padding='SAME',
              data_format=data_format)
          inputs = tf.identity(inputs, 'initial_max_pool')

        if debug:
            print("After first pool = {0}".format(inputs.get_shape()))

        # multiple res layers
        for i, num_blocks in enumerate(blocks_per_layer_list):
            current_num_filters = num_filters * (growth_factor**i)
            inputs = resnet.block_layer(
              inputs=inputs, filters=current_num_filters, bottleneck=(parameters["block_fn"]=="bottleneck"),
              block_fn=block_fn, blocks=num_blocks, strides=block_strides_list[i],
              training=is_train_holder, name='block_layer{}'.format(i + 1),
              data_format=data_format)
            if debug:
              print("After {0}th res_block = {1}".format(i, inputs.get_shape()))
            # add hidden units to the counter
            # for each ResLayer, total hidden units = 5 * output if its the first layer in a block, otherwise 4 * output
            if count_unit is not None:
                for j in range(num_blocks):
                    # first layer: shortcut = 1*1 conv stride 2, x' = 3*3 conv stride 2, BN, ReLu, 3*3 conv, y = shortcut + x'
                    # otherwise: shortcut = x, x' = 3*3 conv stride 2, BN, ReLu, 3*3 conv, y = shortcut + x'
                    if j == 0:
                        count_unit.add_units(inputs.get_shape().as_list(), "{0}_ResBlock_{1}_ResLayer_{2}_1by1_conv".format(scope_name, i, j))
                    count_unit.add_units(inputs.get_shape().as_list(), "{0}_ResBlock_{1}_ResLayer_{2}_3by3_conv".format(scope_name, i, j))
                    count_unit.add_units(inputs.get_shape().as_list(), "{0}_ResBlock_{1}_ResLayer_{2}_BN_conv".format(scope_name, i, j))
                    count_unit.add_units(inputs.get_shape().as_list(), "{0}_ResBlock_{1}_ResLayer_{2}_ReLU_conv".format(scope_name, i, j))
                    count_unit.add_units(inputs.get_shape().as_list(), "{0}_ResBlock_{1}_ResLayer_{2}_3by3_conv".format(scope_name, i, j))

        # Only apply the BN and ReLU for model that does pre_activation in each
        # building/bottleneck block, eg resnet V2.
        if resnet_version == 2:
            inputs = resnet.batch_norm(inputs, is_train_holder, data_format)
            inputs = tf.nn.relu(inputs)

    # last pooling
    # The original ResNet has 7 * 7 avg pooling. This is replaced by global avg pooling.
    # inputs = tf.layers.average_pooling2d(
    #     inputs=inputs, pool_size=second_pool_size,
    #     strides=second_pool_stride, padding='VALID',
    #     data_format=data_format)
    # inputs = tf.identity(inputs, 'final_avg_pool')

    return inputs

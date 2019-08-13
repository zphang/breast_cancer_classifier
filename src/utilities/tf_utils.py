import numpy as np

import tensorflow as tf


def get_tf_variables(graph, batch_norm_key="bn"):
    param_variables = graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    bn_running_variables = []
    for variable in graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
        if batch_norm_key in variable.name \
                and "moving" in variable.name:
            bn_running_variables.append(variable)
    return param_variables + bn_running_variables


def construct_weight_assign_ops(match_dict):
    assign_list = []
    for tf_var, np_weights in match_dict.items():
        assign_list.append(tf_var.assign(np_weights))
    return assign_list


def convert_conv_torch2tf(w):
    # [C_out, C_in, H, W] => [H, W, C_in, C_out]
    return np.transpose(w, [2, 3, 1, 0])


def convert_fc_weight_torch2tf(w):
    return w.swapaxes(0, 1)

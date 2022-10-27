import tensorflow as tf
from keras import backend as K
from keras.metrics import categorical_crossentropy

if tf.keras.backend.image_data_format() == 'channels_last':
    import tensorflow as tf

lambda_rpn_regr = 1.0
lambda_rpn_class = 1.0

lambda_cls_regr = 1.0
lambda_cls_class = 1.0

epsilon = 1e-4


def rpn_loss_regr(num_anchors):
    def rpn_loss_regr_fixed_num(y_true, y_pred):
        if tf.keras.backend.image_data_format() == 'channels_first':
            x = y_true[:, 4 * num_anchors:, :, :] - y_pred
            x_abs = K.abs(x)
            x_bool = K.less_equal(x_abs, 1.0)
            return lambda_rpn_regr * K.sum(
                y_true[:, :4 * num_anchors, :, :] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(
                epsilon + y_true[:, :4 * num_anchors, :, :])
        else:
            # smooth l1 loss
            x = y_true[:, :, :, 4 * num_anchors:] - y_pred
            x_abs = K.abs(x)
            x_bool = K.cast(K.less_equal(x_abs, 1.0), tf.float32)

            return lambda_rpn_regr * K.sum(
                y_true[:, :, :, :4 * num_anchors] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(
                epsilon + y_true[:, :, :, :4 * num_anchors])

    return rpn_loss_regr_fixed_num


def rpn_loss_cls(num_anchors):
    def rpn_loss_cls_fixed_num(y_true, y_pred):
        if tf.keras.backend.image_data_format() == 'channels_last':
            # y_true： TensorShape([1, 38, 53, 18])
            # y_pred： TensorShape([1, 38, 53, 9])
            # 注意：K.binary_crossentropy返回值的shape和输入一样，不会对最后一维度求mean
            # 但是 tf.keras.losses.binary_crossentropy则会对最后一维度的交叉熵求mean
            return lambda_rpn_class * K.sum(
                y_true[:, :, :, :num_anchors] *
                K.binary_crossentropy(y_pred[:, :, :, :],
                                      y_true[:, :, :, num_anchors:])) / K.sum(epsilon + y_true[:, :, :, :num_anchors])
        else:
            return lambda_rpn_class * K.sum(
                y_true[:, :num_anchors, :, :] * K.binary_crossentropy(y_pred[:, :, :, :],
                                                                      y_true[:, num_anchors:, :, :])) / K.sum(
                epsilon + y_true[:, :num_anchors, :, :])

    return rpn_loss_cls_fixed_num


def class_loss_regr(num_classes):
    def class_loss_regr_fixed_num(y_true, y_pred):
        y_true = K.cast(y_true, 'float32')
        x = y_true[:, :, 4 * num_classes:] - y_pred
        x_abs = K.abs(x)
        x_bool = K.cast(K.less_equal(x_abs, 1.0), 'float32')
        return lambda_cls_regr * K.sum(y_true[:, :, :4 * num_classes] * (x_bool * (0.5 * x * x) + (1 - x_bool) *
                                                                         (x_abs - 0.5))) / K.sum(
            epsilon + y_true[:, :, :4 * num_classes])

    return class_loss_regr_fixed_num


def class_loss_cls(y_true, y_pred):
    return lambda_cls_class * K.mean(categorical_crossentropy(y_true[0, :, :], y_pred[0, :, :]))

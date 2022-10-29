import numpy as np
import keras.backend as K


pool_size = 7
num_rois = 32
img = np.random.random((1, 1024, 38, 50))
rois = np.asarray([[[12, 12, 19, 8]] * 16 + [[12, 12, 19, 18]] * 16])
rois = np.asarray(rois, dtype=np.int)
input_shape = K.shape(img)
nb_channels = 1024

outputs = []
for roi_idx in range(num_rois):

    x = rois[0, roi_idx, 0]
    y = rois[0, roi_idx, 1]
    w = rois[0, roi_idx, 2]
    h = rois[0, roi_idx, 3]

    row_length = w / float(pool_size)
    col_length = h / float(pool_size)

    num_pool_regions = pool_size

    # NOTE: the RoiPooling implementation differs between theano and tensorflow due to the lack of a resize op
    # in theano. The theano implementation is much less efficient and leads to long compile times

    for jy in range(num_pool_regions):
        for ix in range(num_pool_regions):
            x1 = x + ix * row_length
            x2 = x1 + row_length
            y1 = y + jy * col_length
            y2 = y1 + col_length

            x1 = K.cast(x1, 'int32')
            x2 = K.cast(x2, 'int32')
            y1 = K.cast(y1, 'int32')
            y2 = K.cast(y2, 'int32')

            x2 = x1 + K.maximum(1, x2 - x1)
            y2 = y1 + K.maximum(1, y2 - y1)

            new_shape = [input_shape[0], input_shape[1],
                         y2 - y1, x2 - x1]

            x_crop = img[:, :, y1:y2, x1:x2]
            xm = K.reshape(x_crop, new_shape)
            pooled_val = K.max(xm, axis=(2, 3))
            outputs.append(pooled_val)

final_output = K.concatenate(outputs, axis=0)
final_output = K.reshape(final_output, (1, num_rois, pool_size, pool_size, nb_channels))
final_output = K.permute_dimensions(final_output, (0, 1, 4, 2, 3))
print(final_output.shape)
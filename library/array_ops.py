import tensorflow as tf

from library.vector_math import dot


@tf.function
def assign_index(arr, index, element):
    arr_shape = tf.shape(arr)

    pos_mask = tf.eye(arr_shape[0])[index]
    pos_mask = tf.transpose(tf.expand_dims(pos_mask, 0))
    neg_mask = 1 - pos_mask

    tiled_element = tf.reshape(tf.tile(element, [arr_shape[0]]), arr_shape)

    arr = arr * neg_mask + tiled_element * pos_mask

    return arr


@tf.function
def assign_index_vectored(arr, index, element):
    arr_shape = tf.shape(arr)

    pos_mask = tf.transpose(tf.expand_dims(index, 0))
    neg_mask = 1 - pos_mask

    tiled_element = tf.reshape(tf.tile(element, [arr_shape[0]]), arr_shape)

    arr = arr * neg_mask + tiled_element * pos_mask

    return arr


@tf.function
def naive_lookup(arr, index):
    index = tf.round(index)
    index = tf.cast(index, tf.int32)
    result = arr[index]
    return result


@tf.function
def interp_factor(index):
    t1 = tf.math.floor(index)
    t2 = tf.math.ceil(index)

    t = tf.math.divide_no_nan((index - t1), (t2 - t1))

    i1 = tf.cast(t1, tf.int32)
    i2 = tf.cast(t2, tf.int32)

    return t, i1, i2


@tf.function
def linear_lookup(arr, index):
    t, i1, i2 = interp_factor(index)

    # Linear interpolation
    result = t * arr[i1] + (1 - t) * arr[i2]

    return result


@tf.function
def superposition_lookup_vectored(arr, indices):
    if tf.rank(arr) == 1:
        arr = tf.expand_dims(arr, -1)
    indices = tf.expand_dims(indices, -1)
    result = arr * indices
    return tf.reduce_sum(result, axis=0)


@tf.function
@tf.custom_gradient
def asymmetrical_vectored_lookup(v, k):
    '''
    If chained, we assume that closer elements are in the same dimension. (TODO: Clarify)
    '''
    k_shape = tf.shape(k)
    v_shape = tf.shape(v)

    tf.debugging.assert_equal(k_shape, v_shape)

    # Pick the value at the most likely index, non-differentiably
    flat_k = tf.reshape(k, [-1, k_shape[-1]])
    collapsed_k = tf.argmax(flat_k, -1)
    collapsed_k = tf.one_hot(collapsed_k, k_shape[-1])
    unflat_k = tf.reshape(collapsed_k, k_shape)
    forward_result = dot(v, unflat_k)

    def grad(upstream_grads):
        # Estimate the target scalar which we want to look up
        target = forward_result - upstream_grads
        target = tf.expand_dims(target, -1)

        # Find the index of element in the array which is closest to target
        diff_vector = tf.math.squared_difference(v, target)
        d_idx = tf.argmin(diff_vector, axis=-1)

        # Create a vector which is 1 everywhere except the idx
        # of the target, where it is -1
        ones = tf.ones(k_shape)
        eyes = tf.one_hot([d_idx], k_shape[-1])[0]
        k_grad = -(2 * eyes - ones)

        # d/dv (v . k) = k
        v_grad = k

        upstream_grads = tf.expand_dims(upstream_grads, -1)

        # 1. The k_grad should dictate the direction of the vector. So, upstream grad is always positive
        # 2. We want it to scale to zero as it gets closer to target. So we clip it between 0 and 1.
        # 3. If there is an exact match in the vector, then we dont send the gradients downstream for other entries. (disabled)
        min_clipped_abs_grad = tf.abs(upstream_grads)
        min_clipped_abs_grad = tf.clip_by_value(min_clipped_abs_grad, 0, 1)
        # min_clipped_abs_grad = tf.reduce_min(min_clipped_abs_grad)

        return upstream_grads * v_grad, min_clipped_abs_grad * k_grad

    return forward_result, grad


@tf.function
def bandwidthify(index, bandwidth):
    t, i1, i2 = interp_factor(index)

    # Prevent array out of bounds
    i1 = tf.clip_by_value(i1, 0, bandwidth - 1)
    i2 = tf.clip_by_value(i2, 0, bandwidth - 1)
    t = tf.clip_by_value(t, 0, 1)

    # Linear interpolation
    eye = tf.eye(bandwidth)
    result = t * eye[i1] + (1 - t) * eye[i2]

    return result


@tf.function
def bulk_bandwidthify(indices, bandwidth):
    num_indices = tf.shape(indices)[0]

    indices = tf.unstack(indices)
    result = tf.zeros((num_indices, bandwidth), dtype=tf.float32)
    result = tf.unstack(result)

    for i, index in enumerate(indices):
        b_index = bandwidthify(index, bandwidth)
        result[i] += b_index

    result = tf.stack(result)
    return result


@tf.function
def superposition_lookup(arr, index):
    bandwidth = tf.shape(arr)[0]
    vectored_index = bandwidthify(index, bandwidth)
    result = superposition_lookup_vectored(arr, vectored_index)

    return result


@tf.function
def residual_lookup(arr, index):
    i = tf.round(index)
    residue = index - i
    i = tf.cast(i, tf.int32)

    result = arr[i]

    return result, residue


@tf.function
def match_shapes(x, y):
    # Find which one needs to be broadcasted
    low, high = (y, x) if tf.rank(x) > tf.rank(y) else (x, y)
    l_rank, l_shape = tf.rank(low), tf.shape(low)
    h_rank, h_shape = tf.rank(high), tf.shape(high)

    # Find the difference in ranks
    common_shape = h_shape[:l_rank]
    tf.debugging.assert_equal(common_shape, l_shape, "No common shape to broadcast")
    padding = tf.ones(h_rank - l_rank, dtype=tf.int32)

    # Pad the difference with ones and reshape
    new_shape = tf.concat((common_shape, padding), axis=0)
    low = tf.reshape(low, new_shape)

    return high, low


@tf.function
def broadcast_multiply(x, y):
    x, y = match_shapes(x, y)
    return x * y


@tf.function
def tensor_lookup_2d(arr, x_index, y_index):
    # Calculate outer product
    mask = tf.tensordot(x_index, y_index, axes=0)

    # Broadcast the mask to match dimensions with arr
    masked_arr = broadcast_multiply(mask, arr)

    # Reduce max to extract the cell
    element = tf.math.reduce_max(masked_arr, axis=[0, 1])
    return element


@tf.function
def tensor_write_2d(arr, element, x_index, y_index):
    arr_shape = tf.shape(arr)
    mask = tf.tensordot(x_index, y_index, axes=0)

    # Broadcast the mask to match dimensions with arr
    _, mask = match_shapes(arr, mask)

    element = tf.reshape(element, [1, 1, -1])
    element = tf.tile(element, [arr_shape[0], arr_shape[1], 1])

    result = (1.0 - mask) * arr + mask * element

    return result

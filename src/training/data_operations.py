import tensorflow as tf
import numpy as np


def simple_scipy_norm_x(x):
    x = x.tocoo()
    idx   = np.stack([x.row, x.col], axis=1).astype(np.int64)
              # counts as float32 (required by stateless_binomial)
    shape = np.array(x.shape, dtype=np.int64)

    vals  = tf.constant(x.data.astype(np.float16))
    vals = tf.math.log1p(vals)

    sp = tf.sparse.SparseTensor(idx, vals, shape)
    sp = tf.sparse.reorder(sp)
    
    x_tf_dense = tf.sparse.to_dense(sp)
    return x_tf_dense


def pretrain_batch_from_x_tf(x_batch):
    # scipy sparse -> TF SparseTensor (COO)
    x = x_batch.tocoo()
    idx   = np.stack([x.row, x.col], axis=1).astype(np.int64)
    vals  = x.data.astype(np.float32)          # counts as float32 (required by stateless_binomial)
    shape = np.array(x.shape, dtype=np.int64)

    sp = tf.sparse.SparseTensor(idx, vals, shape)
    sp = tf.sparse.reorder(sp)

    B = tf.cast(sp.dense_shape[0], tf.int32)

    # dilution and keep-prob
    dilution = tf.random.uniform([B, 1], minval=0.1, maxval=1.0, dtype=tf.float32)
    p = 1.0 - tf.squeeze(dilution, axis=1)  # [B]

    # binomial thinning ONLY on nnz
    row = tf.cast(sp.indices[:, 0], tf.int32)
    p_vals = tf.gather(p, row)

    seed = tf.random.uniform([2], maxval=2**31 - 1, dtype=tf.int32)
    diluted_i = tf.random.stateless_binomial(
        shape=tf.shape(sp.values),
        seed=seed,
        counts=sp.values,          # float32
        probs=p_vals,              # float32
        output_dtype=tf.int32
    )
    diluted = tf.cast(diluted_i, tf.float32)

    # x_diluted sparse (drop zeros)
    keep = diluted > 0.0
    x_dil_sp = tf.sparse.reorder(tf.sparse.SparseTensor(
        indices=tf.boolean_mask(sp.indices, keep),
        values=tf.boolean_mask(diluted, keep),
        dense_shape=sp.dense_shape
    ))

    # complement on original support
    comp_vals = sp.values - diluted
    comp_sp = tf.sparse.SparseTensor(sp.indices, comp_vals, sp.dense_shape)

    # y = complement / row-sum (sparse -> dense)
    row2 = tf.cast(comp_sp.indices[:, 0], tf.int32)
    y_sum = tf.math.unsorted_segment_sum(comp_sp.values, row2, num_segments=B)  # [B]
    valid_y = y_sum > 0.0
    y_vals = tf.math.divide_no_nan(comp_sp.values, tf.gather(y_sum, row2))
    y_sp = tf.sparse.reorder(tf.sparse.SparseTensor(comp_sp.indices, y_vals, comp_sp.dense_shape))

    # log1p norm
    x_dil_sp = tf.SparseTensor(
        indices=x_dil_sp.indices,
        values=tf.math.log1p(x_dil_sp.values),
        dense_shape=x_dil_sp.dense_shape
    )

    x_diluted = tf.sparse.to_dense(x_dil_sp)
    y = tf.sparse.to_dense(y_sp)

    # Replace zero-target rows with uniform labels.
    G = tf.cast(tf.shape(y)[1], y.dtype)
    uniform_value = tf.cast(1.0, y.dtype) / G
    y = y + tf.cast(~valid_y[:, None], y.dtype) * uniform_value

    return x_diluted, y

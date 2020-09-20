import tensorflow as tf
import numpy as np


def pretty_print_guess_tensor(const_guess, operand_guess, operator_guess):
    # TODO: const_guess

    s = []

    for t in operand_guess:
        s += [f'x_{tf.argmax(t)}']

    operator_lookup = ['+', '-', '*', '/']
    result = s[::]
    for i, op_one_hot in enumerate(operator_guess):
        operators = tf.argmax(op_one_hot, axis=-1)
        left = result[::2]
        right = (result[1:] + result[:1])[::2]
        ops = operators[:len(left)]
        result = []
        for l, op, r in zip(left, ops, right):
            result += [f'({l} {operator_lookup[op]} {r})']

    return ' '.join(result)

# NUM_LEAVES = 8
# NUM_OPERATORS = 4
# v1 = tf.range(NUM_LEAVES)
# v2 = tf.range(NUM_OPERATORS)

# cgv = tf.one_hot(v1 // 2, NUM_LEAVES//2, dtype=tf.float32)
# const_guess = tf.concat([cgv, cgv],axis=1)
# operand_guess = tf.one_hot(v1, NUM_LEAVES, dtype=tf.float32)
# ogv = tf.expand_dims(tf.one_hot(v2, NUM_OPERATORS, dtype=tf.float32), axis=0)
# operator_guess = tf.concat([ogv,ogv,ogv], axis=0)

# pretty_print_guess_tensor(const_guess, operand_guess, operator_guess)


def eqn_to_block_tensor(fn, xs, tree_depth=3, num_op_types=4):
    NUM_LEAVES = 1 << tree_depth
    NUM_VARS = len(xs[0])

    target = []
    values = []

    for x in xs:
        values.append(x)
        target.append(fn(*x))

    operators = []
    for _ in range(tree_depth):
        operators.append(tf.one_hot(np.arange(NUM_LEAVES // 2) % num_op_types, num_op_types))
    operators = tf.stack(operators)

    values = tf.constant([values], dtype=tf.float32)  # [1, datapoint_x, data_point_dim]
    target = tf.expand_dims(tf.constant([target], dtype=tf.float32), -1)  # [1, datapoint_y, 1]
    operand_guess = tf.one_hot(np.arange(NUM_LEAVES) % NUM_VARS, NUM_VARS)
    operand_guess = tf.expand_dims(operand_guess, 0)  # [1, NUM_LEAVES, NUM_VARS]
    operator_guess = tf.expand_dims(operators, 0)  # [1, tree_level, op_pair, op_type_one_hot]
    const_guess = tf.constant([1.])  # TODO

    return const_guess, operand_guess, operator_guess, values, target

# fn = lambda x, y: x + y
# xs = np.array([[1,2], [2,3], [4,5]], dtype=np.float32)

# eqn_to_block_tensor(fn, xs)

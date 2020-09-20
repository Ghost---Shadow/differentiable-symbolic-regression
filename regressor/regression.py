import tensorflow as tf

from library.array_ops import asymmetrical_vectored_lookup


@tf.function
def resolve_values(const_guess, values, operand_guess):
    # TODO: const_guess

    # tf.debugging.assert_rank(const_guess, 3)
    tf.debugging.assert_rank(values, 3)  # [outer_batch, inner_batch, VALUES_SIZE]
    tf.debugging.assert_rank(operand_guess, 3)  # [outer_batch, LEAVES_SIZE, VALUES_SIZE]

    values_shape = tf.shape(values)
    operands_shape = tf.shape(operand_guess)

    outer_batch, inner_batch, VALUES_SIZE = [values_shape[0], values_shape[1], values_shape[2]]
    outer_batch, LEAVES_SIZE, VALUES_SIZE = [operands_shape[0], operands_shape[1], operands_shape[2]]

    # Broadcast the operand choices
    operand_guess = tf.expand_dims(operand_guess, axis=1)
    operand_guess = tf.tile(operand_guess, [1, inner_batch, 1, 1])  # [outer_batch, inner_batch, LEAVES_SIZE, VALUES_SIZE]

    # Broadcast the values
    values = tf.expand_dims(values, axis=2)
    values = tf.tile(values, [1, 1, LEAVES_SIZE, 1])  # [outer_batch, inner_batch, LEAVES_SIZE, VALUES_SIZE]

    # Dot product
    # operand_guess = to_prob_dist_all(operand_guess)
    # result = dot(values, operand_guess) # [outer_batch, inner_batch, LEAVES_SIZE]
    result = asymmetrical_vectored_lookup(values, operand_guess)

    return result

# v1 = tf.range(NUM_LEAVES)
# cgv = tf.one_hot(v1 // 2, NUM_LEAVES//2, dtype=tf.float32)
# const_guess = tf.concat([cgv, cgv],axis=1)

# operand_guess = tf.Variable([
#     [[1,0,0], [0,1,0], [0,0,1]],
#     [[1,1,0], [0,0,1], [1,1,1]],
# ],dtype=tf.float32)

# values = tf.constant([
#     [[1,2,3],
#     [4,5,6]],
#     [[7,8,9],
#     [11,22,33]],
# ], dtype=tf.float32)

# with tf.GradientTape() as tape:
#     result = resolve_values(const_guess, values, operand_guess)

# grads = tape.gradient(result, operand_guess)

# tf.print(tf.round(result))
# tf.print(grads)


@tf.function
def operate(operands, operators):
    tf.debugging.assert_rank(operands, 3, 'Expected operands to be rank 3. [equation, datapoint, level]')
    tf.debugging.assert_rank(operators, 3, 'Expected operators to be rank 3. [equation, op_pair, op_type_one_hot]')

    opd_shape = tf.shape(operands)
    tf.debugging.assert_equal(opd_shape[-1] % 2, 0, 'Shape of axis -1 of operands must be div by 2')

    left = operands[:, :, ::2]
    right = tf.roll(operands, shift=-1, axis=-1)[:, :, ::2]

    r_add = left + right
    r_sub = left - right
    r_mul = left * right
    r_div = tf.math.divide_no_nan(left, right)

    r = tf.stack([r_add, r_sub, r_mul, r_div], axis=-1)  # [equation, datapoint, op_pair, op_type_one_hot]

    opt = tf.expand_dims(operators, axis=1)
    opt = tf.tile(opt, [1, opd_shape[1], 1, 1])  # [equation, datapoint, op_pair, op_type_one_hot]

    # operators = tf.nn.softmax(operators, axis=-1)
    # operators = to_prob_dist_all(operators)

    result = asymmetrical_vectored_lookup(r, opt)  # [equation, datapoint, op_pair]

    return result

# operands = tf.constant([
#     [[1, 2, 3, 4], [5, 6, 7, 8]],
#     [[11, 22, 33, 44], [55, 66, 77, 88]],
# ],dtype=tf.float32)

# add = [1,0,0,0]
# sub = [0,1,0,0]
# mul = [0,0,1,0]
# div = [0,0,0,1]

# operators = tf.constant([
#     [add, sub],
#     [mul, div]
# ],dtype=tf.float32)

# operands = tf.Variable(operands)
# operators = tf.Variable(operators)
# target = tf.constant([
#     [[1 + 2, 3 - 4], [5 + 6, 7 - 8]],
#     [[11 * 22, 33 / 44], [55 * 66, 77 / 88]],
# ],dtype=tf.float32)

# with tf.GradientTape() as tape:
#     result = operate(operands, operators)
#     loss = tf.nn.l2_loss(result - target)

# tf.print(result)
# tape.gradient(loss, operators)


def eager_process_block(operands, operators_arr):
    acc = operands

    levels = tf.shape(operators_arr)[1]

    for level in tf.range(levels):
        num_operands = tf.shape(acc)[-1]
        op = operators_arr[:, level, :num_operands // 2, :]
        acc = operate(acc, op)

    return acc

# fn1 = lambda x, y: x + y
# fn2 = lambda x, y: x - y
# xs1 = np.array([[1,2], [2,3], [4,5]], dtype=np.float32)
# xs2 = np.array([[1,2], [2,3], [4,5]], dtype=np.float32)

# cg1, og1, ot1, v1, t1 = eqn_to_block_tensor(fn1, xs1)
# cg2, og2, ot2, v2, t2 = eqn_to_block_tensor(fn2, xs2)

# const_guess = tf.concat([cg1, cg2], axis=0)
# values = tf.concat([v1, v2], axis=0)
# operand_guess = tf.concat([og1, og2], axis=0)
# operator_guess = tf.concat([ot1, ot2], axis=0)
# target = tf.concat([t1, t2], axis=0)

# operands = resolve_values(const_guess, values, operand_guess)

# with tf.GradientTape(persistent=True) as tape:
#     tape.watch(operands)
#     tape.watch(operator_guess)
#     result = eager_process_block(operands, operator_guess)

# print_idx = 0
# tf.print(pretty_print_guess_tensor(const_guess[print_idx], operand_guess[print_idx], operator_guess[print_idx]))
# x_0, x_1 = 1, 2
# tf.print((((x_0 + x_1) + (x_0 - x_1)) + ((x_0 * x_1) - (x_0 / x_1))))
# tf.print(result)
# tf.print(tape.gradient(result, operator_guess))


@tf.function
def unrolled_process_block(operands, operators_arr, levels):
    acc = operands
    num_operands = 2 ** levels

    for level in range(levels):
        num_operands //= 2
        op = operators_arr[:, level, :num_operands, :]
        acc = operate(acc, op)

    return acc

# fn1 = lambda x, y: x + y
# fn2 = lambda x, y: x - y
# xs1 = np.array([[1,2], [2,3], [4,5]], dtype=np.float32)
# xs2 = np.array([[1,2], [2,3], [4,5]], dtype=np.float32)

# cg1, og1, ot1, v1, t1 = eqn_to_block_tensor(fn1, xs1)
# cg2, og2, ot2, v2, t2 = eqn_to_block_tensor(fn2, xs2)

# const_guess = tf.concat([cg1, cg2], axis=0)
# values = tf.concat([v1, v2], axis=0)
# operand_guess = tf.concat([og1, og2], axis=0)
# operator_guess = tf.concat([ot1, ot2], axis=0)
# target = tf.concat([t1, t2], axis=0)

# operands = resolve_values(const_guess, values, operand_guess)

# with tf.GradientTape(persistent=True) as tape:
#     tape.watch(operands)
#     tape.watch(operator_guess)
#     result = unrolled_process_block(operands, operator_guess, 3)

# print_idx = 0
# tf.print(pretty_print_guess_tensor(const_guess[print_idx], operand_guess[print_idx], operator_guess[print_idx]))
# x_0, x_1 = 1, 2
# tf.print((((x_0 + x_1) + (x_0 - x_1)) + ((x_0 * x_1) - (x_0 / x_1))))
# tf.print(result)
# tf.print(tape.gradient(result, operator_guess))

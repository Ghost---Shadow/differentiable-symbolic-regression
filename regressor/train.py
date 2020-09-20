import tensorflow as tf

from library.statistical_math import to_prob_dist_all
from regressor.regression import unrolled_process_block, resolve_values


def bind_opt_train_step(opt, levels):
    @tf.function
    def train_step(const_guess, operand_guess, operator_guess, values, target):
        with tf.GradientTape() as tape:
            cg, opg, otg = const_guess, operand_guess, operator_guess

            # cg = to_prob_dist_all(cg)
            # opg = to_prob_dist_all(opg)
            # otg = to_prob_dist_all(otg)

            # cg_entropy = 0.0 # TODO
            # opg_entropy = tf.reduce_sum(entropy(opg))
            # otg_entropy = tf.reduce_sum(entropy(otg))

            operands = resolve_values(cg, values, opg)
            result = unrolled_process_block(operands, otg, levels)

            target_loss = tf.nn.l2_loss(result - target)

            loss = target_loss

            # if target_loss < 1:
            #     loss += entropy_weight * (opg_entropy + otg_entropy)

        variables = [operand_guess, operator_guess]
        grads = tape.gradient(loss, variables)
        # grads = [tf.clip_by_norm(g, 100.0) for g in grads]
        opt.apply_gradients(zip(grads, variables))

        # const_guess.assign(to_prob_dist_all(const_guess))
        operand_guess.assign(to_prob_dist_all(operand_guess))
        operator_guess.assign(to_prob_dist_all(operator_guess))

        return loss, result

    return train_step

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
# levels = 3

# operand_guess = tf.Variable(operand_guess)
# operator_guess = tf.Variable(operator_guess)

# lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
#     1e-1,
#     decay_steps=100,
#     decay_rate=1e-1,
#     staircase=True)
# opt = tf.keras.optimizers.Adam(lr_schedule)
# train_step = bind_opt_train_step(opt, levels)

# with tf.GradientTape(persistent=True) as tape:
#     # tape.watch(const_guess)
#     tape.watch(operand_guess)
#     tape.watch(operator_guess)

#     loss, result = train_step(const_guess, operand_guess, operator_guess, values, target)

# tf.print(result)
# tf.print(loss)
# tf.print(tape.gradient(loss, operand_guess))
# tf.print(tape.gradient(loss, operator_guess))

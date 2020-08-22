import tensorflow as tf

from src.grammar.ast import generate
from src.grammar.utils import tokens_pretty_print


def train_step_wrapped():
    opt = tf.keras.optimizers.Adam(1e-1)

    @tf.function
    def train_step(grammar, productions, stack_shape, S, is_phi_fn, output, print_steps=False):
        zero_stack = tf.one_hot([0] * stack_shape[0], stack_shape[1], dtype=tf.float32)
        phi = tf.one_hot([0], stack_shape[1], dtype=tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(productions)
    #         # Soften the grammar
    #         gs, go = grammar
    #         sgs, sgo = tf.nn.softmax(gs), tf.nn.softmax(go)
    #         soft_g = (sgs, sgo)

    #         # Soften the productions
    #         soft_p = tf.nn.softmax(productions,axis=-1)

    #         # Soften S
    #         soft_s = tf.nn.softmax(S)

    #         soft_phi = tf.nn.softmax(phi, axis=-1)

    #         output_, stack_ = generate(soft_g, soft_p, stack_shape, soft_s, soft_phi, is_phi_fn, True)

    #         loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(output, output_))
    #         loss += tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(zero_stack, stack_))

            output_, stack_ = generate(grammar, productions, stack_shape, S, phi, is_phi_fn, print_steps)
            loss = tf.nn.l2_loss(output - output_) + tf.nn.l2_loss(zero_stack - stack_)

        grads = tape.gradient(loss, productions)
        opt.apply_gradients(zip([grads], [productions]))

        return loss, output_, stack_

    return train_step


def optimize(grammar, productions, stack_shape, d_S, is_phi, output, steps):
    train_step = train_step_wrapped()

    for i in range(steps):
        loss, output_, stack_ = train_step(grammar, productions, stack_shape, d_S, is_phi, output)
        if i % 10 == 0:
            p_output = tokens_pretty_print(output_)
            p_stack = tokens_pretty_print(stack_)

            tf.print(loss, p_output, p_stack, tf.argmax(productions, axis=-1))

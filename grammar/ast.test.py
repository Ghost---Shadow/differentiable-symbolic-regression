import tensorflow as tf

from src.library.stacks import new_stack
from src.grammar.utils import is_phi, safe_push, tokens_pretty_print
from src.grammar.ast import production_step, generate
from src.grammar.transition_matrix import grammar, S, STACK_SIZE, PRODUCTION_DIM, TOKEN_DIM


def test_production_step():
    stack = new_stack(((STACK_SIZE, TOKEN_DIM)))
    output = new_stack(((STACK_SIZE, TOKEN_DIM)))

    stack = safe_push(stack, tf.constant(S, dtype=tf.float32), is_phi)
    production = tf.one_hot(2, PRODUCTION_DIM)
    phi = tf.one_hot(0, TOKEN_DIM, dtype=tf.float32)

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(grammar)
        tape.watch(production)
        tape.watch(stack)
        tape.watch(output)

        new_s, new_o = production_step(grammar, production, stack, output, phi, is_phi)

    tf.print(tokens_pretty_print(new_s[0]))
    tf.print(tape.gradient(new_o, output))
    tf.print(tape.gradient(new_s, stack))
    tf.print(tape.gradient(new_s[0], grammar[0]).shape)
    tf.print(tape.gradient(new_s[1], grammar[0]).shape)
    tf.print(tape.gradient(new_o[0], grammar[1]).shape)
    tf.print(tape.gradient(new_o[1], grammar[1]).shape)
    tf.print(tape.gradient(new_s, production))


def test_generate():
    tf.config.experimental_run_functions_eagerly(True)

    productions = tf.one_hot([2, 3, 0, 1, 0], PRODUCTION_DIM)

    stack_shape = (STACK_SIZE, TOKEN_DIM)
    d_S = tf.constant(S, dtype=tf.float32)
    d_phi = tf.constant(tf.one_hot([0], TOKEN_DIM))

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(productions)
        output, final_stack = generate(grammar, productions, stack_shape, d_S, d_phi, is_phi, True)

    tf.config.experimental_run_functions_eagerly(False)
    tf.print('Final result:')
    tf.print(tokens_pretty_print(output))
    tf.print('-'*80)
    tf.print('Final stack:')
    tf.print(tokens_pretty_print(final_stack))
    tf.print('-'*80)
    tf.print(tape.gradient(output, productions))

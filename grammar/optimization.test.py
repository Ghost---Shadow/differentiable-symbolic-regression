import tensorflow as tf

from src.grammar.optimization import train_step_wrapped, optimize
from src.grammar.utils import encode_to_tokens, tokens_pretty_print, is_phi
from src.grammar.transition_matrix import grammar, S, STACK_SIZE, PRODUCTION_DIM, TOKEN_DIM


def test_train_step():
    train_step = train_step_wrapped()
    # MAX_PRODUCTIONS = 5
    # productions = tf.Variable(tf.one_hot([0] * MAX_PRODUCTIONS, PRODUCTION_DIM), dtype=tf.float32)
    productions = tf.Variable(tf.one_hot([2, 3, 0, 0, 0], PRODUCTION_DIM, dtype=tf.float32))
    # productions = tf.Variable(tf.one_hot([2, 0, 0, 0, 0], PRODUCTION_DIM), dtype=tf.float32)
    stack_shape = (STACK_SIZE, TOKEN_DIM)
    d_S = tf.constant(S, dtype=tf.float32)
    output = encode_to_tokens('x + x', TOKEN_DIM, STACK_SIZE)

    tf.config.experimental_run_functions_eagerly(True)
    loss, output_, stack_ = train_step(grammar, productions, stack_shape, d_S, is_phi, output, True)
    tf.config.experimental_run_functions_eagerly(False)
    tf.print(loss)
    tf.print(tokens_pretty_print(output_), tokens_pretty_print(output))
    tf.print(tokens_pretty_print(stack_))


def test_optimize():
    productions = tf.Variable(tf.one_hot([2, 3, 0, 0, 0], PRODUCTION_DIM), dtype=tf.float32)
    output = encode_to_tokens('x + x', TOKEN_DIM, STACK_SIZE)
    steps = 100
    stack_shape = (STACK_SIZE, TOKEN_DIM)
    d_S = tf.constant(S, dtype=tf.float32)

    optimize(grammar, productions, stack_shape, d_S, is_phi, output, steps)

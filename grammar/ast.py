import tensorflow as tf

from src.library.array_ops import tensor_lookup_2d
from src.library.stacks import new_stack_from_buffer, stack_peek
from src.grammar.utils import is_phi, \
    safe_push, \
    pop_and_purge, \
    tokens_pretty_print


@tf.function
def production_step(grammar, production, stack, output, phi, is_phi_fn):
    tf.debugging.assert_rank(grammar[0], 4)
    tf.debugging.assert_rank(grammar[1], 4)
    tf.debugging.assert_rank(production, 1)
    tf.debugging.assert_rank(stack[0], 2)
    tf.debugging.assert_rank(output[0], 2)

    G_s, G_o = grammar

    # Save the shapes
    # stack_0_shape = tf.shape(stack[0])
    # stack_1_shape = tf.shape(stack[1])
    # output_0_shape = tf.shape(output[0])
    # output_1_shape = tf.shape(output[1])

    # Get next token from stack
    stack, stack_top_token = pop_and_purge(stack, phi)

    # Push tokens back onto the stack
    tokens_to_push = tensor_lookup_2d(G_s, production, stack_top_token)
    for token in tf.reverse(tokens_to_push, axis=[0]):
        stack = safe_push(stack, token, is_phi_fn)

    # Push tokens to output
    tokens_to_push = tensor_lookup_2d(G_o, production, stack_top_token)
    for token in tokens_to_push:
        output = safe_push(output, token, is_phi_fn)

    return stack, output


def dump_step_info(grammar, production, stack, output):
    gs, go = grammar
    top = stack_peek(stack)
    tf.print('p\t', tf.argmax(production))
    i = tf.argmax(production)
    j = tf.argmax(top)
    tf.print('G_s\t', tokens_pretty_print(gs[i][j]), (i, j))
    tf.print('G_o\t', tokens_pretty_print(go[i][j]), (i, j))
    tf.print('S_i+1\t', tokens_pretty_print(stack[0]), tf.argmax(stack[1]))
    tf.print('O_i+1\t', tokens_pretty_print(output[0]), tf.argmax(output[1]))
    tf.print('-'*80)


@tf.function
def generate(grammar, productions, stack_shape, S, phi, is_phi_fn, print_steps=False):
    # Reserve space for stack and output
    stack_buffer = tf.tile(phi, (stack_shape[0], 1))
    stack = new_stack_from_buffer(stack_buffer)
    output_buffer = tf.tile(phi, (stack_shape[0], 1))
    output = new_stack_from_buffer(output_buffer)

    # Push S to top of stack
    stack = safe_push(stack, S, is_phi)

    productions = tf.unstack(productions)

    for production in productions:
        stack, output = production_step(grammar, production, stack, output, phi[0], is_phi_fn)

        if print_steps:
            dump_step_info(grammar, production, stack, output)

    return tf.reverse(output[0], axis=[0]), stack[0]

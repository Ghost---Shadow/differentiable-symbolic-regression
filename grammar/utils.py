import tensorflow as tf

from library.stacks import stack_push, stack_pop, stack_peek, new_stack, new_stack_from_buffer


@tf.function
def is_phi(element):
    tf.debugging.assert_rank(element, 1)

    elem_dim = tf.shape(element)[0]
    phi = tf.one_hot(0, elem_dim)

    element = tf.math.l2_normalize(element)
    t = tf.tensordot(element, phi, axes=1)

    return t


@tf.function
def safe_push(stack, element, is_phi_fn):
    tf.debugging.assert_rank_at_least(stack[0], 2)
    tf.debugging.assert_rank(stack[1], 1)
    tf.debugging.assert_equal(tf.shape(stack[0])[1:], tf.shape(element))
    tf.debugging.assert_equal(tf.rank(stack[0]) - 1, tf.rank(element))

    t = is_phi_fn(element)

    old_buffer, old_index = stack
    new_buffer, new_index = stack_push(stack, element)

    buffer = t * old_buffer + (1 - t) * new_buffer
    index = t * old_index + (1 - t) * new_index

    # tf.print(tokens_pretty_print(old_buffer))
    # tf.print(tokens_pretty_print(new_buffer))
    # tf.print(tokens_pretty_print(buffer))
    # tf.print('-'*80)

    # Hack to tell tensorflow that the shape has not changed
    # TODO: Why does this hack work?
    buffer = tf.reshape(buffer, tf.shape(old_buffer))
    index = tf.reshape(index, tf.shape(old_index))

    new_stack = (buffer, index)

    return new_stack


@tf.function
def pop_and_purge(stack, phi):
    # stack_len = tf.shape(stack[0])[1]
    stack, element = stack_pop(stack)
    stack = stack_push(stack, phi)
    stack, _ = stack_pop(stack)

    return stack, element


def tokens_pretty_print(tokens):
    tokens = tf.argmax(tokens, axis=1)
    lookup = ['_', 'S', 'O', 'T', 'x', '+']

    result = ''

    for token in tokens:
        result += f'{lookup[token]} '

    return result


def encode_to_tokens(s, token_dim, total_length):
    lookup = ['_', 'S', 'O', 'T', 'x', '+']
    arr = []
    for t in s.split(' '):
        arr.append(lookup.index(t))

    phi = lookup.index('_')
    arr = ([phi] * (total_length - len(arr))) + arr

    return tf.one_hot(arr, token_dim)

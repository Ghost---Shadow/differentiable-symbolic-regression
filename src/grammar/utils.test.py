import tensorflow as tf

from src.grammar.utils import is_phi, \
    safe_push, \
    pop_and_purge, \
    tokens_pretty_print, \
    encode_to_tokens

from src.library.stacks import new_stack, new_stack_from_buffer


def test_is_phi():
    test1 = tf.Variable([1, 0, 0], dtype=tf.float32)
    test2 = tf.Variable([0, 1, 0], dtype=tf.float32)
    test3 = tf.Variable([.5, .5, 0], dtype=tf.float32)

    with tf.GradientTape(persistent=True) as tape:
        result1 = is_phi(test1)
        result2 = is_phi(test2)
        result3 = is_phi(test3)

    tf.print(result1, tape.gradient(result1, test1))
    tf.print(result2, tape.gradient(result2, test2))
    tf.print(result3, tape.gradient(result3, test3))


def test_safe_push():
    stack = new_stack((3, 3), True)
    original_stack = stack

    element1 = tf.Variable([0, 1, 0], dtype=tf.float32)
    element2 = tf.Variable([0.5, 0.5, 0], dtype=tf.float32)
    element3 = tf.Variable([0, 0, 1], dtype=tf.float32)
    element4 = tf.Variable([0, 1, 0], dtype=tf.float32)

    with tf.GradientTape(persistent=True) as tape:
        stack = safe_push(stack, element1, is_phi)
        stack = safe_push(stack, element2, is_phi)
        stack = safe_push(stack, element3, is_phi)
        stack = safe_push(stack, element4, is_phi)

    tf.print(stack[0])
    tf.print(tf.round(stack[0]))
    tf.print(stack[1])
    tf.print(tf.round(stack[1]))
    tf.print(tape.gradient(stack[0], element3))
    tf.print(tape.gradient(stack, original_stack))


def test_safe_push2():
    stack = new_stack((3, 3), True)

    element1 = tf.Variable([0, 1, 0], dtype=tf.float32)
    element2 = tf.Variable([1, 0, 0], dtype=tf.float32)
    element3 = tf.Variable([0, 0, 1], dtype=tf.float32)
    element4 = tf.Variable([0, 1, 0], dtype=tf.float32)

    original_stack = stack

    with tf.GradientTape(persistent=True) as tape:
        stack = safe_push(stack, element1, is_phi)
        stack = safe_push(stack, element2, is_phi)
        stack = safe_push(stack, element3, is_phi)
        stack = safe_push(stack, element4, is_phi)

    tf.print(stack[0])
    tf.print(stack[1])
    tf.print(tape.gradient(stack[0], element3))
    tf.print(tape.gradient(stack, original_stack))


def test_pop_and_purge():
    stack = new_stack_from_buffer(tf.ones((3, 3), dtype=tf.float32))
    old_stack = stack
    phi = tf.one_hot(0, 3, dtype=tf.float32)

    with tf.GradientTape(persistent=True) as tape:
        stack, element = pop_and_purge(stack, phi)

    tf.print(stack)
    tf.print(element)
    tf.print(tape.gradient(stack, old_stack))


def test_tokens_pretty_print():
    TOKEN_DIM = 6
    tokens = tf.transpose(tf.one_hot([0, 1, 2, 3, 4, 5], TOKEN_DIM, dtype=tf.float32))
    tokens_pretty_print(tokens)


def test_encode_to_tokens():
    TOKEN_DIM = 6
    encode_to_tokens('x + x +', TOKEN_DIM, 5)

import unittest
from unittest import TestCase

import tensorflow as tf
import numpy.testing as npt

from .array_ops import tensor_lookup_2d, asymmetrical_vectored_lookup


# python -m unittest library.array_ops_test.AsymmetricalVectoredLookup
class AsymmetricalVectoredLookup(TestCase):
    # python -m unittest library.array_ops_test.AsymmetricalVectoredLookup.test_exact_match
    def test_exact_match(self):
        vec = tf.Variable([[1, 2, 3], [10, 20, 30]], dtype=tf.float32)
        key = tf.Variable([[0, 1, 0], [1, 0, 0]], dtype=tf.float32)
        target = tf.constant([2, 10], dtype=tf.float32)

        with tf.GradientTape(persistent=True) as tape:
            result = asymmetrical_vectored_lookup(vec, key)
            loss = tf.nn.l2_loss(result - target)

        npt.assert_almost_equal(result, [2, 10])
        npt.assert_almost_equal(loss, 0)
        npt.assert_almost_equal(tape.gradient(loss, vec), [[0, 0, 0], [0, 0, 0]])
        npt.assert_almost_equal(tape.gradient(loss, key), [[0, 0, 0], [0, 0, 0]])

    # python -m unittest library.array_ops_test.AsymmetricalVectoredLookup.test_single_mismatch
    def test_single_mismatch(self):
        vec = tf.Variable([[1, 2, 3], [10, 20, 30]], dtype=tf.float32)
        key = tf.Variable([[0, 1, 0], [1, 0, 0]], dtype=tf.float32)
        target = tf.constant([2, 20], dtype=tf.float32)

        with tf.GradientTape(persistent=True) as tape:
            result = asymmetrical_vectored_lookup(vec, key)
            loss = tf.nn.l2_loss(result - target)

        npt.assert_almost_equal(result, [2, 10])
        npt.assert_almost_equal(loss, 50)
        npt.assert_almost_equal(tape.gradient(loss, vec), [[0, 0, 0], [-10, 0, 0]])
        npt.assert_almost_equal(tape.gradient(loss, key), [[0, 0, 0], [1, -1, 1]])

    # python -m unittest library.array_ops_test.AsymmetricalVectoredLookup.test_chained_exact_match
    def test_chained_exact_match(self):
        vec = tf.Variable([[1, 2, 3], [10, 20, 30]], dtype=tf.float32)
        key1 = tf.Variable([[0, 1, 0], [1, 0, 0]], dtype=tf.float32)
        key2 = tf.Variable([[1, 0]], dtype=tf.float32)
        target1 = tf.constant([2, 10], dtype=tf.float32)
        target2 = tf.constant([2], dtype=tf.float32)

        with tf.GradientTape(persistent=True) as tape:
            result1 = asymmetrical_vectored_lookup(vec, key1)
            loss1 = tf.nn.l2_loss(result1 - target1)

            result2 = asymmetrical_vectored_lookup(tf.expand_dims(result1, 0), key2)
            loss2 = tf.nn.l2_loss(result2 - target2)

        npt.assert_almost_equal(result1, [2, 10])
        npt.assert_almost_equal(loss1, 0)
        npt.assert_almost_equal(tape.gradient(loss1, vec), [[0, 0, 0], [0, 0, 0]])
        npt.assert_almost_equal(tape.gradient(loss1, key1), [[0, 0, 0], [0, 0, 0]])

        npt.assert_almost_equal(result2, [2])
        npt.assert_almost_equal(loss2, 0)
        npt.assert_almost_equal(tape.gradient(loss2, vec), [[0, 0, 0], [0, 0, 0]])
        npt.assert_almost_equal(tape.gradient(loss2, key1), [[0, 0, 0], [0, 0, 0]])
        npt.assert_almost_equal(tape.gradient(loss2, key2), [[0, 0]])

    # python -m unittest library.array_ops_test.AsymmetricalVectoredLookup.test_chained_mismatch_last_layer
    def test_chained_mismatch_last_layer(self):
        vec = tf.Variable([[1, 2, 3], [10, 20, 30]], dtype=tf.float32)
        key1 = tf.Variable([[0, 1, 0], [1, 0, 0]], dtype=tf.float32)
        key2 = tf.Variable([[1, 0]], dtype=tf.float32)
        target1 = tf.constant([2, 10], dtype=tf.float32)
        target2 = tf.constant([10], dtype=tf.float32)

        with tf.GradientTape(persistent=True) as tape:
            result1 = asymmetrical_vectored_lookup(vec, key1)
            loss1 = tf.nn.l2_loss(result1 - target1)

            result2 = asymmetrical_vectored_lookup(tf.expand_dims(result1, 0), key2)
            loss2 = tf.nn.l2_loss(result2 - target2)

        npt.assert_almost_equal(result1, [2, 10])
        npt.assert_almost_equal(loss1, 0)
        npt.assert_almost_equal(tape.gradient(loss1, vec), [[0, 0, 0], [0, 0, 0]])
        npt.assert_almost_equal(tape.gradient(loss1, key1), [[0, 0, 0], [0, 0, 0]])

        npt.assert_almost_equal(result2, [2])
        npt.assert_almost_equal(loss2, 32)
        npt.assert_almost_equal(tape.gradient(loss2, vec), [[0, -8, 0], [0, 0, 0]])
        npt.assert_almost_equal(tape.gradient(loss2, key1), [[1, 1, -1], [0, 0, 0]])
        npt.assert_almost_equal(tape.gradient(loss2, key2), [[1, -1]])

    # python -m unittest library.array_ops_test.AsymmetricalVectoredLookup.test_chained_mismatch_2nd_last_layer
    def test_chained_mismatch_2nd_last_layer(self):
        vec = tf.Variable([[1, 2, 3], [10, 20, 30]], dtype=tf.float32)
        key1 = tf.Variable([[0, 1, 0], [1, 0, 0]], dtype=tf.float32)
        key2 = tf.Variable([[1, 0]], dtype=tf.float32)
        target1 = tf.constant([2, 10], dtype=tf.float32)
        target2 = tf.constant([30], dtype=tf.float32)

        with tf.GradientTape(persistent=True) as tape:
            result1 = asymmetrical_vectored_lookup(vec, key1)
            loss1 = tf.nn.l2_loss(result1 - target1)

            result2 = asymmetrical_vectored_lookup(tf.expand_dims(result1, 0), key2)
            loss2 = tf.nn.l2_loss(result2 - target2)

        npt.assert_almost_equal(result1, [2, 10])
        npt.assert_almost_equal(loss1, 0)
        npt.assert_almost_equal(tape.gradient(loss1, vec), [[0, 0, 0], [0, 0, 0]])
        npt.assert_almost_equal(tape.gradient(loss1, key1), [[0, 0, 0], [0, 0, 0]])

        npt.assert_almost_equal(result2, [2])
        npt.assert_almost_equal(loss2, 392)
        npt.assert_almost_equal(tape.gradient(loss2, vec), [[0, -28, 0], [0, 0, 0]])
        npt.assert_almost_equal(tape.gradient(loss2, key1), [[1, 1, -1], [0, 0, 0]])
        npt.assert_almost_equal(tape.gradient(loss2, key2), [[1, -1]])


class TestTensorLookup2D(TestCase):

    # python3 -m unittest notebooks.library.array_ops_test.TestTensorLookup2D.test_lookup1
    def test_lookup1(self):
        arr = tf.Variable(
            [
                [[1, 1], [1, 11], [1, 111]],
                [[2, 2], [2, 22], [2, 222]],
                [[3, 3], [3, 33], [3, 333]],
            ],
            dtype=tf.float32,
        )
        x_index = tf.Variable(tf.one_hot(1, 3), dtype=tf.float32)
        y_index = tf.Variable(tf.one_hot(2, 3), dtype=tf.float32)

        with tf.GradientTape(persistent=True) as tape:
            element = tensor_lookup_2d(arr, x_index, y_index)

        d_element_arr = tape.gradient(element, arr)
        d_element_x_index = tape.gradient(element, x_index)
        d_element_y_index = tape.gradient(element, y_index)

        self.assertEqual(element.shape, (2,))
        self.assertEqual(d_element_arr.shape, (3, 3, 2))
        self.assertEqual(d_element_x_index.shape, (3,))
        self.assertEqual(d_element_y_index.shape, (3,))

        npt.assert_almost_equal(element, [2, 222])


if __name__ == "__main__":
    unittest.main()

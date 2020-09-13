import tensorflow as tf
import numpy as np

TOKEN_DIM = 6
PRODUCTION_DIM = 4
STACK_SIZE = 10
PHI = np.eye(TOKEN_DIM)[0]
S = np.eye(TOKEN_DIM)[1]
O = np.eye(TOKEN_DIM)[2]
T = np.eye(TOKEN_DIM)[3]
X = np.eye(TOKEN_DIM)[4]
PLUS = np.eye(TOKEN_DIM)[5]

E = [PHI, PHI, PHI]

G_s = tf.constant([
    [E, E, E, E, E, E],
    [E, E, E, E, E, E],
    [E, [S, O, T], E, E, E, E],
    [E, [T, PHI, PHI], E, E, E, E],
], dtype=tf.float32)
G_o = tf.constant([
    [E, E, E, [X, PHI, PHI], E, E],
    [E, E, [PLUS, PHI, PHI], E, E, E],
    [E, E, E, E, E, E],
    [E, E, E, E, E, E],
], dtype=tf.float32)
grammar = (G_s, G_o)

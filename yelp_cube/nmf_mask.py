import tensorflow as tf
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix

def mnmf(mat, dim=10, alpha=0.1):

	mask = (mat != 0).toarray()
	mat = mat.toarray()
	tf_mask = tf.Variable(mask)
	A = tf.constant(mat, dtype=tf.float32)
	shape = mat.shape

	temp_H = np.random.randn(dim, shape[1]).astype(np.float32)
	temp_H = np.divide(temp_H, temp_H.max())
	temp_W = np.random.randn(shape[0], dim).astype(np.float32)
	temp_W = np.divide(temp_W, temp_W.max())

	H =  tf.Variable(temp_H)
	W = tf.Variable(temp_W)
	WH = tf.matmul(W, H)

	cost = tf.reduce_sum(tf.pow(tf.boolean_mask(A, tf_mask) - tf.boolean_mask(WH, tf_mask), 2))+alpha*(tf.norm(H)+tf.norm(W))
	lr = 0.001
	steps = 1000
	train_step = tf.train.GradientDescentOptimizer(lr).minimize(cost)
	init = tf.global_variables_initializer()
	clip_W = W.assign(tf.maximum(tf.zeros_like(W), W))
	clip_H = H.assign(tf.maximum(tf.zeros_like(H), H))
	clip = tf.group(clip_W, clip_H)

	with tf.Session() as sess:
		sess.run(init)
		for i in range(steps):
			sess.run(train_step)
			sess.run(clip)
			#if i%100==0:
				#print("\nCost: %f" % sess.run(cost))
				#print("*"*40)
		learnt_W = sess.run(W)
		learnt_H = sess.run(H)

	return np.dot(learnt_W, learnt_H)

if __name__ == '__main__':
	row = [1, 3, 5]
	col = [0, 2, 4]
	data = [1, 4, 2]
	mat = coo_matrix((np.array(data), (np.array(row), np.array(col))), shape=(6, 7))
	print(mnmf(mat))

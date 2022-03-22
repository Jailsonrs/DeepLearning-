import os
os.environ['TFF_CPP_MIN_LOG_LEVEL'] = "2" 
import tensorflow as tf
from tensorflow import keras


class RegLogLayer(keras.layers.Layer):

	# Estados (tf.Variable) podem ser criados em qualquer canto da classe de acordo com 
	# a conveniencia do implementador da subclasse. 

	def __init__(self, units = data.shape[2]):
		super(Linear, self).__init__()
		self.units = units
		
	def build(self, input_shape ):
#		alternativa para iniciar os pesos:
#		self.w = self.add_weight(shape=(input_shape[-1], self.units),
#                              initializer='random_normal',
#                              trainable=True)
		betas_init = tf.random_normal_initializer()
			self.betas = tf.Variable(
			initial_value = betas_init(shape = (self.units, ), dtype = "float32"),
			trainable = True
			)
		self.b0 = tf.Variable(
			initial_value = b_init(shape = (1, 1), dtype = "float32"),
			trainable = True
			)

	def call(self, inputs):
		xb = tf.matmul(X, self.betas)+self.b0
		return(xb)






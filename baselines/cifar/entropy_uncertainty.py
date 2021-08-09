import tensorflow as tf
import tensorflow_datasets as tfds
import uncertainty_baselines as ub
import utils  # local file import
from tensorboard.plugins.hparams import api as hp
import numpy as np 
import scipy.stats as stats
"""
S = Number of test examples in the array
N = number of models sampled 
D = number of output classes
"""

def row_entropy(row):
	    # _, _, count = tf.unique_with_counts(row)
	    # prob = count / tf.reduce_sum(count)
	    prob = row / tf.reduce_sum(row)
	    return -tf.reduce_sum(prob * tf.math.log(prob))

	#Takes an (N, D) tensor of predictions, outputs a (N) tensor of the entropies for each row. 
def rev_entropy(preds):
    rev = tf.map_fn(row_entropy, preds, dtype=tf.float32)
    return rev

#preds: (N, D) Numpy array a full prediction of probabilities for each model sampled. 
def sample_compute_entropy_uncertainty(preds): 
	# preds = preds.numpy()
	#simple entropy of the predicted distribution
	total = row_entropy(tf.reduce_mean(preds, axis=0))

	#fixing a model class, compute the entropy of that predicted distribution. 
	aleatoric = tf.reduce_mean(rev_entropy(preds))
	epistemic = total - aleatoric

	# print("Total: ", total, " aleatoric: ", aleatoric, " epistemic: ", epistemic)
	uncertainties = tf.stack([total, aleatoric, epistemic], axis=0)
	print("@@@ [Total, Aleatoric, Epistemic]")
	tf.print(uncertainties)
	return uncertainties


class Uncertainty(): 
	def __init__(self): 
		self.uncertainties_overall = None 

	def batch_uncertainty(self): 
		if self.uncertainties_overall == None: 
			print("error: haven't computed any uncertainties")
		else: 
			print("@@@ Batch Overall [Total, Aleatoric, Epistemic]")
			tf.print(tf.reduce_mean(self.uncertainties_overall, axis=0))
	#preds: (N, S, D)
	def compute_entropy_uncertainty(self, preds):
		t_preds = tf.transpose(preds, perm=[1, 0, 2]) #Now (S, N, D)
		uncertainties = tf.map_fn(sample_compute_entropy_uncertainty, t_preds, dtype=tf.float32)
		
		if self.uncertainties_overall == None: #todo move up a layer, should be possible without any bugs
			self.uncertainties_overall = uncertainties
		else: 
			self.uncertainties_overall = tf.stack((self.uncertainties_overall, uncertainties))
			print(self.uncertainties_overall.shape)

		# return uncertainties


def main(): 
	x = tf.constant([[
		[.25, .25, .25, .25],
		[.25, .25, .25, .25],
		[.25, .25, .25, .25]
		# [.75, .05, .1, .1], 
		# [.05, .75, .1, .1], 
		# [.05, .05, .1, .8], 
		]], dtype=tf.float32)
	print(x.shape)
	compute_entropy_uncertainty(x)


if __name__ == "__main__": 
	main() 


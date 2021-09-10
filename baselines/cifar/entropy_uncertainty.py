import tensorflow as tf
import tensorflow_datasets as tfds
import uncertainty_baselines as ub
import utils  # local file import
from tensorboard.plugins.hparams import api as hp
import numpy as np 
import scipy.stats as stats
import sys
import matplotlib.pyplot as plt
"""
S = Number of test examples in the array (64)
N = number of models sampled (5)
D = number of output classes (10)
"""

def row_entropy(row):
        # _, _, count = tf.unique_with_counts(row)
        # prob = count / tf.reduce_sum(count)
        prob = row / tf.reduce_sum(row)
        return -tf.reduce_sum(prob * tf.math.log(prob))

    #Takes an (N, D) tensor of predictions, outputs a (N) tensor of the entropies for each row. 
def rev_entropy(preds):
    # rev = tf.map_fn(row_entropy, preds, dtype=tf.float32)
    rev = []
    for i in range(preds.shape[0]): 
        rev.append(row_entropy(preds[i]))
        
    return tf.convert_to_tensor(rev)

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
    # print("Sample Uncertainty")
    # tf.print(uncertainties, output_stream=sys.stdout)
    return uncertainties


#Labels: S
#preds: (N, S, D)
def compute_entropy_uncertainty(labels, preds, epoch_count, make_plot=False):
    t_preds = tf.transpose(preds, perm=[1, 0, 2])
    S = t_preds.shape[0]
    uncertainties_overall = []
    for i in range(S): 
        uncertainties_overall.append(sample_compute_entropy_uncertainty(t_preds[i]))
    uncertainties_overall = tf.stack(uncertainties_overall)
    # print("@@@ Batch Overall [Total, Aleatoric, Epistemic]")
    # tf.print(uncertainties_overall, output_stream=sys.stdout)
    # print("Batch Avg")
    # tf.print(tf.reduce_mean(uncertainties_overall, axis=0), output_stream=sys.stdout)


    # print("labels", labels.shape) # S 
    # print("t_preds", t_preds.shape) # S N D 
    # exit() 
    negative_log_likelihood = tf.keras.losses.sparse_categorical_crossentropy(labels, tf.reduce_mean(t_preds, axis=1))

    uncertainty_types = ["total", "aleatoric", "epistemic"]
    # print("uncertainties shape", uncertainties_overall.shape) #should be S, 3
    # print("NLL shape", negative_log_likelihood.shape) #should be S
    # print(tf.executing_eagerly())
    # tf.compat.v1.enable_eager_execution()
    # print(tf.executing_eagerly())

    if make_plot: 
        for i in range(3): 
            xvals = uncertainties_overall.numpy()
            yvals = negative_log_likelihood.numpy() 
            # print("got here")
            # xvals = tf.make_ndarray(uncertainties_overall.op.get_attr('value'))
            # yvals = tf.make_ndarray(negative_log_likelihood.op.get_attr('value'))

            # xvals = uncertainties_overall.eval(session=tf.compat.v1.Session())    
            # yvals = negative_log_likelihood.eval(session=tf.compat.v1.Session()) 
            plt.scatter(xvals[:, i], yvals)
            # plt.show()
            path = "/home/thlarsen/ood_detection/uncertainty-baselines/plots/" + uncertainty_types[i] + "_uncertainty_nll_epoch_" + str(epoch_count) + ".png"
            # epoch_count += 1
            plt.savefig(path)

    # exit() 
    # exit()


# class Uncertainty(): 
#   def __init__(self): 
#       self.uncertainties_overall = None 

#   def batch_uncertainty(self): 
#       if self.uncertainties_overall == None: 
#           print("error: haven't computed any uncertainties")
#       else: 
#           print("@@@ Batch Overall [Total, Aleatoric, Epistemic]")
#           tf.print(tf.reduce_mean(self.uncertainties_overall, axis=0))
#   #preds: (N, S, D)
#   def compute_entropy_uncertainty(self, preds):
#       t_preds = tf.transpose(preds, perm=[1, 0, 2])
#       uncertainties = tf.map_fn(sample_compute_entropy_uncertainty, t_preds, dtype=tf.float32)

#       if self.uncertainties_overall == None: #todo move up a layer, should be possible without any bugs
#           self.uncertainties_overall = uncertainties
#       else: 
#           self.uncertainties_overall = tf.stack((self.uncertainties_overall, uncertainties))
#           print(self.uncertainties_overall.shape)

#       return uncertainties


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


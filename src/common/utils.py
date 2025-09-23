import random
import numpy as np 
import tensorflow as tf

# seed
def set_seed(seed=42):
    """Set seed for random, numpy, and tensorflow."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

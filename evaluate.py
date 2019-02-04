import os
import tensorflow as tf
import numpy as np

from model.input_fn import test_input_fn
from model.model_fn import model_fn
from model.utils import Params

from config import *

if __name__ == '__main__':
    tf.reset_default_graph()
    tf.logging.set_verbosity(tf.logging.INFO)

    # Load the parameters
    model_dir = 'experiments/base_model'
    data_dir = os.path.join(DATASET_DIR_PATH,"images_evaluation")
    json_path = os.path.join(model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # Define the model
    tf.logging.info("Creating the model...")
    estimator = tf.estimator.Estimator(model_fn, params=params, model_dir=model_dir)

    # Compute embeddings on the test set
    tf.logging.info("Predicting")
    predictions = estimator.predict(lambda: test_input_fn(data_dir, params))

    embeddings = np.zeros((params.eval_size, params.embedding_size))
    for i, p in enumerate(predictions):
        embeddings[i] = p['embeddings']

    tf.logging.info("Embeddings shape: {}".format(embeddings.shape))

    with tf.Session() as sess:
        # Obtain the test labels
        dataset = get_dataset(test_dir, params)
        dataset = dataset.map(lambda img, lab: lab)
        dataset = dataset.batch(params.eval_size)
        labels_tensor = dataset.make_one_shot_iterator().get_next()
        labels = sess.run(labels_tensor)
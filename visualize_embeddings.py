import os
import pathlib
import shutil

import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

from model.utils import Params
from model.input_fn import test_input_fn, get_dataset
from model.model_fn import model_fn

from config import *

if __name__ == '__main__':
    tf.reset_default_graph()
    tf.logging.set_verbosity(tf.logging.INFO)

    model_dir = 'experiments/base_model'
    test_dir = os.path.join(DATASET_DIR_PATH,"images_evaluation")
    sprite_filename = 'experiments/ominiglot_sprite.png'

    json_path = os.path.join('experiments/batch_hard', 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # Define the model
    tf.logging.info("Creating the model...")
    config = tf.estimator.RunConfig(tf_random_seed=230,
                                    model_dir=model_dir,
                                    save_summary_steps=params.save_summary_steps)
    estimator = tf.estimator.Estimator(model_fn, params=params, config=config)

    # EMBEDDINGS VISUALIZATION

    # Compute embeddings on the test set
    tf.logging.info("Predicting")
    predictions = estimator.predict(lambda: test_input_fn(test_dir, params))

    # TODO (@omoindrot): remove the hard-coded 10000
    embeddings = np.zeros((params.eval_size, params.embedding_size))
    for i, p in enumerate(predictions):
        embeddings[i] = p['embeddings']

    tf.logging.info("Embeddings shape: {}".format(embeddings.shape))

    # Visualize test embeddings
    embedding_var = tf.Variable(embeddings, name='ominiglot_embedding')

    eval_dir = os.path.join(model_dir, "eval")
    summary_writer = tf.summary.FileWriter(eval_dir)

    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_var.name

    # Specify where you find the sprite (we will create this later)
    # Copy the embedding sprite image to the eval directory
    #shutil.copy2(sprite_filename, eval_dir)
    #embedding.sprite.image_path = pathlib.Path(sprite_filename).name
    #embedding.sprite.single_image_dim.extend([params.image_size, params.image_size])

    with tf.Session() as sess:
        # TODO (@omoindrot): remove the hard-coded 10000
        # Obtain the test labels
        dataset = get_dataset(test_dir, params)
        dataset = dataset.map(lambda img, lab: lab)
        dataset = dataset.batch(params.eval_size)
        labels_tensor = dataset.make_one_shot_iterator().get_next()
        labels = sess.run(labels_tensor)

    # Specify where you find the metadata
    # Save the metadata file needed for Tensorboard projector
    metadata_filename = "ominiglot_metadata.tsv"
    with open(os.path.join(eval_dir, metadata_filename), 'w') as f:
        for i in range(params.eval_size):
            c = labels[i]
            f.write('{}\n'.format(c))
    embedding.metadata_path = metadata_filename

    # Say that you want to visualise the embeddings
    projector.visualize_embeddings(summary_writer, config)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(embedding_var.initializer)
        saver.save(sess, os.path.join(eval_dir, "embeddings.ckpt"))

    

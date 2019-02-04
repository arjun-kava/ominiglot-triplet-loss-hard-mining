import os
import pathlib
import shutil

import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import numpy as np

from model.input_fn import train_input_fn, test_input_fn, valid_input_fn, get_dataset, get_labels
from model.model_fn import model_fn
from model.utils import Params, vector_to_matrix_mnist, invert_grayscale, create_sprite_image
from config import *

model_dir = "experiments/base_model"
train_dir = os.path.join(DATASET_DIR_PATH,"images_background")
test_dir = os.path.join(DATASET_DIR_PATH,"images_evaluation")
sprite_filename = 'experiments/ominiglot_sprite.png'

tf.reset_default_graph()
tf.logging.set_verbosity(tf.logging.INFO)

json_path = os.path.join("experiments/batch_hard/params.json")
params = Params(json_path)

# Define the model
tf.logging.info("Creating the model...")
config = tf.estimator.RunConfig(tf_random_seed=230,
                                model_dir=model_dir,
                                save_summary_steps=params.save_summary_steps)
estimator = tf.estimator.Estimator(model_fn, params=params, config=config)

# Train the model
tf.logging.info("Starting training for {} epoch(s).".format(params.num_epochs))

train_spec = tf.estimator.TrainSpec(
    input_fn = lambda: train_input_fn(train_dir, params), 
    #max_steps= ((params.train_size * (1 - params.val_split)) / params.batch_size) * params.num_epochs
)

eval_spec = tf.estimator.EvalSpec(
    input_fn = lambda: valid_input_fn(train_dir, params),
    #steps=((params.train_size * params.val_split) / params.batch_size) * params.num_epochs,
    throttle_secs=120,
    start_delay_secs=120,
)

tf.estimator.train_and_evaluate(
    estimator,
    train_spec,
    eval_spec
)

"""
# For Training Only
estimator.train(lambda: train_input_fn(train_dir, params))
"""

# Evaluate the model on the test set
tf.logging.info("Evaluation on test set.")
res = estimator.evaluate(lambda: test_input_fn(test_dir, params))
for key in res:
    print("{}: {}".format(key, res[key]))
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
shutil.copy2(sprite_filename, eval_dir)
embedding.sprite.image_path = pathlib.Path(sprite_filename).name
embedding.sprite.single_image_dim.extend([28, 28])

# Specify where you find the metadata
# Save the metadata file needed for Tensorboard projector
metadata_filename = "ominiglot_metadata.tsv"
label_captions = get_labels(test_dir)
print("len(label_captions)", len(label_captions))
with open(os.path.join(eval_dir, metadata_filename), 'w') as f:
    for i in range(params.eval_size):
        c = label_captions[i]
        f.write('{}\n'.format(c))
embedding.metadata_path = metadata_filename

# Say that you want to visualise the embeddings
projector.visualize_embeddings(summary_writer, config)

saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(embedding_var.initializer)
    saver.save(sess, os.path.join(eval_dir, "embeddings.ckpt"))


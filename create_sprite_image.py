import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from model.utils import *
from model.input_fn import get_dataset, parse_function, train_preprocess, read_image

from config import *

test_dir = os.path.join(DATASET_DIR_PATH,"images_evaluation")
sprite_filename = 'experiments/ominiglot_sprite.png'
json_path = os.path.join("experiments/batch_hard/params.json")
params = Params(json_path)
params.image_size = 28

with tf.Session() as sess:
    # TODO (@omoindrot): remove the hard-coded 10000
    # Obtain the test labels
    dataset = get_dataset(test_dir, params)
    dataset = dataset.map((lambda x,y: read_image(x,y, params)), num_parallel_calls=4)
    dataset = dataset.batch(params.eval_size)
    iterator = dataset.make_one_shot_iterator()
    data_X, data_y = iterator.get_next()
    X = sess.run(data_X)

print("X.shape", X.shape)
to_visualise = vector_to_matrix_mnist(X, params.image_size, params.image_size)
to_visualise = invert_grayscale(to_visualise)

sprite_image = create_sprite_image(to_visualise)

plt.imsave(sprite_filename,sprite_image,cmap='gray')
plt.imshow(sprite_image,cmap='gray')



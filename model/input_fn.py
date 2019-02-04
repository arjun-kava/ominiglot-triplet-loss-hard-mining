import tensorflow as tf
import gzip
import os
import shutil
import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split

from config import *

def get_dataset(data_dir, params, subset=None):
    data_frames = get_data_frames(data_dir)

    if subset == "train":
        train_indexes, val_indexes = train_test_split(data_frames, 
                                                        test_size=params.val_split, 
                                                        random_state=42, 
                                                        stratify=data_frames[['Id']])
        data_frames = train_indexes
    elif subset == "val":
        train_indexes, val_indexes = train_test_split(data_frames, 
                                                        test_size=params.val_split, 
                                                        random_state=42, 
                                                        stratify=data_frames[['Id']])
        data_frames = val_indexes


    filenames = data_frames.Image.values
    labels = data_frames.Label.values
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    return dataset

def train_input_fn(data_dir, params):
    dataset = get_dataset(data_dir, params, subset="train")
    dataset = dataset.shuffle(params.train_size)
    dataset = dataset.map((lambda x,y: parse_function(x, y, params)), num_parallel_calls=4)
    dataset = dataset.map(train_preprocess, num_parallel_calls=4)
    dataset = dataset.repeat(params.num_epochs)
    dataset = dataset.batch(params.batch_size)
    dataset = dataset.prefetch(1)
    return dataset


def valid_input_fn(data_dir, params):
    dataset = get_dataset(data_dir, params, subset="train")
    dataset = dataset.map((lambda x,y: parse_function(x, y, params)), num_parallel_calls=4)
    dataset = dataset.map(train_preprocess, num_parallel_calls=4)
    dataset = dataset.repeat(params.num_epochs)
    dataset = dataset.batch(params.batch_size)
    dataset = dataset.prefetch(1)
    return dataset

def test_input_fn(data_dir, params):
    dataset = get_dataset(data_dir, params)
    dataset = dataset.map((lambda x,y: parse_function(x, y, params)), num_parallel_calls=4)
    dataset = dataset.map(train_preprocess, num_parallel_calls=4)
    dataset = dataset.batch(params.batch_size)
    dataset = dataset.prefetch(1)  # make sure you always have one batch ready to serve
    return dataset

def parse_function(filename, label, params):
    image_string = tf.read_file(filename)

    # Don't use tf.image.decode_image, or the output shape will be undefined
    image = tf.image.decode_jpeg(image_string, channels=params.image_channels)

    # This will convert to float values in [0, 1]
    image = tf.image.convert_image_dtype(image, tf.float32)

    image = tf.image.resize_images(image, [params.image_size, params.image_size])
    return image, label

def read_image(filename, label, params):
    image_string = tf.read_file(filename)

    # Don't use tf.image.decode_image, or the output shape will be undefined
    image = tf.image.decode_jpeg(image_string, channels=params.image_channels)


    image = tf.image.resize_images(image, [params.image_size, params.image_size])
    return image, label

def train_preprocess(image, label):
    image = tf.image.random_flip_left_right(image)

    #image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
    #image = tf.image.random_saturation(image, lower=0.5, upper=1.5)

    # Make sure the image is still in [0, 1]
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label

def get_data_frames(dataset_dir_path):
    raw_data = {"Id":[], "Image": [], "Label": [] }
    alphabeths = os.listdir(dataset_dir_path)
    labels = []

    for alphabet in tqdm(alphabeths, desc="Fetching"):
        if alphabet != ".DS_Store":
            characters_path = os.path.join(dataset_dir_path, alphabet)
            characters = os.listdir(characters_path)

            for character in characters:
                images_path = os.path.join(characters_path, character)
                _id = alphabet + "_" + character
                _images = [os.path.join(dp, f) for dp, dn, filenames in os.walk(images_path) for f in filenames if os.path.splitext(f)[1] == '.png']

                if _id not in labels:
                    labels.append(_id)

                for image_path in _images:
                    raw_data["Id"].append(_id)
                    raw_data["Image"].append(image_path)
                    raw_data["Label"].append(labels.index(_id))
    df = pd.DataFrame(raw_data, columns = ["Id", "Image", "Label"])
    return df

def get_labels(dataset_dir_path):
    alphabeths = os.listdir(dataset_dir_path)
    labels = []

    for alphabet in tqdm(alphabeths, desc="Get Labels"):
        if alphabet != ".DS_Store":
            characters_path = os.path.join(dataset_dir_path, alphabet)
            characters = os.listdir(characters_path)

            for character in characters:
                images_path = os.path.join(characters_path, character)
                _id = alphabet + "_" + character
                _images = [os.path.join(dp, f) for dp, dn, filenames in os.walk(images_path) for f in filenames if os.path.splitext(f)[1] == '.png']

                for index in range(len(_images)):
                    labels.append(_id)
    return labels

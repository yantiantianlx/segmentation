import tensorflow as tf
import os
import numpy as np

def get_dataset(parent_path, batch_size = 32,train = 'train', img_width = 320, img_height = 240):
    train_path = os.path.join(parent_path, train)
    img_path = os.path.join(train_path, 'imgs')
    label_path = os.path.join(train_path, 'labels')
    imgs = []
    labels = []
    for name in os.listdir(img_path):
        # read img
        img_name = os.path.join(img_path, name)
        img = tf.read_file(img_name)
        img = tf.image.decode_png(img)
        imgs.append(img)
        # read label
        label_name = os.path.join(label_path, name)
        label = tf.read_file(label_name)
        label = tf.image.decode_png(label)
        labels.append(label)
    imgs = tf.convert_to_tensor(np.array(imgs), dtype=tf.string)
    labels = tf.convert_to_tensor(np.array(labels), dtype=tf.string)

    # data processing func used in map
    def preprocessing(image, label):
        image = tf.image.resize_images(image, [img_width, img_height])
        # image = tf.image.random_flip_left_right(image)
        # image = tf.image.random_brightness(image, max_delta=0.1)
        # image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
        image = tf.cast(image, tf.float32)
        image = image / 255.0
        return image, label

    # make dataset
    dataset = tf.data.Dataset.from_tensor_slices((imgs, labels))
    dataset = dataset.map(preprocessing)
    dataset = dataset.shuffle(batch_size * 10)
    dataset = dataset.batch(batch_size)
    return dataset

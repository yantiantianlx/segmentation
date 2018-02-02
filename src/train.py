import tensorflow as tf
from DataReader.tfrecords import TFRecords_Reader
from net import net

import time
import argparse
import numpy as np
import os
import matplotlib.pyplot as pl

parent_dir = os.path.abspath('..') # 获得当前工作目录父目录

# Training settings
parser = argparse.ArgumentParser(description='example')
parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training (default: 32)')
parser.add_argument('--img_width', type=int, default=400, help='the width of the image(default: 320)')
parser.add_argument('--img_height', type=int, default=300, help='the height of the image(default: 240)')
parser.add_argument('--num_classes', type=int, default=2, help='the kinds of classes(default: 2)')
parser.add_argument('--init_lr', type=float, default=1e-3, help='the init of the learning_rate(default: 1e-3)')
parser.add_argument('--max_epoch', type=int, default=1000, help='the init of the learning_rate(default: 1e-3)')
parser.add_argument('--snapshot_iter', type=int, default=2, help='the init of the learning_rate(default: 1e-3)')
parser.add_argument('--display_iter', type=int, default=100, help='the init of the learning_rate(default: 1e-3)')

parser.add_argument('--log_path', type=str, default=parent_dir + '\\logs\\', help='the path to save logs')
parser.add_argument('--load_model_path', type=str, default=parent_dir + '\\models\\load', help='the path to load model')
parser.add_argument('--save_model_path', type=str, default=parent_dir + '\\models\\save', help='the path to save model')

parser.add_argument('--datas_path', type=str, default=parent_dir + '\\datas\\train.tfrecords', help='the path to save logs')
args = parser.parse_args()

tfreader = TFRecords_Reader()
#tfreader.write_records(img_dir,annot_image_dir,tfrecords_name)
index, image, label = tfreader.readbatch_by_queue(args.datas_path,batch_size=args.batch_size,num_epoch=args.max_epoch)

config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
with tf.Session(config=config) as sess:
    # 构建模型
    UNET = net(sess, args.batch_size, args.img_height, args.img_width, args.init_lr, num_classes=args.num_classes)

    # 初始化权重参数
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    saver = tf.train.Saver()
    UNET.load_model(saver,args.load_model_path)
    summary_writer = tf.summary.FileWriter(args.log_path, graph=tf.get_default_graph())

    iter = 0
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    try:
        while not coord.should_stop():
            iter = iter + 1
             # 改变学习率
            if iter <= 100:
                UNET.set_learning_rate(learning_rate=1e-2)
            elif (iter > 100 and iter <= 1000):
                UNET.set_learning_rate(learning_rate=1e-3)
            else:
                UNET.set_learning_rate(learning_rate=1e-4)

            # 训练
            index_batch, image_batch, label_batch = sess.run([index, image, label])
            batch_start = time.time()
            train_op, cost, pred, summary = UNET.train_batch(image_batch, label_batch)
            time_taken = time.time() - batch_start
            images_per_sec = args.batch_size / time_taken
            summary_writer.add_summary(summary, iter)

            # 保存模型
            if (iter + 1) % args.snapshot_iter == 0:
                UNET.save_model(saver, args.save_model_path, iter)

            # 显示测试数据准确率/损失/速度
            if (iter + 1) % args.display_iter == 0:
                pass
    except tf.errors.OutOfRangeError:
        print('Done training')
    finally:
        coord.request_stop()

    coord.join(threads)

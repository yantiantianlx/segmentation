import tensorflow as tf
from nets.unet import unet as network
from nets import layers

class net(object):
    def __init__(self,sess, batch_size, img_height, img_width, learning_rate, num_classes=2,is_training=True):
        self.sess = sess
        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width

        self.input_tensor = tf.placeholder(tf.float32, [batch_size, img_height, img_width, 3])
        self.gt_labels = tf.placeholder(tf.int32, [batch_size, img_height, img_width, 1])
        self.gt = tf.cast(tf.reshape(self.gt_labels, [-1]), tf.int32)

        he_initializer = tf.contrib.layers.variance_scaling_initializer()
        def build_network(initializer, input_batch, label_batch, is_training, num_classes=2):
            pred = network(input=input_batch, initializer=initializer, num_classes=num_classes, training=is_training)
            classes = tf.cast(tf.argmax(pred, 3), tf.uint8)
            cost = layers.loss(predictions=pred, labels=label_batch)
            return pred, classes, cost
        self.prediction, self.pred_classes, self.cost = build_network(initializer = he_initializer,
                                                                           input_batch = self.input_tensor,
                                                                           label_batch=self.gt,
                                                                           is_training=is_training,
                                                                           num_classes=num_classes)
        self.learning_rate_placeholder = tf.placeholder(tf.float32, [], name='learning_rate')
        self.learning_rate = 1e-3

        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_placeholder)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = optimizer.minimize(self.cost)

        tf.summary.scalar("loss", self.cost)
        self.merged_summary_op = tf.summary.merge_all()

    def get_cost_accuracy(self, inputs, labels):
        cost, pred = self.sess.run([self.cost, self.prediction], feed_dict={self.input_tensor: inputs, self.gt_labels: labels})
        acc = layers.accuracy(predictions=pred, labels=labels)
        return cost, acc

    def get_pred_classes(self, inputs):
        return self.sess.run([self.pred_classes], feed_dict={self.input_tensor: inputs })

    def set_learning_rate(self, learning_rate=1e-3):
        self.learning_rate = learning_rate

    def train_batch(self, inputs, labels):
        return self.sess.run([self.train_op, self.cost, self.prediction, self.merged_summary_op], feed_dict={
            self.input_tensor: inputs, self.gt_labels: labels,
            self.learning_rate_placeholder: self.learning_rate
        })

    def save_model(self,saver, model_path, iter):
        saver.save(self.sess, model_path + '\\model.ckpt', global_step=iter)

    def load_model(self,saver, model_path):
        ckpt = tf.train.get_checkpoint_state(model_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(self.sess, ckpt.model_checkpoint_path)
            print('model loading')
        else:
            print('model loading error !!')


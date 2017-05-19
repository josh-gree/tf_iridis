import tensorflow as tf

import os
import fire

from mnist_load import mnist_load


class Model(object):

    def __init__(self, sess):

        # initialization

        self.sess = sess
        self.loaded_model = False

        self.train_set, self.test_set = mnist_load()

        # END build model
        self.build_model()

    def build_model(self):

        # Declare Inputs
        self.x_ = tf.placeholder(
            tf.float32, shape=[None, 28 * 28], name="inp")
        self.y_ = tf.placeholder(tf.float32, shape=[None, 10], name="targ")

        # Create model and params
        self.out = self.model(self.x_)

        # Loss
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=self.y_, logits=self.out))
        # Saver object
        self.saver = tf.train.Saver()

    def model(self, x):

        # model should take training input and produce transformed as output

        im = tf.reshape(x, [-1, 28, 28, 1])
        c1 = tf.layers.conv2d(
            inputs=im,
            filters=16,
            kernel_size=[3, 3],
            padding="valid"
        )
        c2 = tf.layers.conv2d(
            inputs=c1,
            filters=32,
            kernel_size=[3, 3],
            padding="valid",
            activation=tf.nn.sigmoid
        )
        flatten = tf.reshape(c2, [-1, 24 * 24 * 32])
        fc = tf.layers.dense(
            inputs=flatten,
            units=10,
            activation=None
        )

        return fc

    def init(self):
        # function to call to initialse variables
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def train(self, N_epochs=10):

        # training op created
        g_optim = tf.train.AdamOptimizer(1e-2).minimize(self.loss)

        if not self.loaded_model:
            self.init()

        # main training loop
        for i in range(N_epochs):

            for ind, (inp_, target_) in enumerate(self.train_set):
                self.sess.run(g_optim, feed_dict={self.x_: inp_,
                                                  self.y_: target_})

                curr_loss = self.sess.run(self.loss, feed_dict={self.x_: inp_,
                                                                self.y_: target_})

                if ind % 100 == 0:
                    print('{}: {}'.format(ind, curr_loss))

                    self.save(i)

            print("Test accuracy {:.2f}%".format(self.acc()))

    def acc(self):

        correct = 0
        for x, y in self.test_set:
            correct += sum(
                self.sess.run(tf.nn.softmax(self.out),
                              feed_dict={self.x_: x, self.y_: y}).argmax(axis=1) == y.argmax(axis=1)
            )

        return correct / float(len(self.test_set))

    def save(self, step):

        # to save model state
        model_name = "tf_mnist.model"

        if not os.path.exists(model_name):
            os.makedirs(model_name)

        self.saver.save(self.sess,
                        os.path.join(model_name, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):

        self.loaded_model = True
        self.init()

        saver = tf.train.Saver()
        # to load model state

        model_name = "tf_mnist.model"

        ckpt = tf.train.get_checkpoint_state(model_name)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(self.sess, os.path.join(model_name, ckpt_name))
            return True
        else:
            return False


if __name__ == '__main__':
    sess = tf.Session()
    m = Model(sess)
    fire.Fire(m.train)

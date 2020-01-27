import os

import tensorflow as tf

from model import graph_plotter


class BaseModel(object):
    """Generic class for general methods that are not specific to NER"""

    def __init__(self, config):
        """Defines self.config and self.logger

        Args:
            config: (Config instance) class with hyper parameters,
                vocab and embeddings

        """
        self.config = config
        self.logger = config.logger
        self.sess = None
        self.saver = None

    def reinitialize_weights(self, scope_name):
        """Reinitializes the weights of a given layer"""
        variables = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=scope_name)
        init = tf.compat.v1.variables_initializer(variables)
        self.sess.run(init)

    def add_train_op(self, lr_method, lr, loss, clip=-1, is_restored=False):
        """Defines self.train_op that performs an update on a batch

        Args:
            lr_method: (string) sgd method, for example "adam"
            lr: (tf.placeholder) tf.float32, learning rate
            loss: (tensor) tf.float32 loss to minimize
            clip: (python float) clipping of gradient. If < 0, no clipping

        """
        _lr_m = lr_method.lower()  # lower to make sure

        with tf.compat.v1.variable_scope("train_step", reuse=tf.compat.v1.AUTO_REUSE, auxiliary_name_scope=not is_restored):
            if _lr_m == 'adam':  # sgd method
                optimizer = tf.compat.v1.train.AdamOptimizer(lr)
            elif _lr_m == 'adagrad':
                optimizer = tf.compat.v1.train.AdagradOptimizer(lr)
            elif _lr_m == 'sgd':
                optimizer = tf.compat.v1.train.GradientDescentOptimizer(lr)
            elif _lr_m == 'rmsprop':
                optimizer = tf.compat.v1.train.RMSPropOptimizer(lr)
            else:
                raise NotImplementedError("Unknown method {}".format(_lr_m))

            if clip > 0:  # gradient clipping if clip is positive
                grads, vs = zip(*optimizer.compute_gradients(loss))
                grads, gnorm = tf.clip_by_global_norm(grads, clip)
                self.train_op = optimizer.apply_gradients(zip(grads, vs))
            else:
                self.train_op = optimizer.minimize(loss)

    def initialize_session(self):
        """Defines self.sess and initialize the variables"""
        self.logger.info("Initializing tf session")
        self.sess = tf.compat.v1.Session()
        self.sess.run(tf.compat.v1.global_variables_initializer())
        self.saver = tf.compat.v1.train.Saver()

    def restore_session(self, dir_model):
        """Reload weights into session

        Args:
            sess: tf.Session()
            dir_model: dir with weights

        """
        self.logger.info("Reloading the latest trained model...")

        self.saver.restore(self.sess, dir_model)
        # self.saver.restore(self.sess, dir_model+".data-00000-of-00001")

    def save_session(self, directory=None):
        if directory == None:
            directory = self.config.dir_model
        """Saves session = weights"""
        if not os.path.exists(directory):
            os.makedirs(directory)
        self.saver.save(self.sess, directory)

    def close_session(self):
        """Closes the session"""
        self.sess.close()

    def add_summary(self):
        """Defines variables for Tensorboard

        Args:
            dir_output: (string) where the results are written

        """
        self.merged = tf.compat.v1.summary.merge_all()
        self.file_writer = tf.compat.v1.summary.FileWriter(self.config.dir_model_output,
                                                           self.sess.graph)

    def train(self, train, dev, nepochs=None, output_dir=None):
        """Performs training with early stopping and lr exponential decay

        Args:
            train: dataset that yields tuple of (sentences, tags)
            dev: dataset

        """
        if output_dir == None:
            output_dir = self.config.dir_model
        best_score = 0
        nepoch_no_imprv = 0  # for early stopping
        self.add_summary()  # tensorboard
        if nepochs == None:
            nepochs = self.config.nepochs
        validation_metrics = list()
        training_metrics = list()
        epoch = 0
        # for epoch in range(nepochs):
        while True:
            self.logger.info("Epoch {:} out of {:}".format(epoch + 1,
                                                           nepochs))

            epoch_validation_metric, epoch_training_metric = self.run_epoch(train, dev, epoch + 1)
            validation_metrics.append(epoch_validation_metric)
            training_metrics.append(epoch_training_metric)
            self.config.lr *= self.config.lr_decay  # decay learning rate
            graph_plotter.plot_graphs(validation_metrics, training_metrics, self.config.dir_model_output)

            # early stopping and saving best parameters
            if float(epoch_validation_metric['F1 score']) > best_score:
                nepoch_no_imprv = 0
                self.save_session(output_dir)
                best_score = epoch_validation_metric['F1 score']
                self.logger.info("- new best score!")
            else:
                nepoch_no_imprv += 1
                if nepoch_no_imprv >= self.config.nepoch_no_imprv:
                    self.logger.info("- early stopping {} epochs without " \
                                     "improvement".format(nepoch_no_imprv))
                    break
            epoch += 1
        graph_plotter.plot_graphs(validation_metrics, training_metrics, self.config.dir_model_output)

    def evaluate(self, test):
        """Evaluate model on test set

        Args:
            test: instance of class Dataset

        """
        self.logger.info("Testing model over test set")
        metrics = self.run_evaluate(test)
        msg = " - ".join(["{} {:04.2f}".format(k, v)
                          for k, v in metrics.items()])
        self.logger.info(msg)

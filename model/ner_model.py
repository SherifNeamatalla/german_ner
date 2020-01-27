import sys

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from .base_model import BaseModel
from .data_utils import load_vocab_rev, load_vocab, Word
from .data_utils import minibatches, pad_sequences, get_chunks, get_oov_embeddings
from .general_utils import Progbar


class NERModel(BaseModel):
    """Specialized class of Model for NER"""

    def __init__(self, config):
        tf.compat.v1.disable_eager_execution()
        print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
        super(NERModel, self).__init__(config)
        self.idx_to_tag = {idx: tag for tag, idx in
                           self.config.vocab_tags.items()}

    def add_placeholders(self):
        """Define placeholders = entries to computational graph"""
        # shape = (batch size, max length of sentence in batch)
        self.word_ids = tf.compat.v1.placeholder(tf.int32, shape=[None, None],
                                                 name="word_ids")

        # shape = (batch size)
        self.sequence_lengths = tf.compat.v1.placeholder(tf.int32, shape=[None],
                                                         name="sequence_lengths")

        # shape = (batch size, max length of sentence, max length of word)
        self.char_ids = tf.compat.v1.placeholder(tf.int32, shape=[None, None, None],
                                                 name="char_ids")

        # shape = (batch_size, max_length of sentence)
        self.word_lengths = tf.compat.v1.placeholder(tf.int32, shape=[None, None],
                                                     name="word_lengths")

        # shape = (batch size, max length of sentence in batch)
        self.labels = tf.compat.v1.placeholder(tf.int32, shape=[None, None],
                                               name="labels")

        # hyper parameters
        self.dropout = tf.compat.v1.placeholder(dtype=tf.float32, shape=[],
                                                name="dropout")
        self.lr = tf.compat.v1.placeholder(dtype=tf.float32, shape=[],
                                           name="lr")

        self.accuracy_value = tf.compat.v1.placeholder(dtype=tf.float32,
                                                       shape=[],
                                                       name="accuracy_value")

        self.loss_value = tf.compat.v1.placeholder(dtype=tf.float32,
                                                   shape=[],
                                                   name="loss_value")

    def extract_identifiers(self, sentences):
        res = []
        for sentence in sentences:
            s = []
            for word in sentence:
                s.append(word.identifier)
            res.append(s)
        return res

    def extract_labels(self, sentences):
        res = []
        for sentence in sentences:
            s = []
            for word in sentence:
                s.append(word)
            res.append(s)
        return res

    def get_feed_dict(self, words, labels=None, lr=None, dropout=None):
        """Given some data, pad it and build a feed dictionary

        Args:
            words: list of sentences. A sentence is a list of ids of a list of
                words. A word is a list of ids
            labels: list of ids
            lr: (float) learning rate
            dropout: (float) keep prob

        Returns:
            dict {placeholder: value}

        """

        # perform padding of the given data
        if self.config.use_chars:
            char_ids, word_ids = zip(*words)
            word_ids, sequence_lengths = pad_sequences(word_ids, self.config.pad_token)
            char_ids, word_lengths = pad_sequences(char_ids, pad_tok=0,
                                                   nlevels=2)
        else:
            word_ids, sequence_lengths = pad_sequences(words, self.config.pad_token)
        word_ids_word = word_ids

        word_ids = self.extract_identifiers(word_ids_word)

        # print('in creating feed dict')
        # for id in word_ids_word:
        #     for kek in id :
        #         print(kek.identifier)
        #         print(kek.word)
        #         print(self.config.embeddings[int(kek.identifier)])
        #         print(self.config.embeddings.shape)
        labels_word = labels

        if not labels == None:
            labels = self.extract_labels(labels_word)

        feed = {
            self.word_ids: word_ids,
            self.sequence_lengths: sequence_lengths
        }
        if self.config.use_large_embeddings:
            feed[self.word_embeddings_values] = self.config.embeddings
            feed[self.backoff_embeddings_change_values] = self.config.oov_embeddings
        if self.config.use_chars:
            feed[self.char_ids] = char_ids
            feed[self.word_lengths] = word_lengths

        if labels is not None:
            labels, _ = pad_sequences(labels, 0)
            feed[self.labels] = labels

        if lr is not None:
            feed[self.lr] = lr

        if dropout is not None:
            feed[self.dropout] = dropout

        return feed, sequence_lengths

    def add_word_embeddings_op(self, is_restored=False):
        """Defines self.word_embeddings

        If self.config.embeddings is not None and is a np array initialized
        with pre-trained word vectors, the word embeddings is just a look-up
        and we don't train the vectors. Otherwise, a random matrix with
        the correct shape is initialized.
        """
        with tf.compat.v1.variable_scope("words", reuse=tf.compat.v1.AUTO_REUSE, auxiliary_name_scope=not is_restored):
            # Embeddings are trained from scratch
            if self.config.embeddings is None:
                self.logger.info("WARNING: randomly initializing word vectors")
                if self.config.use_large_embeddings == True and self.config.use_pretrained == False:
                    sys.stderr.write("Using large embeddings without pre-trained embeddings is not valid")
                    sys.exit(0)
                if self.config.use_large_embeddings == True:
                    _word_embeddings = tf.compat.v1.placeholder(
                        name="word_embeddings_values",
                        dtype=tf.float32,
                        shape=[self.config.nwords, self.config.dim_word])
                else:
                    _word_embeddings = tf.compat.v1.get_variable(
                        name="_word_embeddings",
                        dtype=tf.float32,
                        shape=[self.config.nwords, self.config.dim_word])


            else:

                if self.config.use_large_embeddings == True:
                    _word_embeddings = tf.compat.v1.placeholder(
                        name="word_embeddings_values",
                        dtype=tf.float32,
                        shape=(self.config.nwords, self.config.dim_word))
                else:
                    # _word_embeddings = tf.Variable(
                    #     self.config.embeddings,
                    #     name="_word_embeddings",
                    #     dtype=tf.float32,
                    #     trainable=self.config.train_embeddings)
                    embeddings_init = tf.constant(np.array(self.config.embeddings, dtype=np.float32))
                    _word_embeddings = tf.compat.v1.get_variable(
                        "_word_embeddings",
                        initializer=embeddings_init,
                        dtype=tf.float32,
                        trainable=False)
                    # _word_embeddings.assign(self.config.embeddings)
                # check if random or OOV embeddings are added

                if self.config.oov_size > 0:
                    embeddings_oov = get_oov_embeddings(self.config)

                    if self.config.use_large_embeddings == True:
                        backoff_embeddings_change = tf.compat.v1.placeholder(
                            name="backoff_embeddings_change_values",
                            dtype=tf.float32,
                            shape=[self.config.oov_words, self.config.dim_word])
                        self.config.oov_embeddings = embeddings_oov
                    else:
                        backoff_embeddings_change = tf.Variable(
                            embeddings_oov,
                            name="backoff_embeddings_change",
                            dtype=tf.float32,
                            trainable=self.config.train_embeddings)
                    _new_word_embeddings = tf.concat([_word_embeddings, backoff_embeddings_change], axis=0)

            if self.config.use_large_embeddings:
                self.word_embeddings_values = _word_embeddings
                self.backoff_embeddings_change_values = backoff_embeddings_change
            if self.config.oov_size > 0:
                word_embeddings = tf.nn.embedding_lookup(params=_new_word_embeddings,
                                                         ids=self.word_ids, name="word_embeddings")
            else:
                word_embeddings = tf.nn.embedding_lookup(params=_word_embeddings,
                                                         ids=self.word_ids, name="word_embeddings")

        with tf.compat.v1.variable_scope("chars", reuse=tf.compat.v1.AUTO_REUSE, auxiliary_name_scope=not is_restored):
            if self.config.use_chars:
                # get char embeddings matrix
                _char_embeddings = tf.compat.v1.get_variable(
                    name="_char_embeddings",
                    dtype=tf.float32,
                    shape=[self.config.nchars, self.config.dim_char])
                char_embeddings = tf.nn.embedding_lookup(params=_char_embeddings,
                                                         ids=self.char_ids, name="char_embeddings")

                # put the time dimension on axis=1
                s = tf.shape(input=char_embeddings)
                char_embeddings = tf.reshape(char_embeddings,
                                             shape=[s[0] * s[1], s[-2], self.config.dim_char])
                word_lengths = tf.reshape(self.word_lengths, shape=[s[0] * s[1]])

                # bi lstm on chars
                cell_fw = tf.compat.v1.nn.rnn_cell.LSTMCell(self.config.hidden_size_char,
                                                            state_is_tuple=True)
                cell_bw = tf.compat.v1.nn.rnn_cell.LSTMCell(self.config.hidden_size_char,
                                                            state_is_tuple=True)
                _output = tf.compat.v1.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, char_embeddings,
                    sequence_length=word_lengths, dtype=tf.float32)

                # read and concat output
                _, ((_, output_fw), (_, output_bw)) = _output
                output = tf.concat([output_fw, output_bw], axis=-1)

                # shape = (batch size, max sentence length, char hidden size)
                output = tf.reshape(output,
                                    shape=[s[0], s[1], 2 * self.config.hidden_size_char])
                word_embeddings = tf.concat([word_embeddings, output], axis=-1)

        self.word_embeddings = tf.nn.dropout(word_embeddings, 1 - (self.dropout))

    def add_logits_op(self, is_restored=False):
        """Defines self.logits

        For each word in each sentence of the batch, it corresponds to a vector
        of scores, of dimension equal to the number of tags.
        """
        with tf.compat.v1.variable_scope("bi-lstm", reuse=tf.compat.v1.AUTO_REUSE,
                                         auxiliary_name_scope=not is_restored):
            cell_fw = tf.compat.v1.nn.rnn_cell.LSTMCell(self.config.hidden_size_lstm)
            cell_bw = tf.compat.v1.nn.rnn_cell.LSTMCell(self.config.hidden_size_lstm)
            (output_fw, output_bw), _ = tf.compat.v1.nn.bidirectional_dynamic_rnn(
                cell_fw, cell_bw, self.word_embeddings,
                sequence_length=self.sequence_lengths, dtype=tf.float32)
            output = tf.concat([output_fw, output_bw], axis=-1)
            output = tf.nn.dropout(output, 1 - (self.dropout))

        with tf.compat.v1.variable_scope("proj", reuse=tf.compat.v1.AUTO_REUSE, auxiliary_name_scope=not is_restored):
            W = tf.compat.v1.get_variable("W", dtype=tf.float32,
                                          shape=[2 * self.config.hidden_size_lstm, self.config.ntags])

            b = tf.compat.v1.get_variable("b", shape=[self.config.ntags],
                                          dtype=tf.float32, initializer=tf.compat.v1.zeros_initializer())

            nsteps = tf.shape(input=output)[1]
            output = tf.reshape(output, [-1, 2 * self.config.hidden_size_lstm])
            pred = tf.matmul(output, W) + b
            self.logits = tf.reshape(pred, [-1, nsteps, self.config.ntags])

    def add_pred_op(self):
        """Defines self.labels_pred

        This op is defined only in the case where we don't use a CRF since in
        that case we can make the prediction "in the graph" (thanks to tf
        functions in other words). With theCRF, as the inference is coded
        in python and not in pure tensroflow, we have to make the prediciton
        outside the graph.
        """
        if not self.config.use_crf:
            self.labels_pred = tf.cast(tf.argmax(input=self.logits, axis=-1),
                                       tf.int32)

    def add_loss_op(self):
        """Defines the loss"""
        if self.config.use_crf:
            log_likelihood, trans_params = tfa.text.crf.crf_log_likelihood(
                self.logits, self.labels, self.sequence_lengths)
            self.trans_params = trans_params  # need to evaluate it for decoding
            self.loss = tf.reduce_mean(input_tensor=-log_likelihood)
        else:
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.logits, labels=self.labels)
            mask = tf.sequence_mask(self.sequence_lengths)
            losses = tf.boolean_mask(tensor=losses, mask=mask)
            self.loss = tf.reduce_mean(input_tensor=losses)

        # for tensorboard
        tf.compat.v1.summary.scalar("loss", self.loss)

    def build(self, is_restored=False):
        # NER specific functions
        self.add_placeholders()
        self.add_word_embeddings_op(is_restored)
        self.add_logits_op(is_restored)
        self.add_pred_op()
        self.add_loss_op()

        # Generic functions that add training op and initialize session
        self.add_train_op(self.config.lr_method, self.lr, self.loss,
                          self.config.clip, is_restored)

        self.tf_accuracy_summary = tf.summary.scalar('accuracy', self.accuracy_value)
        self.tf_loss_summary = tf.summary.scalar('loss', self.loss_value)

        self.initialize_session()  # now self.sess is defined and vars are init

    def predict_batch(self, words):
        """
        Args:
            words: list of sentences

        Returns:
            labels_pred: list of labels for each sentence
            sequence_length

        """
        fd, sequence_lengths = self.get_feed_dict(words, dropout=1.0)

        if self.config.use_crf:
            # get tag scores and transition params of CRF
            viterbi_sequences = []
            logits, trans_params = self.sess.run(
                [self.logits, self.trans_params], feed_dict=fd)

            # iterate over the sentences because no batching in vitervi_decode
            for logit, sequence_length in zip(logits, sequence_lengths):
                logit = logit[:sequence_length]  # keep only the valid steps
                viterbi_seq, viterbi_score = tfa.text.crf.viterbi_decode(
                    logit, trans_params)
                viterbi_sequences += [viterbi_seq]

            return viterbi_sequences, sequence_lengths

        else:
            labels_pred = sess.run(self.labels_pred, feed_dict=fd)
            return labels_pred, sequence_lengths

    def run_epoch(self, train, dev, epoch):
        """Performs one complete pass over the train set and evaluate on dev

        Args:
            train: dataset that yields tuple of sentences, tags
            dev: dataset
            epoch: (int) index of the current epoch

        Returns:
            f1: (python float), score to select model on, higher is better

        """

        # summaries and statistics

        # progbar stuff for logging
        batch_size = self.config.batch_size
        nbatches = (len(train) + batch_size - 1) // batch_size
        prog = Progbar(target=nbatches)

        # iterate over dataset
        for i, (words, labels) in enumerate(minibatches(train, batch_size)):
            fd, _ = self.get_feed_dict(words, labels, self.config.lr,
                                       self.config.dropout)
            _, train_loss, summary = self.sess.run(
                [self.train_op, self.loss, self.merged], feed_dict=fd)

            # print('Current Data : ')
            # for word in words:
            #     for x in word :
            #         print(len(x))
            #         for kek in x :
            #             if(type(kek)) is Word:
            #                 print(kek.word)
            # print('-----------')
            prog.update(i + 1, [("train loss", train_loss)])

            # tensorboard
            if i % 10 == 0:
                self.file_writer.add_summary(summary, epoch * nbatches + i)
        validation_metrics = self.run_evaluate(dev)
        training_metrics = self.run_evaluate(train)
        # acc_summary = self.sess.run(self.tf_accuracy_summary, feed_dict={self.accuracy_value: metrics['acc']})
        # self.file_writer.add_summary(acc_summary, epoch)
        msg = " - ".join(["{} {:04.2f}".format(k, v)
                          for k, v in validation_metrics.items()])
        self.logger.info(msg)

        return validation_metrics, training_metrics

    def run_evaluate(self, test):
        """Evaluates performance on test set

        Args:
            test: dataset that yields tuple of (sentences, tags)

        Returns:
            metrics: (dict) metrics["acc"] = 98.4, ...

        """
        accs = []
        correct_preds, total_correct, total_preds = 0., 0., 0.
        for words, labels in minibatches(test, self.config.batch_size):
            labels_pred, sequence_lengths = self.predict_batch(words)

            for lab, lab_pred, length in zip(labels, labels_pred,
                                             sequence_lengths):
                lab = lab[:length]
                lab_pred = lab_pred[:length]
                accs += [a == b for (a, b) in zip(lab, lab_pred)]
                lab_chunks = set(get_chunks(lab, self.config.vocab_tags))
                lab_pred_chunks = set(get_chunks(lab_pred, self.config.vocab_tags))

                correct_preds += len(lab_chunks & lab_pred_chunks)
                total_preds += len(lab_pred_chunks)
                total_correct += len(lab_chunks)

        p = correct_preds / total_preds if correct_preds > 0 else 0
        r = correct_preds / total_correct if correct_preds > 0 else 0
        f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0
        acc = np.mean(accs)

        return {"Accuracy": 100 * acc, "F1 score": 100 * f1, "Precision": p, "Recall": r}

    def write_output(self, text, output, write_binary):
        if write_binary:
            output.write(bytes(text, 'utf-8'))
        else:
            output.write(text)

    def predict_test(self, test, separate="\t", output=sys.stdout, write_binary=False):

        # idx2word = load_vocab_rev(self.config.filename_words)
        idx2tag = load_vocab_rev(self.config.filename_tags)
        tag2idx = load_vocab(self.config.filename_tags)

        for words, labels in minibatches(test, self.config.batch_size):
            labels_pred, sequence_lengths = self.predict_batch(words)

            for lab, lab_pred, length, word in zip(labels, labels_pred, sequence_lengths, words):
                if self.config.use_chars:
                    for i in range(len(word[1])):

                        we = word[1][i]
                        unk = we.unknown.name
                        w = we.word + separate + we.processed_word + separate + unk
                        t = "O"

                        t = lab[i]
                        t2 = "O"
                        if lab_pred[i] in idx2tag:
                            t2 = idx2tag[lab_pred[i]]
                        if t in tag2idx:
                            self.write_output(w + separate + t + separate + t2 + "\n", output, write_binary)
                        # else:
                        # self.write_output(w + separate + t2 + "\n", output, write_binary)
                else:
                    for i in range(len(word)):
                        we = word[i]
                        unk = we.unknown.name
                        w = we.word + separate + we.processed_word + separate + unk
                        t = "O"

                        t = lab[i]
                        t2 = "O"
                        if lab_pred[i] in idx2tag:
                            t2 = idx2tag[lab_pred[i]]
                        if t in tag2idx:
                            self.write_output(w + separate + t + separate + t2 + "\n", output, write_binary)
                        # else:
                        # self.write_output(w + separate + t2 + "\n", output, write_binary)
                # self.write_output("\n", output, write_binary)

    def predict(self, words_raw):
        """Returns list of tags

        Args:
            words_raw: list of words (string), just one sentence (no batch)

        Returns:
            preds: list of tags (string), one for each word in the sentence

        """
        words = [self.config.processing_word(w) for w in words_raw]
        if type(words[0]) == tuple:
            words = zip(*words)
        pred_ids, _ = self.predict_batch([words])
        preds = [self.idx_to_tag[idx] for idx in list(pred_ids[0])]

        return preds

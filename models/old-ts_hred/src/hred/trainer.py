""" File to build the entire computation graph in tensorflow
    Modified by: Xiao Ling
    Date:        April 13th, 2017
"""

import numpy as np
import tensorflow as tf
import subprocess
import cPickle
import math


import models.ts_hred.src.sordoni.data_iterator as sordoni_data_iterator
from models.ts_hred.src.hred.hred import HRED
from models.ts_hred.src.hred.optimizer import Optimizer
from models.ts_hred.src.hred.utils import make_attention_mask


class Trainer(object):
    def __init__( self
                , idx2w_file      = None
                , train_file      = None
                , valid_file      = None

                , checkpoint_file = None
                , logs_dir        = None
 
                , vocab_size      = None
                , unk_symbol      = None
                , eoq_symbol      = None
                , eos_symbol      = None

                , restore         = None
                , n_buckets       = None
                , max_itter       = None

                , embedding_dim   = None
                , query_dim       = None
                , session_dim     = None
                , batch_size      = None
                , max_length      = None
                , seed            = None
               ):

        '''
            note they open the vocab file and throw away the count
            we change it so that count is not in vocab
        '''
        vocab_lookup_dict = cPickle.load(open(idx2w_file, 'r'))
        self.vocab_lookup_dict = vocab_lookup_dict

        self.train_data, self.valid_data = sordoni_data_iterator.get_batch_iterator(
              np.random.RandomState(seed)
    
            , {
                'eoq_sym'       : eoq_symbol,
                'eos_sym'       : eos_symbol,
                'sort_k_batches': n_buckets,
                'bs'            : batch_size,
                'train_session' : train_file,
                'seqlen'        : max_length,
                'valid_session' : valid_file
            })
    
        self.train_data.start()
        self.valid_data.start()


        '''
            xiao: adhoc bandaids
        '''
        self.RESTORE         = restore
        self.CHECKPOINT_FILE = checkpoint_file
        self.LOGS_DIR        = logs_dir
        self.MAX_ITTER       = max_itter

        vocab_size = len(self.vocab_lookup_dict)

        # vocab_size = VOCAB_SIZE
        # self.vocab_lookup_dict = read_data.read_vocab_lookup(OUR_VOCAB_FILE)

        self.hred = HRED(vocab_size=vocab_size, embedding_dim=embedding_dim, query_dim=query_dim,
                         session_dim=session_dim, decoder_dim=query_dim, output_dim=embedding_dim,
                         eoq_symbol=eoq_symbol, eos_symbol=eos_symbol, unk_symbol=unk_symbol)

        batch_size = None
        max_length = None

        self.X = tf.placeholder(tf.int64, shape=(max_length, batch_size))
        self.Y = tf.placeholder(tf.int64, shape=(max_length, batch_size))
        self.attention_mask = tf.placeholder(tf.float32, shape=(max_length, batch_size, max_length))

        self.X_sample = tf.placeholder(tf.int64, shape=(batch_size,))
        self.H_query = tf.placeholder(tf.float32, shape=(None, batch_size, self.hred.query_dim))
        self.H_session = tf.placeholder(tf.float32, shape=(batch_size, self.hred.session_dim))
        self.H_decoder = tf.placeholder(tf.float32, shape=(batch_size, self.hred.decoder_dim))

        self.logits = self.hred.step_through_session(self.X, self.attention_mask)
        self.loss = self.hred.loss(self.X, self.logits, self.Y)
        self.softmax = self.hred.softmax(self.logits)
        self.accuracy = self.hred.non_padding_accuracy(self.logits, self.Y)
        self.non_symbol_accuracy = self.hred.non_symbol_accuracy(self.logits, self.Y)

        self.session_inference = self.hred.step_through_session(
             self.X, self.attention_mask, return_softmax=True, return_last_with_hidden_states=True, reuse=True
        )
        self.step_inference = self.hred.single_step(
             self.X_sample, self.H_query, self.H_session, self.H_decoder, reuse=True
        )

        self.optimizer = Optimizer(self.loss)
        # self.summary = tf.merge_all_summaries()
        self.summary = tf.contrib.deprecated.merge_all_summaries()

        # Add ops to save and restore all the variables.
        self.saver = tf.train.Saver()

    def train(self, max_epochs=1000, max_length=50, batch_size=80):

        # Add an op to initialize the variables.
        init_op = tf.global_variables_initializer()

        with tf.Session() as tf_sess:

            if self.RESTORE:
                # Restore variables from disk.
                self.saver.restore(tf_sess, self.CHECKPOINT_FILE)
                print("Model restored.")
            else:
                tf_sess.run(init_op)

            summary_writer = tf.summary.FileWriter(self.LOGS_DIR, tf_sess.graph)

            total_loss = 0.0
            n_pred     = 0.0

            for iteration in range(self.MAX_ITTER):

                x_batch, y_batch, seq_len = self.get_batch(self.train_data)

                attention_mask = make_attention_mask(x_batch)

                if iteration % 10 == 0:
                    loss_out, _, acc_out, accuracy_non_special_symbols_out = tf_sess.run(
                        [self.loss, self.optimizer.optimize_op, self.accuracy, self.non_symbol_accuracy],
                        {self.X: x_batch, self.Y: y_batch, self.attention_mask: attention_mask}
                    )

                    # Accumulative cost, like in hred-qs
                    total_loss_tmp = total_loss + loss_out
                    n_pred_tmp = n_pred + seq_len * batch_size
                    cost = total_loss_tmp / n_pred_tmp

                    print("Step %d - Cost: %f   Loss: %f   Accuracy: %f   Accuracy (no symbols): %f  Length: %d" %
                          (iteration, cost, loss_out, acc_out, accuracy_non_special_symbols_out, seq_len))

                else:
                    loss_out, _ = tf_sess.run(
                        [self.loss, self.optimizer.optimize_op],
                        {self.X: x_batch, self.Y: y_batch, self.attention_mask: attention_mask}
                    )

                    # Accumulative cost, like in hred-qs
                    total_loss_tmp = total_loss + loss_out
                    n_pred_tmp = n_pred + seq_len * batch_size
                    cost = total_loss_tmp / n_pred_tmp

                if math.isnan(loss_out) or math.isnan(cost) or cost > 100:
                    print("Found inconsistent results, restoring model...")
                    self.saver.restore(tf_sess, self.CHECKPOINT_FILE)
                else:
                    total_loss = total_loss_tmp
                    n_pred = n_pred_tmp

                    if iteration % 25 == 0:
                        print("Saving..")
                        self.save_model(tf_sess, loss_out)

                # Sumerize
                if iteration % 10 == 0:
                #if iteration % 100 == 0:
                    summary_str = tf_sess.run(self.summary, {self.X: x_batch, self.Y: y_batch, self.attention_mask: attention_mask})
                    summary_writer.add_summary(summary_str, iteration)
                    summary_writer.flush()

                if iteration % 250 == 0:
                     # self.sample(tf_sess)
                     self.sample_beam(tf_sess)

                iteration += 1

    def sample(self, sess, max_sample_length=30, num_of_samples=3, min_queries = 3):

        for i in range(num_of_samples):

            x_batch, _, seq_len = self.get_batch(self.valid_data)
            input_x = np.expand_dims(x_batch[:-(seq_len / 2), 1], axis=1)

            softmax_out, hidden_query, hidden_session, hidden_decoder = sess.run(
                self.session_inference,
                feed_dict={self.X: input_x}
            )

            queries_accepted = 0
            arg_sort = np.argsort(softmax_out, axis=1)[0][::-1]

            # Ignore UNK and EOS (for the first min_queries)
            arg_sort_i = 0
            while arg_sort[arg_sort_i] == self.hred.unk_symbol or (
                            arg_sort[arg_sort_i] == self.hred.eos_symbol and queries_accepted < min_queries):
                arg_sort_i += 1
            x = arg_sort[arg_sort_i]

            if x == self.hred.eoq_symbol:
                queries_accepted += 1

            result = [x]
            i = 0

            while x != self.hred.eos_symbol and i < max_sample_length:
                softmax_out, hidden_query, hidden_session, hidden_decoder = sess.run(
                    self.step_inference,
                    {self.X_sample: [x], self.H_query: hidden_query, self.H_session: hidden_session,
                     self.H_decoder: hidden_decoder}
                )
                print("INFO -- Sample hidden states", tf.shape(hidden_query))
                arg_sort = np.argsort(softmax_out, axis=1)[0][::-1]

                # Ignore UNK and EOS (for the first min_queries)
                arg_sort_i = 0
                while arg_sort[arg_sort_i] == self.hred.unk_symbol or (
                                arg_sort[arg_sort_i] == self.hred.eos_symbol and queries_accepted < min_queries):
                    arg_sort_i += 1
                x = arg_sort[arg_sort_i]

                if x == self.hred.eoq_symbol:
                    queries_accepted += 1

                result += [x]
                i += 1

            input_x = np.array(input_x).flatten()
            result = np.array(result).flatten()
            print('Sample input:  %s' % (' '.join([self.vocab_lookup_dict.get(x, '?') for x in input_x]),))
            print('Sample output: %s' % (' '.join([self.vocab_lookup_dict.get(x, '?') for x in result])))
            print('')

    def sample_beam(self, sess, max_steps=30, num_of_samples=3, beam_size=10, min_queries=2):

        for step in range(num_of_samples):

            x_batch, _, seq_len = self.get_batch(self.valid_data)
            input_x = np.expand_dims(x_batch[:-(seq_len / 2), 1], axis=1)

            attention_mask = make_attention_mask(input_x)

            softmax_out, hidden_query, hidden_session, hidden_decoder = sess.run(
                self.session_inference,
                feed_dict={self.X: input_x, self.attention_mask: attention_mask}
            )

            current_beam_size = beam_size
            current_hypotheses = []
            final_hypotheses = []

            # Reverse arg sort (highest prob above)
            arg_sort = np.argsort(softmax_out, axis=1)[0][::-1]
            arg_sort_i = 0

            # create original current_hypotheses
            while len(current_hypotheses) < current_beam_size:
                # Ignore UNK and EOS (for the first min_queries)
                

                while arg_sort[arg_sort_i] == self.hred.unk_symbol \
                   or arg_sort[arg_sort_i] == self.hred.eos_symbol:
                    arg_sort_i += 1

                x = arg_sort[arg_sort_i]
                arg_sort_i += 1

                queries_accepted = 1 if x == self.hred.eoq_symbol else 0
                result = [x]
                prob = softmax_out[0][x]
                current_hypotheses += [
                    (prob, x, result, hidden_query, hidden_session, hidden_decoder, queries_accepted)]

            # Create hypotheses per step
            step = 0
            while current_beam_size > 0 and step <= max_steps:

                step += 1
                next_hypotheses = []

                # expand all hypotheses
                for prob, x, result, hidden_query, hidden_session, hidden_decoder, queries_accepted in current_hypotheses:

                    input_for_mask = np.concatenate((input_x, np.expand_dims(np.array(result), axis=1)), axis=0)
                    attention_mask = make_attention_mask(input_for_mask)

                    softmax_out, hidden_query, hidden_session, hidden_decoder = sess.run(
                        self.step_inference,
                        {self.X_sample: [x], self.H_query: hidden_query, self.H_session: hidden_session,
                         self.H_decoder: hidden_decoder, self.attention_mask: attention_mask}
                    )

                    # Reverse arg sort (highest prob above)
                    arg_sort = np.argsort(softmax_out, axis=1)[0][::-1]
                    arg_sort_i = 0

                    expanded_hypotheses = []

                    # create hypothesis
                    while len(expanded_hypotheses) < current_beam_size:

                        # Ignore UNK and EOS (for the first min_queries)
                        while arg_sort[arg_sort_i] == self.hred.unk_symbol or (
                                        arg_sort[
                                            arg_sort_i] == self.hred.eos_symbol and queries_accepted < min_queries):
                            arg_sort_i += 1

                        new_x = arg_sort[arg_sort_i]
                        arg_sort_i += 1

                        new_queries_accepted = queries_accepted + 1 if x == self.hred.eoq_symbol else queries_accepted
                        new_result = result + [new_x]
                        new_prob = softmax_out[0][new_x] * prob

                        expanded_hypotheses += [(new_prob, new_x, new_result, hidden_query, hidden_session,
                                                 hidden_decoder, new_queries_accepted)]

                    next_hypotheses += expanded_hypotheses

                # sort hypotheses and remove the least likely
                next_hypotheses = sorted(next_hypotheses, key=lambda x: x[0], reverse=True)[:current_beam_size]
                current_hypotheses = []

                for hypothesis in next_hypotheses:
                    _, x, _, _, _, _, queries_accepted = hypothesis

                    if x == self.hred.eos_symbol:
                        final_hypotheses += [hypothesis]
                        current_beam_size -= 1
                    else:
                        current_hypotheses += [hypothesis]

            final_hypotheses += current_hypotheses

            input_x = np.array(input_x).flatten()
            print('Sample input:  %s' % (' '.join([self.vocab_lookup_dict.get(x, '?') for x in input_x]),))

            for _, _, result, _, _, _, _ in final_hypotheses:
                result = np.array(result).flatten()
                print('Sample output: %s' % (' '.join([self.vocab_lookup_dict.get(x, '?') for x in result])))

            print('')

    def save_model(self, sess, loss_out):
        if not math.isnan(loss_out):
            # Save the variables to disk.
            save_path = self.saver.save(sess, self.CHECKPOINT_FILE)
            print("Model saved in file: %s" % save_path)

    def get_batch(self, train_data):

        # The training is done with a trick. We append a 
        # special </q> at the beginning of the dialog
        # so that we can predict also the first sent in the 
        # dialog starting from the dialog beginning token (</q>).

        data        = train_data.next()
        seq_len     = data['max_length']
        prepend     = np.ones((1, data['x'].shape[1]))
        x_data_full = np.concatenate((prepend, data['x']))
        x_batch     = x_data_full[:seq_len]
        y_batch     = x_data_full[1:seq_len + 1]

        # x_batch = np.transpose(np.asarray(x_batch))
        # y_batch = np.transpose(np.asarray(y_batch))

        return x_batch, y_batch, seq_len


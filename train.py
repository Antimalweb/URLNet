import re 
import time 
import os
import pickle
import numpy as np 
from bisect import bisect_left  
import tensorflow as tf 
from tensorflow.contrib import learn 
from tflearn.data_utils import to_categorical, pad_sequences 
from TextCNN import * 
from utils import * 

tf.flags.DEFINE_integer("MAX_LENGTH_WORDS", 200, "Max length of url in words") 
tf.flags.DEFINE_integer("MAX_LENGTH_CHARS", 200, "Max length of url in chars") 
tf.flags.DEFINE_integer("MAX_LENGTH_SUBWORDS",20, "Max length of word in ngrams") 
tf.flags.DEFINE_integer("MIN_WORD_FREQ", 1, "Minimum frequency of word to build vocabulary")  
tf.flags.DEFINE_integer("EMB_DIM", 32, "embedding dimension size") 
tf.flags.DEFINE_integer("NB_EPOCHS",5, "number of training epochs") 
tf.flags.DEFINE_integer("BATCH_SIZE", 128, "Size of a training batch") 
tf.flags.DEFINE_float("DEV_PERCENTAGE", 0.001, "portion of training used for dev") 
tf.flags.DEFINE_string("FILE_DIR","train_10000.txt", "directory of the training data") 
tf.flags.DEFINE_string("OUTPUT_DIR", "runs/10000/", "directory o the output model")  
tf.flags.DEFINE_integer("PRINT_EVERY", 50, "print training result every this number of steps") 
tf.flags.DEFINE_integer("EVAL_EVERY", 500, "evaluate the model every this number of steps") 
tf.flags.DEFINE_integer("CHECKPOINT_EVERY", 500, "Save a model every this number of steps") 
tf.flags.DEFINE_float("L2_REG_LAMBDA", 0.0, "l2 lambda for regularization") 
tf.flags.DEFINE_string("FILTER_SIZES", "3,4,5,6", "filter sizes of the convolution layer")
tf.flags.DEFINE_float("LR", 0.001, "learning rate of the optimizer") 
tf.flags.DEFINE_integer("EMB_MODE", 1, "1: charCNN, 2: wordCNN, 3: char + wordCNN, 4: char-level wordCNN, 5: char + char-level wordCNN")  
tf.flags.DEFINE_integer("DELIMIT_MODE", 1, "0: delimit by special chars, 1: delimit by special chars + each char as a word")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags() 
print("\nParameters:") 
for attr, value in FLAGS.__flags.items(): 
    print("{}={}".format(attr, value))
print("") 

urls, labels = read_data(FLAGS.FILE_DIR) 

high_freq_words = None
if FLAGS.MIN_WORD_FREQ > 0: 
    x1, word_reverse_dict = get_word_vocab(urls, FLAGS.MAX_LENGTH_WORDS, FLAGS.MIN_WORD_FREQ) 
    high_freq_words = sorted(list(word_reverse_dict.values()))
    print("Number of words with freq >={}: {}".format(FLAGS.MIN_WORD_FREQ, len(high_freq_words)))  

x, word_reverse_dict = get_word_vocab(urls, FLAGS.MAX_LENGTH_WORDS) 
word_x = get_words(x, word_reverse_dict, FLAGS.DELIMIT_MODE, urls)
ngramed_id_x, ngrams_dict, worded_id_x, words_dict = ngram_id_x(word_x, FLAGS.MAX_LENGTH_SUBWORDS, high_freq_words)
#ngramed_id_x, ngrams_dict, worded_id_x, words_dict = ngram_id_x(word_x, FLAGS.WORD_NGRAM_LEN, FLAGS.CHAR_NGRAM_LEN, FLAGS.MAX_LENGTH_SUBWORDS, 3, high_freq_words)  

chars_dict = ngrams_dict
chared_id_x = char_id_x(urls, chars_dict, FLAGS.MAX_LENGTH_CHARS)

pos_x = []
neg_x = []
for i in range(len(labels)):
    label = labels[i] 
    if label == 1: 
        pos_x.append(i)
    else: 
        neg_x.append(i)
print("Overall Mal/Ben split: {}/{}".format(len(pos_x), len(neg_x)))
pos_x = np.array(pos_x) 
neg_x = np.array(neg_x) 

x_train, y_train, x_test, y_test = prep_train_test(pos_x, neg_x, FLAGS.DEV_PERCENTAGE)

x_train_char = get_ngramed_id_x(x_train, ngramed_id_x) 
x_test_char = get_ngramed_id_x(x_test, ngramed_id_x) 

x_train_word = get_ngramed_id_x(x_train, worded_id_x) 
x_test_word = get_ngramed_id_x(x_test, worded_id_x)  

x_train_char_seq = get_ngramed_id_x(x_train, chared_id_x)
x_test_char_seq = get_ngramed_id_x(x_test, chared_id_x)


###################################### Training #########################################################

def train_dev_step(x, y, emb_mode, is_train=True):
    if is_train: 
        p = 0.5
    else: 
        p = 1.0
    if emb_mode == 1: 
        feed_dict = {
            cnn.input_x_char_seq: x[0],
            cnn.input_y: y,
            cnn.dropout_keep_prob: p}  
    elif emb_mode == 2: 
        feed_dict = {
            cnn.input_x_word: x[0],
            cnn.input_y: y,
            cnn.dropout_keep_prob: p}
    elif emb_mode == 3: 
        feed_dict = {
            cnn.input_x_char_seq: x[0],
            cnn.input_x_word: x[1],
            cnn.input_y: y,
            cnn.dropout_keep_prob: p}
    elif emb_mode == 4: 
        feed_dict = {
            cnn.input_x_word: x[0],
            cnn.input_x_char: x[1],
            cnn.input_x_char_pad_idx: x[2],
            cnn.input_y: y,
            cnn.dropout_keep_prob: p}
    elif emb_mode == 5:  
        feed_dict = {
            cnn.input_x_char_seq: x[0],
            cnn.input_x_word: x[1],
            cnn.input_x_char: x[2],
            cnn.input_x_char_pad_idx: x[3],
            cnn.input_y: y,
            cnn.dropout_keep_prob: p}
    if is_train:
        _, step, loss, acc = sess.run([train_op, global_step, cnn.loss, cnn.accuracy], feed_dict)
    else: 
        step, loss, acc = sess.run([global_step, cnn.loss, cnn.accuracy], feed_dict)
    return step, loss, acc

with tf.Graph().as_default(): 
    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False) 
    session_conf.gpu_options.allow_growth=True 
    sess = tf.Session(config=session_conf) 

    with sess.as_default():  
        cnn = TextCNN(
                char_ngram_vocab_size = len(ngrams_dict)+1, 
                word_ngram_vocab_size = len(words_dict)+1,
                char_vocab_size = len(chars_dict)+1,
                embedding_size=FLAGS.EMB_DIM,
                word_seq_len=FLAGS.MAX_LENGTH_WORDS,
                char_seq_len=FLAGS.MAX_LENGTH_CHARS,
                l2_reg_lambda=FLAGS.L2_REG_LAMBDA,
                mode=FLAGS.EMB_MODE,
                filter_sizes=list(map(int, FLAGS.FILTER_SIZES.split(","))))

        global_step = tf.Variable(0, name="global_step", trainable=False) 
        optimizer = tf.train.AdamOptimizer(FLAGS.LR) 
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step = global_step) 
        
        print("Writing to {}\n".format(FLAGS.OUTPUT_DIR))
        if not os.path.exists(FLAGS.OUTPUT_DIR): 
            os.makedirs(FLAGS.OUTPUT_DIR)
        
        # Save dictionary files 
        ngrams_dict_dir = FLAGS.OUTPUT_DIR + "ngrams_dict.p"
        pickle.dump(ngrams_dict, open(ngrams_dict_dir,"wb"))  
        words_dict_dir = FLAGS.OUTPUT_DIR + "words_dict.p"
        pickle.dump(words_dict, open(words_dict_dir, "wb"))
        chars_dict_dir = FLAGS.OUTPUT_DIR + "chars_dict.p"
        pickle.dump(chars_dict, open(chars_dict_dir, "wb"))

        checkpoint_dir = FLAGS.OUTPUT_DIR + "checkpoints/" 
        if not os.path.exists(checkpoint_dir): 
            os.makedirs(checkpoint_dir) 
        checkpoint_prefix = checkpoint_dir + "model"
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=5) 
        
        sess.run(tf.global_variables_initializer())

        if FLAGS.EMB_MODE == 1: 
            batch_data = list(zip(x_train_char_seq, y_train))
        elif FLAGS.EMB_MODE == 2: 
            batch_data = list(zip(x_train_word, y_train))
        elif FLAGS.EMB_MODE == 3: 
            batch_data = list(zip(x_train_char_seq, x_train_word, y_train))
        elif FLAGS.EMB_MODE == 4:
            batch_data = list(zip(x_train_char, x_train_word, y_train))
        elif FLAGS.EMB_MODE == 5: 
            batch_data = list(zip(x_train_char, x_train_word, x_train_char_seq, y_train))
        batches = batch_iter(batch_data, FLAGS.BATCH_SIZE, FLAGS.NB_EPOCHS)
        
        x_test = []
        if FLAGS.EMB_MODE == 1 or FLAGS.EMB_MODE == 3 or FLAGS.EMB_MODE == 5: 
            x_test_char_seq = pad_seq_in_word(x_test_char_seq, FLAGS.MAX_LENGTH_CHARS) 
            x_test.append(x_test_char_seq)
        if FLAGS.EMB_MODE == 2 or FLAGS.EMB_MODE == 3 or FLAGS.EMB_MODE == 4 or FLAGS.EMB_MODE == 5: 
            x_test_word = pad_seq_in_word(x_test_word, FLAGS.MAX_LENGTH_WORDS) 
            x_test.append(x_test_word)
        if FLAGS.EMB_MODE == 4 or FLAGS.EMB_MODE == 5: 
            x_test_char, x_test_char_pad_idx = pad_seq(x_test_char, FLAGS.MAX_LENGTH_WORDS, FLAGS.MAX_LENGTH_SUBWORDS, FLAGS.EMB_DIM)
            x_test.extend([x_test_char, x_test_char_pad_idx])

        min_dev_loss = float('Inf') 
        nb_batches_per_epoch = int(len(batch_data)/FLAGS.BATCH_SIZE)
        if len(batch_data)%FLAGS.BATCH_SIZE != 0: 
            nb_batches_per_epoch += 1
        nb_batches = int(nb_batches_per_epoch * FLAGS.NB_EPOCHS)
        print("Number of baches in total: {}".format(nb_batches))
        print("Number of batches per epoch: {}".format(nb_batches_per_epoch))
        for idx, batch in enumerate(batches): 
            if FLAGS.EMB_MODE == 1: 
                x_char_seq, y_batch = zip(*batch) 
            elif FLAGS.EMB_MODE == 2: 
                x_word, y_batch = zip(*batch) 
            elif FLAGS.EMB_MODE == 3: 
                x_char_seq, x_word, y_batch = zip(*batch) 
            elif FLAGS.EMB_MODE == 4: 
                x_char, x_word, y_batch = zip(*batch)
            elif FLAGS.EMB_MODE == 5: 
                x_char, x_word, x_char_seq, y_batch = zip(*batch)            

            x_batch = []    
            if FLAGS.EMB_MODE == 1 or FLAGS.EMB_MODE == 3 or FLAGS.EMB_MODE == 5: 
                x_char_seq = pad_seq_in_word(x_char_seq, FLAGS.MAX_LENGTH_CHARS) 
                x_batch.append(x_char_seq)
            if FLAGS.EMB_MODE == 2 or FLAGS.EMB_MODE == 3 or FLAGS.EMB_MODE == 4 or FLAGS.EMB_MODE == 5: 
                x_word = pad_seq_in_word(x_word, FLAGS.MAX_LENGTH_WORDS) 
                x_batch.append(x_word)
            if FLAGS.EMB_MODE == 4 or FLAGS.EMB_MODE == 5: 
                x_char, x_char_pad_idx = pad_seq(x_char, FLAGS.MAX_LENGTH_WORDS, FLAGS.MAX_LENGTH_SUBWORDS, FLAGS.EMB_DIM)
                x_batch.extend([x_char, x_char_pad_idx])
            step, loss, acc = train_dev_step(x_batch, y_batch, emb_mode=FLAGS.EMB_MODE, is_train=True)                      

            if step % FLAGS.PRINT_EVERY == 0: 
                print("step {}, loss {}, acc {}".format(step, loss, acc)) 
            if step % FLAGS.EVAL_EVERY == 0: 
                print("\nEvaluation") 
                step, dev_loss, dev_acc = train_dev_step(x_test, y_test, emb_mode=FLAGS.EMB_MODE, is_train=False) 
                print("step {}, loss {}, acc {}".format(step, dev_loss, dev_acc))
                if step % FLAGS.CHECKPOINT_EVERY == 0 or idx == (nb_batches-1): 
                    if dev_loss < min_dev_loss: 
                        path = saver.save(sess, checkpoint_prefix, global_step = step) 
                        print("Dev loss improved: {} -> {}".format(min_dev_loss, dev_loss))
                        print("Saved model checkpoint to {}\n".format(path))
                        min_dev_loss = dev_loss 
                    else: 
                        print("Dev loss did not improve: {} -> {}".format(min_dev_loss, dev_loss))
            

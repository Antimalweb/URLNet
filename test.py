from utils import * 
import pickle 
import time 
import numpy as np 
import pickle 
import tensorflow as tf 
from tensorflow.contrib import learn 
from tflearn.data_utils import to_categorical, pad_sequences

tf.flags.DEFINE_integer("MAX_LENGTH_WORDS", 200, "Max length of url in words") 
tf.flags.DEFINE_integer("MAX_LENGTH_CHARS", 200, "Max length of url in chars") 
tf.flags.DEFINE_integer("MAX_LENGTH_SUBWORDS", 20, "Max length of word in ngrams") 
tf.flags.DEFINE_string("FILE_DIR", "test_1000.txt", "directory of the test data")
tf.flags.DEFINE_string("NGRAM_DICT_DIR", "runs/10000/ngrams_dict.p", "directory of the ngram dictionary") 
tf.flags.DEFINE_string("WORD_DICT_DIR", "runs/10000/words_dict.p", "directory of the word dictionary")  
tf.flags.DEFINE_string("CHAR_DICT_DIR", "runs/10000/chars_dict.p", "directory of the char dictionary") 
tf.flags.DEFINE_string("CHECKPOINT_DIR", "runs/10000/checkpoints/", "directonry of the learned model")
tf.flags.DEFINE_string("OUTPUT_DIR", "runs/10000/train_10000_test_1000.txt", "directory to save the test results") 
tf.flags.DEFINE_integer("DELIMIT_MODE", 1, "0: delimit by special chars, 1: delimit by special chars + each special char as a word") 
tf.flags.DEFINE_integer("EMB_MODE", 1, "1: charCNN, 2: wordCNN, 3: char + wordCNN, 4: char-level wordCNN, 5: char + char-level wordCNN")  
tf.flags.DEFINE_integer("EMB_DIM", 32, "embedding dimension length") 
tf.flags.DEFINE_integer("BATCH_SIZE", 128, "Size of a test batch") 

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags() 
for attr, value in FLAGS.__flags.items(): 
    print("{}={}".format(attr, value))
print("") 

urls, labels = read_data(FLAGS.FILE_DIR) 
 
x, word_reverse_dict = get_word_vocab(urls, FLAGS.MAX_LENGTH_WORDS) 
word_x = get_words(x, word_reverse_dict, FLAGS.DELIMIT_MODE, urls) 

ngram_dict = pickle.load(open(FLAGS.NGRAM_DICT_DIR, "rb")) 
print("Size of ngram vocabulary (train): {}".format(len(ngram_dict)))
word_dict = pickle.load(open(FLAGS.WORD_DICT_DIR, "rb"))
print("size of word vocabulary (train): {}".format(len(word_dict)))
ngramed_id_x, worded_id_x = ngram_id_x_from_dict(word_x, FLAGS.MAX_LENGTH_SUBWORDS, ngram_dict, word_dict) 
chars_dict = pickle.load(open(FLAGS.CHAR_DICT_DIR, "rb"))          
chared_id_x = char_id_x(urls, chars_dict, FLAGS.MAX_LENGTH_CHARS)    

print("Number of testing urls: {}".format(len(labels)))

######################## EVALUATION ########################### 

def test_step(x, emb_mode):
    p = 1.0
    if emb_mode == 1: 
        feed_dict = {
            input_x_char_seq: x[0],
            dropout_keep_prob: p}  
    elif emb_mode == 2: 
        feed_dict = {
            input_x_word: x[0],
            dropout_keep_prob: p}
    elif emb_mode == 3: 
        feed_dict = {
            input_x_char_seq: x[0],
            input_x_word: x[1],
            dropout_keep_prob: p}
    elif emb_mode == 4: 
        feed_dict = {
            input_x_word: x[0],
            input_x_char: x[1],
            input_x_char_pad_idx: x[2],
            dropout_keep_prob: p}
    elif emb_mode == 5:  
        feed_dict = {
            input_x_char_seq: x[0],
            input_x_word: x[1],
            input_x_char: x[2],
            input_x_char_pad_idx: x[3],
            dropout_keep_prob: p}
    preds, s = sess.run([predictions, scores], feed_dict)
    return preds, s

checkpoint_file = tf.train.latest_checkpoint(FLAGS.CHECKPOINT_DIR)
graph = tf.Graph() 
with graph.as_default(): 
    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_conf.gpu_options.allow_growth=True 
    sess = tf.Session(config=session_conf)
    with sess.as_default(): 
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file) 
        
        if  FLAGS.EMB_MODE == 1 or FLAGS.EMB_MODE == 3 or FLAGS.EMB_MODE == 5: 
            input_x_char_seq = graph.get_operation_by_name("input_x_char_seq").outputs[0]
        if FLAGS.EMB_MODE == 2 or FLAGS.EMB_MODE == 3 or FLAGS.EMB_MODE == 4 or FLAGS.EMB_MODE == 5:
            input_x_word = graph.get_operation_by_name("input_x_word").outputs[0]
        if FLAGS.EMB_MODE == 4 or FLAGS.EMB_MODE == 5:  
            input_x_char = graph.get_operation_by_name("input_x_char").outputs[0]
            input_x_char_pad_idx = graph.get_operation_by_name("input_x_char_pad_idx").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0] 

        predictions = graph.get_operation_by_name("output/predictions").outputs[0]
        scores = graph.get_operation_by_name("output/scores").outputs[0]
         
        if FLAGS.EMB_MODE == 1: 
            batches = batch_iter(list(chared_id_x), FLAGS.BATCH_SIZE, 1, shuffle=False) 
        elif FLAGS.EMB_MODE == 2: 
            batches = batch_iter(list(worded_id_x), FLAGS.BATCH_SIZE, 1, shuffle=False) 
        elif FLAGS.EMB_MODE == 3: 
            batches = batch_iter(list(zip(chared_id_x, worded_id_x)), FLAGS.BATCH_SIZE, 1, shuffle=False)
        elif FLAGS.EMB_MODE == 4: 
            batches = batch_iter(list(zip(ngramed_id_x, worded_id_x)), FLAGS.BATCH_SIZE, 1, shuffle=False)
        elif FLAGS.EMB_MODE == 5: 
            batches = batch_iter(list(zip(ngramed_id_x, worded_id_x, chared_id_x)), FLAGS.BATCH_SIZE, 1, shuffle=False)    
        all_predictions = []
        all_scores = []
        
        counter = 0
        for batch in batches:
            if counter%1000 == 0: 
                print("Processing #batch {}".format(counter))
            counter += 1

            if FLAGS.EMB_MODE == 1: 
                x_char_seq = batch 
            elif FLAGS.EMB_MODE == 2: 
                x_word = batch 
            elif FLAGS.EMB_MODE == 3: 
                x_char_seq, x_word = zip(*batch) 
            elif FLAGS.EMB_MODE == 4: 
                x_char, x_word = zip(*batch)
            elif FLAGS.EMB_MODE == 5: 
                x_char, x_word, x_char_seq = zip(*batch)            

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
            
            batch_predictions, batch_scores = test_step(x_batch, FLAGS.EMB_MODE)            
            all_predictions = np.concatenate([all_predictions, batch_predictions]) 
            all_scores.extend(batch_scores) 

if labels is not None: 
    correct_preds = float(sum(all_predictions == labels)) 
    print("Total number of test examples: {}".format(len(labels)))
    print("Accuracy: {}".format(correct_preds/float(len(labels))))

save_test_result(labels, all_predictions, all_scores, FLAGS.OUTPUT_DIR) 

import time 
import os 
import numpy as np
from collections import defaultdict
from bisect import bisect_left 
import tensorflow as tf 
from tflearn.data_utils import to_categorical 
from tensorflow.contrib import learn 

def read_data(file_dir): 
    with open(file_dir) as file: 
        urls = []
        labels = []
        for line in file.readlines(): 
            items = line.split('\t') 
            label = int(items[0]) 
            if label == 1: 
                labels.append(1) 
            else: 
                labels.append(0) 
            url = items[1][:-1]
            urls.append(url) 
    return urls, labels 

def split_url(line, part):
    if line.startswith("http://"):
        line=line[7:]
    if line.startswith("https://"):
        line=line[8:]
    if line.startswith("ftp://"):
        line=line[6:]
    if line.startswith("www."):
        line = line[4:]
    slash_pos = line.find('/')
    if slash_pos > 0 and slash_pos < len(line)-1: # line = "fsdfsdf/sdfsdfsd"
        primarydomain = line[:slash_pos]
        path_argument = line[slash_pos+1:]
        path_argument_tokens = path_argument.split('/')
        pathtoken = "/".join(path_argument_tokens[:-1])
        last_pathtoken = path_argument_tokens[-1]
        if len(path_argument_tokens) > 2 and last_pathtoken == '':
            pathtoken = "/".join(path_argument_tokens[:-2])
            last_pathtoken = path_argument_tokens[-2]
        question_pos = last_pathtoken.find('?')
        if question_pos != -1:
            argument = last_pathtoken[question_pos+1:]
            pathtoken = pathtoken + "/" + last_pathtoken[:question_pos]     
        else:
            argument = ""
            pathtoken = pathtoken + "/" + last_pathtoken          
        last_slash_pos = pathtoken.rfind('/')
        sub_dir = pathtoken[:last_slash_pos]
        filename = pathtoken[last_slash_pos+1:]
        file_last_dot_pos = filename.rfind('.')
        if file_last_dot_pos != -1:
            file_extension = filename[file_last_dot_pos+1:]
            filename = filename[:file_last_dot_pos]
        else:
            file_extension = "" 
    elif slash_pos == 0:    # line = "/fsdfsdfsdfsdfsd"
        primarydomain = line[1:]
        pathtoken = ""
        argument = ""
        sub_dir = ""
        filename = ""
        file_extension = ""
    elif slash_pos == len(line)-1:   # line = "fsdfsdfsdfsdfsd/"
        primarydomain = line[:-1]
        pathtoken = ""
        argument = ""
        sub_dir = ""     
        filename = ""
        file_extension = ""
    else:      # line = "fsdfsdfsdfsdfsd"
        primarydomain = line
        pathtoken = ""
        argument = ""
        sub_dir = "" 
        filename = ""
        file_extension = ""
    if part == 'pd':
        return primarydomain
    elif part == 'path':
        return pathtoken
    elif part == 'argument': 
        return argument 
    elif part == 'sub_dir': 
        return sub_dir 
    elif part == 'filename': 
        return filename 
    elif part == 'fe': 
        return file_extension
    elif part == 'others': 
        if len(argument) > 0: 
            return pathtoken + '?' +  argument 
        else: 
            return pathtoken 
    else:
        return primarydomain, pathtoken, argument, sub_dir, filename, file_extension

def get_word_vocab(urls, max_length_words, min_word_freq=0): 
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_length_words, min_frequency=min_word_freq) 
    start = time.time() 
    x = np.array(list(vocab_processor.fit_transform(urls)))
    print("Finished build vocabulary and mapping to x in {}".format(time.time() - start))
    vocab_dict = vocab_processor.vocabulary_._mapping
    reverse_dict = dict(zip(vocab_dict.values(), vocab_dict.keys()))
    print("Size of word vocabulary: {}".format(len(reverse_dict)))
    return x, reverse_dict 

def get_words(x, reverse_dict, delimit_mode, urls=None): 
    processed_x = []
    if delimit_mode == 0: 
        for url in x: 
            words = []
            for word_id in url: 
                if word_id != 0: 
                    words.append(reverse_dict[word_id])
                else: 
                    break
            processed_x.append(words) 
    elif delimit_mode == 1: 
        for i in range(x.shape[0]):
            word_url = x[i]
            raw_url = urls[i]
            words = []
            for w in range(len(word_url)): 
                word_id = word_url[w]
                if word_id == 0: 
                    words.extend(list(raw_url))
                    break
                else: 
                    word = reverse_dict[word_id]
                    idx = raw_url.index(word) 
                    special_chars = list(raw_url[0:idx])
                    words.extend(special_chars) 
                    words.append(word) 
                    raw_url = raw_url[idx+len(word):]
                    if w == len(word_url) - 1: 
                        words.extend(list(raw_url))
            processed_x.append(words)
    return processed_x 

def get_char_ngrams(ngram_len, word): 
    word = "<" + word + ">" 
    chars = list(word) 
    begin_idx = 0
    ngrams = []
    while (begin_idx + ngram_len) <= len(chars): 
        end_idx = begin_idx + ngram_len 
        ngrams.append("".join(chars[begin_idx:end_idx])) 
        begin_idx += 1 
    return ngrams 

def char_id_x(urls, char_dict, max_len_chars): 
    chared_id_x = []
    for url in urls: 
        url = list(url) 
        url_in_char_id = []
        l = min(len(url), max_len_chars)
        for i in range(l): 
            c = url[i] 
            try:
                c_id = char_dict[c] 
            except KeyError:
                c_id = 0
            url_in_char_id.append(c_id) 
        chared_id_x.append(url_in_char_id) 
    return chared_id_x 
    
def ngram_id_x(word_x, max_len_subwords, high_freq_words=None):   
    char_ngram_len = 1
    all_ngrams = set() 
    ngramed_x = []
    all_words = set() 
    worded_x = []
    counter = 0
    for url in word_x:
        if counter % 100000 == 0: 
            print("Processing #url {}".format(counter))
        counter += 1  
        url_in_ngrams = []
        url_in_words = []
        words = url
        for word in words:
            ngrams = get_char_ngrams(char_ngram_len, word) 
            if (len(ngrams) > max_len_subwords) or \
                (high_freq_words is not None and len(word)>1 and not is_in(high_freq_words, word)):  
                all_ngrams.update(ngrams[:max_len_subwords])
                url_in_ngrams.append(ngrams[:max_len_subwords]) 
                all_words.add("<UNKNOWN>")
                url_in_words.append("<UNKNOWN>")
            else:     
                all_ngrams.update(ngrams)
                url_in_ngrams.append(ngrams) 
                all_words.add(word) 
                url_in_words.append(word) 
        ngramed_x.append(url_in_ngrams)
        worded_x.append(url_in_words) 

    all_ngrams = list(all_ngrams) 
    ngrams_dict = dict()
    for i in range(len(all_ngrams)):  
        ngrams_dict[all_ngrams[i]] = i+1 # ngram id=0 is for padding ngram   
    print("Size of ngram vocabulary: {}".format(len(ngrams_dict))) 
    all_words = list(all_words) 
    words_dict = dict() 
    for i in range(len(all_words)): 
        words_dict[all_words[i]] = i+1 #word id=0 for padding word 
    print("Size of word vocabulary: {}".format(len(words_dict)))
    print("Index of <UNKNOWN> word: {}".format(words_dict["<UNKNOWN>"]))        

    ngramed_id_x = []
    for ngramed_url in ngramed_x: 
        url_in_ngrams = []
        for ngramed_word in ngramed_url: 
            ngram_ids = [ngrams_dict[x] for x in ngramed_word] 
            url_in_ngrams.append(ngram_ids) 
        ngramed_id_x.append(url_in_ngrams)  
    worded_id_x = []
    for worded_url in worded_x: 
        word_ids = [words_dict[x] for x in worded_url]
        worded_id_x.append(word_ids) 
    
    return ngramed_id_x, ngrams_dict, worded_id_x, words_dict 

def ngram_id_x_from_dict(word_x, max_len_subwords, ngram_dict, word_dict = None): 
    char_ngram_len = 1
    print("Index of <UNKNOWN> word: {}".format(word_dict["<UNKNOWN>"]))
    ngramed_id_x = [] 
    worded_id_x = [] 
    counter = 0
    if word_dict:
        word_vocab = sorted(list(word_dict.keys()))
    for url in word_x:
        if counter % 100000 == 0: 
            print("Processing url #{}".format(counter))
        counter += 1  
        url_in_ngrams = [] 
        url_in_words = [] 
        words = url
        for word in words:
            ngrams = get_char_ngrams(char_ngram_len, word) 
            if len(ngrams) > max_len_subwords:
                word = "<UNKNOWN>"  
            ngrams_id = [] 
            for ngram in ngrams: 
                if ngram in ngram_dict: 
                    ngrams_id.append(ngram_dict[ngram]) 
                else: 
                    ngrams_id.append(0) 
            url_in_ngrams.append(ngrams_id)
            if is_in(word_vocab, word): 
                word_id = word_dict[word]
            else: 
                word_id = word_dict["<UNKNOWN>"] 
            url_in_words.append(word_id)
        ngramed_id_x.append(url_in_ngrams)
        worded_id_x.append(url_in_words)
    
    return ngramed_id_x, worded_id_x 

def bisect_search(a,x):
    i = bisect_left(a,x) 
    if i != len(a) and a[i] == x: 
        return i
    raise ValueError 

def is_in(a,x): 
    i = bisect_left(a,x)
    if i != len(a) and a[i] == x: 
        return True 
    else:
        return False 

def prep_train_test(pos_x, neg_x, dev_pct): 
    np.random.seed(10) 
    shuffle_indices=np.random.permutation(np.arange(len(pos_x)))
    pos_x_shuffled = pos_x[shuffle_indices]
    dev_idx = -1 * int(dev_pct * float(len(pos_x)))
    pos_train = pos_x_shuffled[:dev_idx]
    pos_test = pos_x_shuffled[dev_idx:]

    np.random.seed(10)
    shuffle_indices=np.random.permutation(np.arange(len(neg_x)))
    neg_x_shuffled = neg_x[shuffle_indices]
    dev_idx = -1 * int(dev_pct * float(len(neg_x)))
    neg_train = neg_x_shuffled[:dev_idx]
    neg_test = neg_x_shuffled[dev_idx:] 

    x_train = np.array(list(pos_train) + list(neg_train))
    y_train = len(pos_train)*[1] + len(neg_train)*[0]
    x_test = np.array(list(pos_test) + list(neg_test))
    y_test = len(pos_test)*[1] + len(neg_test)*[0]

    y_train = to_categorical(y_train, nb_classes=2)
    y_test = to_categorical(y_test, nb_classes=2) 

    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(x_train)))
    x_train = x_train[shuffle_indices]
    y_train = y_train[shuffle_indices]

    np.random.seed(10) 
    shuffle_indices = np.random.permutation(np.arange(len(x_test)))
    x_test = x_test[shuffle_indices]
    y_test = y_test[shuffle_indices] 
    
    print("Train Mal/Ben split: {}/{}".format(len(pos_train), len(neg_train)))
    print("Test Mal/Ben split: {}/{}".format(len(pos_test), len(neg_test)))
    print("Train/Test split: {}/{}".format(len(y_train), len(y_test)))
    print("Train/Test split: {}/{}".format(len(x_train), len(x_test)))

    return x_train, y_train, x_test, y_test 

def get_ngramed_id_x(x_idxs, ngramed_id_x): 
    output_ngramed_id_x = [] 
    for idx in x_idxs:  
        output_ngramed_id_x.append(ngramed_id_x[idx])
    return output_ngramed_id_x

def pad_seq(urls, max_d1=0, max_d2=0, embedding_size=128): 
    if max_d1 == 0 and max_d2 == 0: 
        for url in urls: 
            if len(url) > max_d1: 
                max_d1 = len(url) 
            for word in url: 
                if len(word) > max_d2: 
                    max_d2 = len(word) 
    pad_idx = np.zeros((len(urls), max_d1, max_d2, embedding_size))
    pad_urls = np.zeros((len(urls), max_d1, max_d2))
    pad_vec = [1 for i in range(embedding_size)]
    for d0 in range(len(urls)): 
        url = urls[d0]
        for d1 in range(len(url)): 
            if d1 < max_d1: 
                word = url[d1]
                for d2 in range(len(word)): 
                    if d2 < max_d2: 
                        pad_urls[d0,d1,d2] = word[d2]
                        pad_idx[d0,d1,d2] = pad_vec
    return pad_urls, pad_idx

def pad_seq_in_word(urls, max_d1=0, embedding_size=128):
    if max_d1 == 0: 
        url_lens = [len(url) for url in urls]
        max_d1 = max(url_lens)
    pad_urls = np.zeros((len(urls), max_d1))
    #pad_idx = np.zeros((len(urls), max_d1, embedding_size))
    #pad_vec = [1 for i in range(embedding_size)]
    for d0 in range(len(urls)): 
        url = urls[d0]
        for d1 in range(len(url)): 
            if d1 < max_d1: 
                pad_urls[d0,d1] = url[d1]
                #pad_idx[d0,d1] = pad_vec 
    return pad_urls 

def softmax(x): 
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum() 

def batch_iter(data, batch_size, num_epochs, shuffle=True): 
    data = np.array(data) 
    data_size = len(data) 
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1 
    for epoch in range(num_epochs): 
        if shuffle: 
            shuffle_indices = np.random.permutation(np.arange(data_size)) 
            shuffled_data = data[shuffle_indices]
        else: 
            shuffled_data = data 
        for batch_num in range(num_batches_per_epoch): 
            start_idx = batch_num * batch_size 
            end_idx = min((batch_num+1) * batch_size, data_size)
            yield shuffled_data[start_idx:end_idx]

def save_test_result(labels, all_predictions, all_scores, output_dir): 
    output_labels = []
    for i in labels: 
        if i == 1: 
            output_labels.append(i) 
        else: 
            output_labels.append(-1) 
    output_preds = [] 
    for i in all_predictions: 
        if i == 1: 
            output_preds.append(i) 
        else: 
            output_preds.append(-1) 
    softmax_scores = [softmax(i) for i in all_scores]
    with open(output_dir, "w") as file: 
        output = "label\tpredict\tscore\n"
        file.write(output)
        for i in range(len(output_labels)): 
            output = str(int(output_labels[i])) + '\t' + str(int(output_preds[i])) + '\t' + str(softmax_scores[i][1]) + '\n' 
            file.write(output) 

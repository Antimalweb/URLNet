**URLNet**
==========

Introduction
------------

This is an implementation of URLNet - Learning a URL Representation with Deep Learning for Malicious URL Detection
https://arxiv.org/abs/1802.03162

URLNet is a convolutional neural network (CNN) based model used to detect
malicious URLs. The model exploits features of URL text string at both character
and word levels.

Resources
---------

URLNet requires Python 2.7 and the following packages:

-   tensorflow 1.4

-   tflearn 0.3

-   numpy 1.14

Model Designs
-------------

![](img/URLNet.jpg)

Sample Commands
---------------

In all datasets for training or testing, each line includes the label and the
URL text string following the template:

`<URL label><tab><URL string>`

**Example:**

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
+1  http://www.exampledomain.com/urlpath/...

-1  http://www.exampledomain.com/urlpath/...
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The URL label is either +1 (Malicious) or -1 (Benign).


The model can be trained by running the following command:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
python train.py --FILE_DIR <data_file_directory> --EMB_MODE <embedding_mode> --EMB_DIM <nb_of_embedding_dimensions> --DELIMIT_MODE <url_delimit_mode> --FILTER_SIZES <filter_sizes_separated_by_comma> <convolutional_filter_sizes> --NB_EPOCH <nb_of_training_epochs> --BATCH_SIZE <nb_of_urls_per_batch> --OUTPUT_DIR <model_output_folder_directory>
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The training will save all the related word and character dictionaries into an
output folder, and the model checkpoints are saved into `checkpoints/` folder.
By default, maximum 5 checkpoints are stored in the folder.


The model can be tested by running the following command:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
python test.py --FILE_DIR <data_file_directory> --WORD_DICT_DIR <word_dictionary_directory> --NGRAM_DICT_DIR <character_in_word_dictionary_directory> --CHAR_DICT_DIR <character_dictionary_directory> --CHECKPOINT_DIR <model_checkpoint_directory> --OUTPUT_DIR <test_result_directory> --EMB_MODE <embedding_mode> --DELIMIT_MODE <url_delimit_mode> --EMB_DIM <nb_of_embedding_dimensions> --BATCH_SIZE <nb_of_urls_per_batch>
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The test will save the test results for each URL, including 3 columns: `label`
(original label), `predict` (prediction label), and `score` (softmax score). The
orders of test results from top to bottom are the same as their orders in the
test dataset. If the score is more than 0.5, prediction label is +1 (Malicious).
Else, the prediction is -1 (Benign).

**Example:**
~~~~~~~~~~~~~~~~~~~~~~
label   predict score

1   1       0.884

-1  -1      0.359
~~~~~~~~~~~~~~~~~~~~~~

To obtain test metrics such as True Positive, False Positive, True Negative,
False Negative, and the AUC curves, run the following command:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
python auc.py --input_path <test_result_directory> --input_file
<test_result_file>
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The command will output the test metrics into a file with the same name as the
`input_file` but with an `.auc` extension.


Parameters
----------

Training parameters include:

| **Parameter**       | **Description**                                                                                                                                                                                 | **Default**     |
|---------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------|
| MAX_LENGTH_WORDS    | The maximum number of words in a URL. The URL is either truncated or padded with a `<PADDING>` token to reach this length.                                                                    | 200             |
| MAX_LENGTH_CHARS    | The maximum number of characters in a URL. The URL is either truncted or padded with a `<PADDING>` character to reach this length.                                                              | 200             |
| MAX_LENGTH_SUBWORDS | The maximum number of characters in each word in a URL. Each word is either truncated or padded with a `<PADDING>` character to reach this length.                                              | 20              |
| MIN_WORD_FREQ       | Words that have frequency less than or equal to this parameter are considered as rare words, and represented as a single `<UNKNOWN>` token.                                                     | 1               |
| EMB_DIM             | Dimension size of word and character embedding.                                                                                                                                                 | 32              |
| EMB_MODE            | 1: only character-based CNN, 2: only word-based CNN, 3: character and word CNN, 4: character-level word CNN, 5: character and character-level word CNN                                          | 1               |
| DELIMIT_MODE        | 0: delimit URL by special characters (i.e. non-alphanumerical characters), 1: delimit URL by special characters (i.e. non-alphanumerical characters) and treat each special characters as words | 1               |
| EMB_DIM             | Dimension size of word and character embedding.                                                                                                                                                 | 32              |
| FILTER_SIZES        | Sizes of convolutional filters. If more than one branches of CNN, all will have the same set of filter sizes. Separate the filter sizes by comma.                                               | 3,4,5,6         |
| BATCH_SIZE          | Number of URLs in each training batch                                                                                                                                                           | 128             |
| NB_EPOCHS           | Number of training epochs                                                                                                                                                                       | 5               |
| DEV_PERCENTAGE      | Percentage of training data used for validation                                                                                                                                                 | 0.001           |
| LR                  | Learning rate                                                                                                                                                                                   | 0.001           |
| L2_REG_LAMBDA       | regularization parameter of loss function                                                                                                                                                       | 0               |
| FILE_DIR            | Directory of the training dataset                                                                                                                                                               | train_10000.txt |
| OUTPUT_DIR          | Output folder to save dictionaries and model checkpoints                                                                                                                                        | runs/10000/     |
| PRINT_EVERY         | Output the training loss and accuracy after this number of batches                                                                                                                              | 50              |
| EVAL_EVERY          | Evaluate the model on the validation set after this number of batches                                                                                                                           | 500             |
| CHECKPOINT_EVERY    | Checkpoint the model after this number of batches. Only save the model checkpoint if the validation loss is improved.                                                                           | 500             |

 

Test parameters include:

| **Parameter**       | **Description**                                                                                                                                                                                 | **Default**                          |
|---------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------|
| MAX_LENGTH_WORDS    | The maximum number of words in a URL. The URL is either truncated or padded with a `<PADDING>` token to reach this length.                                                                    | 200                                  |
| MAX_LENGTH_CHARS    | The maximum number of characters in a URL. The URL is either truncted or padded with a `<PADDING>` character to reach this length.                                                              | 200                                  |
| MAX_LENGTH_SUBWORDS | The maximum number of characters in each word in a URL. Each word is either truncated or padded with a `<PADDING>` character to reach this length.                                              | 20                                   |
| EMB_DIM             | Dimension size of word and character embedding.                                                                                                                                                 | 32                                   |
| EMB_MODE            | 1: only character-based CNN, 2: only word-based CNN, 3: character and word CNN, 4: character-level word CNN, 5: character and character-level word CNN                                          | 1                                    |
| DELIMIT_MODE        | 0: delimit URL by special characters (i.e. non-alphanumerical characters), 1: delimit URL by special characters (i.e. non-alphanumerical characters) and treat each special characters as words | 1                                    |
| BATCH_SIZE          | Number of URLs in each test batch                                                                                                                                                               | 128                                  |
| FILE_DIR            | Directory of the test dataset                                                                                                                                                                   | test_1000.txt                        |
| WORD_DICT_DIR       | Directory of the word dictionary file. Dictionary file is in pickle extension `.p`                                                                                                              | runs/10000/words_dict.p              |
| CHAR_DICT_DIR       | Directory of the character dictionary file. Dictionary file is in pickle extension `.p`                                                                                                         | runs/10000/chars_dict.p              |
| NGRAM_DICT_DIR      | Directory of the character-in-word dictionary file. Dictionary file is in pickle extension `.p`                                                                                                 | runs/10000/ngrams_dict.p             |
| CHECKPOINT_DIR      | Directory of the model checkpoints                                                                                                                                                              | runs/10000/checkpoints/              |
| OUTPUT_DIR          | Directory of the test results                                                                                                                                                                   | runs/10000/train_10000_test_1000.txt |

The test parameters such as `EMB_MODE` and `DELIMIT_MODE` have to be consistent
with the trained model to get accurate test results.

Refer to the file `run.sh` for example commands.

Acknowledgement
---------------

We are very grateful to our collaborators VirusTotal, for providing us access to their data, without which this research would not have been possible. 


emb_modes=(1 2 2 3 3 4 5)
delimit_modes=(0 0 1 0 1 1 1)
train_size=1000000
test_size=500000
nb_epoch=1

for ((i=0; i <${#emb_modes[@]}; ++i))
    do
    python train.py --FILE_DIR ../../data/virustotal/train_${train_size}.txt \
    --DEV_PERCENTAGE 0.001 \
    --EMB_MODE ${emb_modes[$i]} --DELIMIT_MODE ${delimit_modes[$i]} \
    --EMB_DIM 32 --MIN_WORD_FREQ 1 \
    --FILTER_SIZES 3,4,5,6 \
    --NB_EPOCHS ${nb_epoch} --BATCH_SIZE 1048 --PRINT_EVERY 50 --EVAL_EVERY 500 --CHECKPOINT_EVERY 500 \
    --OUTPUT_DIR runs/${train_size}_emb${emb_modes[$i]}_dlm${delimit_modes[$i]}_32dim_minwf1_1conv3456_${nb_epoch}ep/ 

    python test.py --FILE_DIR ../../data/virustotal/test_${test_size}.txt \
    --WORD_DICT_DIR runs/${train_size}_emb${emb_modes[$i]}_dlm${delimit_modes[$i]}_32dim_minwf1_1conv3456_${nb_epoch}ep/words_dict.p \
    --NGRAM_DICT_DIR runs/${train_size}_emb${emb_modes[$i]}_dlm${delimit_modes[$i]}_32dim_minwf1_1conv3456_${nb_epoch}ep/ngrams_dict.p \
    --CHAR_DICT_DIR runs/${train_size}_emb${emb_modes[$i]}_dlm${delimit_modes[$i]}_32dim_minwf1_1conv3456_${nb_epoch}ep/chars_dict.p \
    --CHECKPOINT_DIR runs/${train_size}_emb${emb_modes[$i]}_dlm${delimit_modes[$i]}_32dim_minwf1_1conv3456_${nb_epoch}ep/checkpoints/ \
    --OUTPUT_DIR runs/${train_size}_emb${emb_modes[$i]}_dlm${delimit_modes[$i]}_32dim_minwf1_1conv3456_${nb_epoch}ep/train_${train_size}_test_${test_size}.txt \
    --EMB_MODE ${emb_modes[$i]} --DELIMIT_MODE ${delimit_modes[$i]} \
    --EMB_DIM 32 --BATCH_SIZE 1048

    python auc.py --input_path runs/${train_size}_emb${emb_modes[$i]}_dlm${delimit_modes[$i]}_32dim_minwf1_1conv3456_${nb_epoch}ep/ --input_file train_${train_size}_test_${test_size}.txt --threshold 0.5
    done

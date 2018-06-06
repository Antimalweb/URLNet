emb_modes=(1 2 2 3 3 4 5)
delimit_modes=(0 0 1 0 1 1 1 )
train_size=100000
test_size=100000
nb_epoch=5

for ((i=0; i <${#emb_modes[@]}; ++i))
    do
    python train.py --data.data_dir ../../data/virustotal/train_${train_size}.txt \
    --data.dev_pct 0.001 --data.delimit_mode ${delimit_modes[$i]} --data.min_word_freq 1 \
    --model.emb_mode ${emb_modes[$i]} --model.emb_dim 32 --model.filter_sizes 3,4,5,6 \
    --train.nb_epochs ${nb_epoch} --train.batch_size 1048 \
    --log.print_every 5 --log.eval_every 10 --log.checkpoint_every 10 \
    --log.output_dir runs/${train_size}_emb${emb_modes[$i]}_dlm${delimit_modes[$i]}_32dim_minwf1_1conv3456_${nb_epoch}ep/ 

    python test.py --data.data_dir ../../data/virustotal/test_${test_size}.txt \
    --data.delimit_mode ${delimit_modes[$i]} \
    --data.word_dict_dir runs/${train_size}_emb${emb_modes[$i]}_dlm${delimit_modes[$i]}_32dim_minwf1_1conv3456_${nb_epoch}ep/words_dict.p \
    --data.subword_dict_dir runs/${train_size}_emb${emb_modes[$i]}_dlm${delimit_modes[$i]}_32dim_minwf1_1conv3456_${nb_epoch}ep/subwords_dict.p \
    --data.char_dict_dir runs/${train_size}_emb${emb_modes[$i]}_dlm${delimit_modes[$i]}_32dim_minwf1_1conv3456_${nb_epoch}ep/chars_dict.p \
    --log.checkpoint_dir runs/${train_size}_emb${emb_modes[$i]}_dlm${delimit_modes[$i]}_32dim_minwf1_1conv3456_${nb_epoch}ep/checkpoints/ \
    --log.output_dir runs/${train_size}_emb${emb_modes[$i]}_dlm${delimit_modes[$i]}_32dim_minwf1_1conv3456_${nb_epoch}ep/train_${train_size}_test_${test_size}.txt \
    --model.emb_mode ${emb_modes[$i]} --model.emb_dim 32 --test.batch_size 1048

    python auc.py --input_path runs/${train_size}_emb${emb_modes[$i]}_dlm${delimit_modes[$i]}_32dim_minwf1_1conv3456_${nb_epoch}ep/ --input_file train_${train_size}_test_${test_size}.txt --threshold 0.5
    done

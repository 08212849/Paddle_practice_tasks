# GPU
python -u -m paddle.distributed.launch --gpus "0" \
    train_batch_neg.py \
    --device gpu \
    --save_dir ./model_checkpoints/ \
    --batch_size 16\
    --learning_rate 5E-5 \
    --epochs 1\
    --output_emb_size 128 \
    --model_name_or_path ernie-3.0-xbase-zh \
    --save_steps 2000 \
    --log_steps 20 \
    --max_seq_length 64 \
    --margin 0.2 \
    --train_set_file "../data/lprecall_data/annret/train.csv"\
    --recall_result_dir "./" \
    --recall_result_file "predict.txt" \
    --hnsw_m 100 \
    --hnsw_ef 100 \
    --recall_num 50 \
    --similar_text_pair_file "../data/lprecall_data/annret/dev.csv" \
    --corpus_file "../data/lprecall_data/annret/corpus.csv"


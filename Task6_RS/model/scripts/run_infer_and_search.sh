#!/bin/bash
# -- coding: utf-8 --**

model_dir="model_checkpoints/model_3500"

python -u -m paddle.distributed.launch --log_level=CRITICAL --gpus "0" --log_dir "recall_log/" \
        infer_and_search.py \
        --device gpu \
        --recall_result_dir "./" \
        --recall_result_file "predict.txt" \
        --params_path "${model_dir}/model_state.pdparams" \
        --model_name_or_path "${model_dir}" \
        --hnsw_m 100 \
        --hnsw_ef 100 \
        --batch_size 512\
        --output_emb_size 128 \
        --max_seq_length 64 \
        --recall_num 10 \
        --input_json_file "$1" \
        --index_path "hnsw_index.bin"
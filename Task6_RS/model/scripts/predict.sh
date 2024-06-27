export GLOG_minloglevel=9
# gpu version
root_dir="model_checkpoints/model_27000" 
python -u -m paddle.distributed.launch --gpus "0" \
    predict.py \
    --device gpu \
    --params_path "${root_dir}/model_state.pdparams" \
    --model_name_or_path "${root_dir}" \
    --output_emb_size 128 \
    --batch_size 128 \
    --max_seq_length 64 \
    --text_pair_file "../data/lprecall/annret/dev.csv"


# cpu
# root_dir="model_checkpoints" 
# python predict.py \
#     --device cpu \
#     --params_path "${root_dir}/model_40/model_state.pdparams" \
#     --output_emb_size 256 \
#     --batch_size 128 \
#     --max_seq_length 64 \
#     --text_pair_file "data/recall/test.csv"

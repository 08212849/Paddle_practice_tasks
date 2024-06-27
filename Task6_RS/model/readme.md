# ANN 双塔召回  Demo

1. sh scripts/run_train.sh 模型训练
2. sh scripts/run_build_and_predict.sh 基于已训练的 checkpoint 预估结果，分位以下3步
    2.1 对 corpus.txt 中所有候选infer向量 -> ad_vec
    2.2 对 ad_vec 用 hnsw 建库，得到候选索引 -> hnsw_index
    2.3 对测试集 Query 调用模型 infer 向量 -> query_vec
    2.4 对 query_vec，hnsw_index 调用hnsw 检索算法，得到索引id，即为 ad_id.


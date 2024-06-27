"""
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""


import paddle
import paddle.nn.functional as F
from base_model import SemanticIndexBase


class SemanticIndexBatchNeg(SemanticIndexBase):
    """ 专注于在批量处理中实现负采样的损失计算，适用于训练阶段。
    """

    def __init__(self, pretrained_model, dropout=None, margin=0.3, scale=30, output_emb_size=None):
        """ 初始化SemanticIndexBatchNeg类。
        """
        super().__init__(pretrained_model, dropout, output_emb_size)
        self.margin = margin
        # Used scaling cosine similarity to ease converge
        self.scale = scale

    def forward(
        self, query_input_ids, title_input_ids,
        query_token_type_ids=None, query_position_ids=None, query_attention_mask=None,
        title_token_type_ids=None, title_position_ids=None, title_attention_mask=None,
    ):
        """ 前向传播函数，返回计算所得的余弦相似度和相关标签，以及查询和标题的嵌入。
        """

        query_cls_embedding = self.get_pooled_embedding(
            query_input_ids, query_token_type_ids, query_position_ids, query_attention_mask
        )

        title_cls_embedding = self.get_pooled_embedding(
            title_input_ids, title_token_type_ids, title_position_ids, title_attention_mask
        )

        cosine_sim = paddle.matmul(query_cls_embedding, title_cls_embedding, transpose_y=True)

        # Subtract margin from all positive samples cosine_sim()
        margin_diag = paddle.full(
            shape=[query_cls_embedding.shape[0]], fill_value=self.margin, dtype=paddle.get_default_dtype()
        )

        cosine_sim = cosine_sim - paddle.diag(margin_diag)

        # Scale cosine to ease training converge
        cosine_sim *= self.scale

        labels = paddle.arange(0, query_cls_embedding.shape[0], dtype="int64")
        labels = paddle.reshape(labels, shape=[-1, 1])

        loss = F.cross_entropy(input=cosine_sim, label=labels)

        return loss


class SemanticIndexCacheNeg(SemanticIndexBase):
    """ 是带缓存机制的负采样，适合在需要保存中间结果以用于后续处理的场景。
    """

    def __init__(self, pretrained_model, dropout=None, margin=0.3, scale=30, output_emb_size=None):
        """ 初始化SemanticIndexCacheNeg类。
        """
        super().__init__(pretrained_model, dropout, output_emb_size)
        self.margin = margin
        # Used scaling cosine similarity to ease converge
        self.scale = scale

    def forward(self, query_input_ids, title_input_ids,
        query_token_type_ids=None, query_position_ids=None, query_attention_mask=None,
        title_token_type_ids=None, title_position_ids=None, title_attention_mask=None,
    ):
        """ 前向传播函数，返回计算所得的余弦相似度和相关标签，以及查询和标题的嵌入。
        """

        query_cls_embedding = self.get_pooled_embedding(
            query_input_ids, query_token_type_ids, query_position_ids, query_attention_mask
        )

        title_cls_embedding = self.get_pooled_embedding(
            title_input_ids, title_token_type_ids, title_position_ids, title_attention_mask
        )

        cosine_sim = paddle.matmul(query_cls_embedding, title_cls_embedding, transpose_y=True)

        # Subtract margin from all positive samples cosine_sim()
        margin_diag = paddle.full(shape=[query_cls_embedding.shape[0]], fill_value=self.margin, dtype=cosine_sim.dtype)

        cosine_sim = cosine_sim - paddle.diag(margin_diag)

        # Scale cosine to ease training converge
        cosine_sim *= self.scale

        labels = paddle.arange(0, query_cls_embedding.shape[0], dtype="int64")
        labels = paddle.reshape(labels, shape=[-1, 1])

        return [cosine_sim, labels, query_cls_embedding, title_cls_embedding]

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

import abc

import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class SemanticIndexBase(nn.Layer):
    """ 
    动态图版本的语义索引模型，用于生成和处理语义嵌入。
    """

    def __init__(self, pretrained_model, dropout=None, output_emb_size=None):
        """ 接受一个预训练模型，可选的dropout率和输出嵌入大小。
        如果指定了输出嵌入大小，会添加一个线性层来减少嵌入的维度。
        """
        super().__init__()
        self.ptm = pretrained_model
        self.dropout = nn.Dropout(dropout if dropout is not None else 0.1)

        # if output_emb_size is not None, then add Linear layer to reduce embedding_size,
        # we recommend set output_emb_size = 256 considering the trade-off between
        # recall performance and efficiency

        self.output_emb_size = output_emb_size
        if output_emb_size > 0:
            weight_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.TruncatedNormal(std=0.02))
            self.emb_reduce_linear = paddle.nn.Linear(
                self.ptm.config.hidden_size, output_emb_size, weight_attr=weight_attr
            )

    def get_pooled_embedding(self, input_ids, token_type_ids=None, position_ids=None, attention_mask=None):
        """ 接收输入标识符、令牌类型标识符等，并从预训练模型获取池化嵌入（通常是CLS标记的嵌入）。
        如果定义了输出嵌入大小，将通过线性层减少维度，然后应用dropout和归一化。
        """
        _, cls_embedding = self.ptm(input_ids, token_type_ids, position_ids, attention_mask)

        if self.output_emb_size > 0:
            cls_embedding = self.emb_reduce_linear(cls_embedding)
        cls_embedding = self.dropout(cls_embedding)
        cls_embedding = F.normalize(cls_embedding, p=2, axis=-1)

        return cls_embedding

    def get_semantic_embedding(self, data_loader):
        """ 用于从数据加载器中批量获取语义嵌入，通常用于索引构建或查询处理。
        """
        self.eval()
        with paddle.no_grad():
            for batch_data in data_loader:
                input_ids, token_type_ids = batch_data
                text_embeddings = self.get_pooled_embedding(input_ids, token_type_ids=token_type_ids)
                yield text_embeddings

    def cosine_sim(self, query_input_ids, title_input_ids,
        query_token_type_ids=None, query_position_ids=None, query_attention_mask=None,
        title_token_type_ids=None, title_position_ids=None, title_attention_mask=None,
    ):
        """
        计算两组嵌入（查询和标题）之间的余弦相似度。
        """

        query_cls_embedding = self.get_pooled_embedding(
            query_input_ids, query_token_type_ids, query_position_ids, query_attention_mask
        )

        title_cls_embedding = self.get_pooled_embedding(
            title_input_ids, title_token_type_ids, title_position_ids, title_attention_mask
        )

        cosine_sim = paddle.sum(query_cls_embedding * title_cls_embedding, axis=-1)
        return cosine_sim

    @abc.abstractmethod
    def forward(self):
        """ 抽象方法，需要在子类中实现具体功能。
        """
        pass


class SemanticIndexBaseStatic(nn.Layer):
    """ 静态图版本的语义索引模型，适用于部署场景，提高执行效率。
    """

    def __init__(self, pretrained_model, dropout=None, output_emb_size=None):
        """ 初始化预训练模型、dropout和可选的输出嵌入大小减少层。
        """
        super().__init__()
        self.ptm = pretrained_model
        self.dropout = nn.Dropout(dropout if dropout is not None else 0.1)

        # if output_emb_size is not None, then add Linear layer to reduce embedding_size,
        # we recommend set output_emb_size = 256 considering the trade-off between
        # recall performance and efficiency

        self.output_emb_size = output_emb_size
        if output_emb_size > 0:
            weight_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.TruncatedNormal(std=0.02))
            self.emb_reduce_linear = paddle.nn.Linear(
                self.ptm.config.hidden_size, output_emb_size, weight_attr=weight_attr
            )

    @paddle.jit.to_static(
        input_spec=[
            paddle.static.InputSpec(shape=[None, None], dtype="int64"),
            paddle.static.InputSpec(shape=[None, None], dtype="int64"),
        ]
    )
    def get_pooled_embedding(self, input_ids, token_type_ids=None, position_ids=None, attention_mask=None):
        """ 与动态图版本类似，但使用了 paddle.jit.to_static 装饰器来优化静态图的执行。
        输入规格通过 InputSpec 定义，确保输入是合适的静态图形状和类型。
        """
        _, cls_embedding = self.ptm(input_ids, token_type_ids, position_ids, attention_mask)

        if self.output_emb_size > 0:
            cls_embedding = self.emb_reduce_linear(cls_embedding)
        cls_embedding = self.dropout(cls_embedding)
        cls_embedding = F.normalize(cls_embedding, p=2, axis=-1)

        return cls_embedding

    def get_semantic_embedding(self, data_loader):
        """ 获取语义嵌入
        """
        self.eval()
        with paddle.no_grad():
            for batch_data in data_loader:
                input_ids, token_type_ids = batch_data
                text_embeddings = self.get_pooled_embedding(input_ids, token_type_ids=token_type_ids)
                yield text_embeddings

    def cosine_sim(self, query_input_ids, title_input_ids,
        query_token_type_ids=None, query_position_ids=None, query_attention_mask=None,
        title_token_type_ids=None, title_position_ids=None, title_attention_mask=None,
    ):
        """ 计算余弦相似度。
        """

        query_cls_embedding = self.get_pooled_embedding(
            query_input_ids, query_token_type_ids, query_position_ids, query_attention_mask
        )

        title_cls_embedding = self.get_pooled_embedding(
            title_input_ids, title_token_type_ids, title_position_ids, title_attention_mask
        )

        cosine_sim = paddle.sum(query_cls_embedding * title_cls_embedding, axis=-1)
        return cosine_sim

    def forward(self, input_ids, token_type_ids=None, position_ids=None, attention_mask=None):
        """ 实现了前向传播功能
        """
        _, cls_embedding = self.ptm(input_ids, token_type_ids, position_ids, attention_mask)

        if self.output_emb_size > 0:
            cls_embedding = self.emb_reduce_linear(cls_embedding)
        cls_embedding = self.dropout(cls_embedding)
        cls_embedding = F.normalize(cls_embedding, p=2, axis=-1)

        return cls_embedding

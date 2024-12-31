"""
MIT License from https://github.com/marmotlab/CAtNIPP/

Copyright (c) 2022 MARMot Lab @ NUS-ME

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import torch
import torch.nn as nn
import torch_geometric.nn as gnn
import math
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torch.cuda.amp.autocast_mode import autocast
from gym_pybullet_drones.envs.IPPArguments import IPPArg


class SingleHeadAttention(nn.Module):

    def __init__(self, embedding_dim):
        super(SingleHeadAttention, self).__init__()
        self.input_dim = embedding_dim
        self.embedding_dim = embedding_dim
        self.value_dim = embedding_dim
        self.key_dim = self.value_dim
        self.tanh_clipping = 10
        self.norm_factor = 1 / math.sqrt(self.key_dim)

        self.w_query = nn.Parameter(torch.Tensor(self.input_dim, self.key_dim))
        self.w_key = nn.Parameter(torch.Tensor(self.input_dim, self.key_dim))

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, h=None, mask=None):
        """
                :param q: queries (batch_size, n_query, input_dim)
                :param h: data (batch_size, graph_size, input_dim)
                :param mask: mask (batch_size, n_query, graph_size) or viewable as that (i.e. can be 2 dim if n_query == 1)
                Mask should contain 1 if attention is not possible (i.e. mask is negative adjacency)
                :return:
                """
        if h is None:
            h = q

        batch_size, target_size, input_dim = h.size()
        n_query = q.size(1)  # n_query = target_size in tsp

        #assert q.size(0) == batch_size
        #assert q.size(2) == input_dim
        #assert input_dim == self.input_dim

        h_flat = h.reshape(-1, input_dim)  # (batch_size*graph_size)*input_dim
        q_flat = q.reshape(-1, input_dim)  # (batch_size*n_query)*input_dim

        shape_k = (batch_size, target_size, -1)
        shape_q = (batch_size, n_query, -1)

        Q = torch.matmul(q_flat, self.w_query).view(
            shape_q)  # batch_size*n_query*key_dim
        K = torch.matmul(h_flat, self.w_key).view(
            shape_k)  # batch_size*targets_size*key_dim

        U = self.norm_factor * torch.matmul(Q, K.transpose(
            1, 2))  # batch_size*n_query*targets_size
        U = self.tanh_clipping * torch.tanh(U)

        if mask is not None:
            mask = mask.view(batch_size, 1, target_size).expand_as(
                U)  # copy for n_heads times
            # U = U-1e8*mask  # ??
            #U[mask.bool()] = -1e8
            U[mask.bool()] = -1e4
        attention = torch.log_softmax(
            U, dim=-1)  # batch_size*n_query*targets_size

        out = attention

        return out


class MultiHeadAttention(nn.Module):

    def __init__(self, embedding_dim, n_heads=8):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.input_dim = embedding_dim
        self.embedding_dim = embedding_dim
        self.value_dim = self.embedding_dim // self.n_heads
        self.key_dim = self.value_dim
        self.norm_factor = 1 / math.sqrt(self.key_dim)

        self.w_query = nn.Parameter(
            torch.Tensor(self.n_heads, self.input_dim, self.key_dim))
        self.w_key = nn.Parameter(
            torch.Tensor(self.n_heads, self.input_dim, self.key_dim))
        self.w_value = nn.Parameter(
            torch.Tensor(self.n_heads, self.input_dim, self.value_dim))
        self.w_out = nn.Parameter(
            torch.Tensor(self.n_heads, self.value_dim, self.embedding_dim))

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, h=None, mask=None):
        """
                :param q: queries (batch_size, n_query, input_dim)
                :param h: data (batch_size, graph_size, input_dim)
                :param mask: mask (batch_size, n_query, graph_size) or viewable as that (i.e. can be 2 dim if n_query == 1)
                Mask should contain 1 if attention is not possible (i.e. mask is negative adjacency)
                :return:
                """
        if h is None:
            h = q

        batch_size, target_size, input_dim = h.size()
        n_query = q.size(1)  # n_query = target_size in tsp
        #assert q.size(0) == batch_size
        #assert q.size(2) == input_dim
        #assert input_dim == self.input_dim

        h_flat = h.contiguous().view(
            -1, input_dim)  # (batch_size*graph_size)*input_dim
        q_flat = q.contiguous().view(
            -1, input_dim)  # (batch_size*n_query)*input_dim
        shape_v = (self.n_heads, batch_size, target_size, -1)
        shape_k = (self.n_heads, batch_size, target_size, -1)
        shape_q = (self.n_heads, batch_size, n_query, -1)

        Q = torch.matmul(q_flat, self.w_query).view(
            shape_q)  # n_heads*batch_size*n_query*key_dim
        K = torch.matmul(h_flat, self.w_key).view(
            shape_k)  # n_heads*batch_size*targets_size*key_dim
        V = torch.matmul(h_flat, self.w_value).view(
            shape_v)  # n_heads*batch_size*targets_size*value_dim

        U = self.norm_factor * torch.matmul(Q, K.transpose(
            2, 3))  # n_heads*batch_size*n_query*targets_size

        if mask is not None:
            mask = mask.view(1, batch_size, -1, target_size).expand_as(
                U)  # copy for n_heads times
            # U = U.masked_fill(mask == 1, -np.inf)
            U[mask.bool()] = -np.inf
        attention = torch.softmax(
            U, dim=-1)  # n_heads*batch_size*n_query*targets_size

        if mask is not None:
            attnc = attention.clone()
            attnc[mask.bool()] = 0
            # attnc = attnc.masked_fill(mask == 1, 0)
            attention = attnc

        heads = torch.matmul(attention,
                             V)  # n_heads*batch_size*n_query*value_dim

        out = torch.mm(
            heads.permute(1, 2, 0, 3).reshape(-1,
                                              self.n_heads * self.value_dim),
            # batch_size*n_query*n_heads*value_dim
            self.w_out.view(-1, self.embedding_dim)
            # n_heads*value_dim*embedding_dim
        ).view(batch_size, n_query, self.embedding_dim)

        return out  # batch_size*n_query*embedding_dim


class Normalization(nn.Module):

    def __init__(self, embedding_dim):
        super(Normalization, self).__init__()
        self.normalizer = nn.LayerNorm(embedding_dim)

    def forward(self, input):
        return self.normalizer(input.contiguous().view(
            -1, input.size(-1))).view(*input.size())


class EncoderLayer(nn.Module):

    def __init__(self, embedding_dim, n_head):
        super(EncoderLayer, self).__init__()
        self.multiHeadAttention = MultiHeadAttention(embedding_dim, n_head)
        self.normalization1 = Normalization(embedding_dim)
        self.feedForward = nn.Sequential(nn.Linear(embedding_dim, 512),
                                         nn.ReLU(inplace=True),
                                         nn.Linear(512, embedding_dim))
        self.normalization2 = Normalization(embedding_dim)

    def forward(self, src, mask=None):
        h0 = src
        h = self.normalization1(src)
        h = self.multiHeadAttention(q=h, mask=mask)
        h = h + h0
        h1 = h
        h = self.normalization2(h)
        h = self.feedForward(h)
        h2 = h + h1
        return h2


class DecoderLayer(nn.Module):

    def __init__(self, embedding_dim, n_head):
        super(DecoderLayer, self).__init__()
        self.multiHeadAttention = MultiHeadAttention(embedding_dim, n_head)
        self.normalization1 = Normalization(embedding_dim)
        self.feedForward = nn.Sequential(nn.Linear(embedding_dim, 512),
                                         nn.ReLU(inplace=True),
                                         nn.Linear(512, embedding_dim))
        self.normalization2 = Normalization(embedding_dim)

    def forward(self, tgt, memory, mask=None):
        h0 = tgt
        tgt = self.normalization1(tgt)
        memory = self.normalization1(memory)
        h = self.multiHeadAttention(q=tgt, h=memory, mask=mask)
        h = h + h0
        h1 = h
        h = self.normalization2(h)
        h = self.feedForward(h)
        h2 = h + h1
        return h2


class Encoder(nn.Module):

    def __init__(self, embedding_dim=128, n_head=8, n_layer=3):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(
            EncoderLayer(embedding_dim, n_head) for i in range(n_layer))

    def forward(self, src, mask=None):
        for layer in self.layers:
            src = layer(src, mask)
        return src


class Decoder(nn.Module):

    def __init__(self, embedding_dim=128, n_head=8, n_layer=1):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(
            [DecoderLayer(embedding_dim, n_head) for i in range(n_layer)])

    def forward(self, tgt, memory, mask=None):
        for layer in self.layers:
            tgt = layer(tgt, memory, mask)
        return tgt


class AttentionNet(nn.Module):

    def __init__(self, embedding_dim):
        '''
        TODO: 初步搞清楚了输入与输出的维度关系， embedding 为线性层，输入的最后一个维度应为线性层的输入维度
        '''
        super(AttentionNet, self).__init__()

        self.feature_embedding = nn.Sequential(
            nn.Linear(2, IPPArg.EMBEDDING_DIM // 2), nn.ReLU(),
            nn.Linear(IPPArg.EMBEDDING_DIM // 2, IPPArg.EMBEDDING_DIM),
            nn.ReLU())

        self.belief_embedding = nn.Linear(1, embedding_dim)
        self.belief_encoder = Decoder(embedding_dim=IPPArg.EMBEDDING_DIM,
                                      n_head=IPPArg.N_HEAD,
                                      n_layer=IPPArg.N_LAYER)
        self.target_encoder = Decoder(embedding_dim=IPPArg.EMBEDDING_DIM,
                                      n_head=IPPArg.N_HEAD,
                                      n_layer=IPPArg.N_LAYER)

        self.temporal_encoder = Decoder(embedding_dim=IPPArg.EMBEDDING_DIM,
                                        n_head=IPPArg.N_HEAD,
                                        n_layer=IPPArg.N_LAYER)
        self.timefusion_layer = nn.Linear(1, embedding_dim)
        self.spatio_encoder = Encoder(embedding_dim=embedding_dim,
                                      n_head=IPPArg.N_HEAD,
                                      n_layer=IPPArg.N_LAYER)
        self.distfusion_layer = nn.Linear(embedding_dim + 1, embedding_dim)
        self.spatio_pos_embedding = nn.Linear(IPPArg.num_eigen_value,
                                              embedding_dim)
        self.spatio_decoder = Decoder(
            embedding_dim=IPPArg.EMBEDDING_DIM,
            n_head=IPPArg.N_HEAD,
            n_layer=IPPArg.N_LAYER
        )  # 用作 node_feature 与 connect_node_feature 的 cross-attention
        self.pointer = SingleHeadAttention(embedding_dim)

    def graph_embedding(self,
                        node_inputs: torch.Tensor,
                        dt_pool_inputs: torch.Tensor,
                        mask=None):
        """
        Description:
            spatio_attention, 做 node_coords 维度的 attention
        Args:
            node_inputs: (batch, history_size, graph_size,num_drone * feature) 
            dt_pool_inputs: (batch, history_size, 1)
        """

        batch_size, history_size, graph_size, input_dim = node_inputs.shape
        target_num = input_dim // IPPArg.BELIEF_FEATURE_DIM  # ego and all target

        node_inputs = node_inputs.reshape(batch_size, history_size, graph_size,
                                          target_num,
                                          IPPArg.BELIEF_FEATURE_DIM)
        # feature_dim 的前两个维度为 node_coord
        target_feature_embedding: torch.Tensor = self.feature_embedding(
            node_inputs[:, :, :, :, :2])
        # (batch_size, history_size, graph_size, target_num, embedding_dim)

        # belief feature
        belief_embedding = torch.softmax(torch.sum(node_inputs[:, :, :, :,
                                                               -1:],
                                                   dim=3),
                                         dim=2)
        belief_feature = self.belief_embedding(belief_embedding)

        # reshape
        belief_feature = belief_feature.reshape(
            -1, graph_size, IPPArg.EMBEDDING_DIM)  #(b*h,graph,128)
        graph_features = []
        ## yaw feature
        for i in range(graph_size):
            target_feature = target_feature_embedding[:, :,
                                                      i, :, :]  #(batch_size,history_size,target_num,128 + 1)
            target_feature = target_feature.reshape(
                -1, target_num, IPPArg.EMBEDDING_DIM
            )  #(batch_sizeb*history_size,target_num,128 + 1)
            ### 使用 belief 作为 mask
            mask = node_inputs[:, :, i, :, -1:].reshape(-1, target_num,
                                                        1).permute(0, 2, 1)
            mask = (mask == 0).int()
            target_feature = self.target_encoder(
                target_feature[:, :1, :], target_feature,
                mask=mask)  # cross_attention (b*h,1,128)
            graph_features.append(target_feature)

        ### cross-attention
        graph_features = torch.cat(graph_features,
                                   dim=1)  #(b*h,graph_size,128)
        graph_features = self.belief_encoder(
            graph_features, torch.cat((graph_features, belief_feature), dim=1))
        #### belief graph cross
        graph_features = graph_features.reshape(batch_size, history_size,
                                                graph_size,
                                                IPPArg.EMBEDDING_DIM)
        graph_features = graph_features.permute(0, 2, 1, 3).reshape(
            -1, history_size, IPPArg.EMBEDDING_DIM)  #(b*g,history_size,128)

        # tim_fusion_layer
        dt_pool_inputs = dt_pool_inputs.unsqueeze(1).repeat(
            1, graph_size, 1, 1).reshape(-1, history_size, 1)
        graph_features += self.timefusion_layer(dt_pool_inputs)

        ### 使用最新特征与其他历史特征做 cross-attention
        embedded_temporal_feature: torch.Tensor = self.temporal_encoder(
            graph_features[:, -1:, :], graph_features)
        embedded_temporal_feature = embedded_temporal_feature.reshape(
            batch_size, graph_size, IPPArg.EMBEDDING_DIM)
        return embedded_temporal_feature

    def spatio_attention(self,
                         embedded_feature: torch.Tensor,
                         edge_inputs: torch.Tensor,
                         curr_index: torch.Tensor,
                         pos_encoding,
                         dist_inputs,
                         spatio_mask=None):
        '''
        Args:
            embedded_feature : (batch, graph_size, embedding_dim)
            edge_inputs: (batch, graph_size, k_size), k_size for KNN , 连接关系
            pos_encoding: (batch. graph_size, num_eigen_value ), 图的 laplace 矩阵的 eigen_value 
            curr_index: (batch, 1 , 1) curr_index in range(0,graph_size)
            dist_inputs: (batch, graph_size, 1)
            spatio_mask: 限制当前节点只能访问想连接的节点
        '''
        batch_size, graph_size, knn_size = edge_inputs.shape
        current_edge = torch.gather(
            edge_inputs, dim=1, index=curr_index.repeat(
                1, 1,
                knn_size)).long()  # 仅在 knn_size 维度repeat , (batch,1, knn_size)
        current_edge = current_edge.permute(
            0, 2, 1)  # (batch, knn_size , 1) ,1 is for current node

        if spatio_mask is None:
            mask = torch.zeros((batch_size, 1, knn_size), dtype=torch.bool)
        else:
            raise NotImplementedError
        # eigen_value 矩阵的 fature
        embedded_feature += self.spatio_pos_embedding(pos_encoding)
        #### self attention for embedded featute
        embedded_feature = self.spatio_encoder(
            embedded_feature)  # shape (batch, graph_size, embedding_dim)
        #
        embedded_feature = self.distfusion_layer(
            torch.cat((embedded_feature, dist_inputs), dim=-1))
        # 提取 node feature
        curr_node_feature = torch.gather(embedded_feature,
                                         dim=1,
                                         index=curr_index.repeat(
                                             1, 1, IPPArg.EMBEDDING_DIM))
        connected_nodes_feature = torch.gather(embedded_feature,
                                               dim=1,
                                               index=current_edge.repeat(
                                                   1, 1, IPPArg.EMBEDDING_DIM))
        embedded_spatio_feature = self.spatio_decoder(curr_node_feature,
                                                      connected_nodes_feature,
                                                      mask)
        # 做 embedded_spatio_feature 与 connected_nodes_feature 的 cross-attention, 输出 logp_list
        logp_list: torch.Tensor = self.pointer(embedded_spatio_feature,
                                               connected_nodes_feature, mask)
        logp_list = logp_list.squeeze(dim=1)  # 去除冗余维度 (1, k_size)
        value = None

        return logp_list, value

    def forward(self,
                node_inputs,
                dt_pool_inputs,
                edge_inputs,
                current_index: torch.Tensor,
                pos_encoding,
                dist_inputs,
                mask=None):
        """
        Args:
            node_inputs: (batch, history_size, graph_size,2 +  num_target * feature) , feature(mean,std)
            dt_pool_inputs: (batch, history_size, 1)
            edge_inputs: (batch, graph_size, k_size), k_size for KNN , 连接关系
            pos_encoding: (batch, graph_size, num_eigen_value) , 图的 laplace 矩阵的 eigen_value 
            current_index: (batch, 1 , 1) curr_index in range(0,graph_size)
            dist_inputs: (batch, graph_size, 1)
        """
        # 此处为 attention_net的 forward, PPO 部分在哪里？
        current_index = current_index.to(torch.int64)
        with autocast():
            embedded_feature = self.graph_embedding(node_inputs,
                                                    dt_pool_inputs,
                                                    mask=mask)
            logp_list, value = self.spatio_attention(embedded_feature,
                                                     edge_inputs,
                                                     current_index,
                                                     pos_encoding,
                                                     dist_inputs,
                                                     spatio_mask=None)
        # 不返回 value, value_net 由 stable_baselines3 自动添加
        return logp_list


if __name__ == '__main__':
    pass

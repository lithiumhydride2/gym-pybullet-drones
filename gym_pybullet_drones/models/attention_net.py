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
        return self.normalizer(input.view(-1,
                                          input.size(-1))).view(*input.size())


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

        self.node_coord_embedding = nn.Linear(IPPArg.NODE_COORD_DIM,
                                              embedding_dim)
        self.belief_embedding = nn.Linear(IPPArg.BELIEF_FEATURE_DIM,
                                          embedding_dim)
        self.target_encoder = Decoder(embedding_dim=IPPArg.EMBEDDING_DIM,
                                      n_head=IPPArg.N_HEAD,
                                      n_layer=IPPArg.N_LAYER)
        self.spatio_encoder = Encoder(embedding_dim=embedding_dim,
                                      n_head=IPPArg.N_HEAD,
                                      n_layer=IPPArg.N_LAYER)
        self.spatio_pos_embedding = nn.Linear(IPPArg.num_eigen_value,
                                              embedding_dim)
        self.spatio_decoder = Decoder(
            embedding_dim=IPPArg.EMBEDDING_DIM,
            n_head=IPPArg.N_HEAD,
            n_layer=IPPArg.N_LAYER
        )  # 用作 node_feature 与 connect_node_feature 的 cross-attention
        self.pointer = SingleHeadAttention(embedding_dim)
        self.value_output = nn.Linear(embedding_dim, 1)

        ## 以上为再次添加
        self.initial_embedding = nn.Linear(
            2, embedding_dim)  # layer for non-end position
        self.end_embedding = nn.Linear(
            2, embedding_dim)  # embedding layer for end position
        self.budget_embedding = nn.Linear(embedding_dim + 2, embedding_dim)

        if IPPArg.k_size >= 8:
            self.pos_embedding = nn.Linear(32, embedding_dim)
        else:
            self.pos_embedding = nn.Linear(IPPArg.k_size, embedding_dim)
        # self.nodes_update_layers = nn.ModuleList([DecoderLayer(embedding_dim, 8) for i in range(3)])

        # self.current_embedding = nn.Linear(embedding_dim*2, embedding_dim)
        self.current_embedding = nn.Linear(embedding_dim, embedding_dim)

        self.encoder = Encoder(embedding_dim=embedding_dim,
                               n_head=4,
                               n_layer=1)
        self.decoder = Decoder(embedding_dim=embedding_dim,
                               n_head=4,
                               n_layer=1)

        self.LSTM = nn.LSTM(embedding_dim, embedding_dim, batch_first=True)

    def graph_embedding(self, node_inputs: torch.Tensor, mask=None):
        """
        Args:
            node_inputs: (batch, graph_size, 2 + num_target * feature) , feature(mean,std) , 前两个维度为 node_coords
            
        """

        batch_size, graph_size, input_dim = node_inputs.shape
        target_num = (input_dim - 2) // IPPArg.BELIEF_FEATURE_DIM
        # reshape node_inputs , 消除 graph_size 的 dim
        node_inputs = node_inputs.reshape(-1, 1, input_dim)
        # node_coord_embedding 最后一个维度的前两部分为 node_coord
        node_coord_embedding = self.node_coord_embedding(
            node_inputs[:, :, :2])  #(batch,graph_size,embedding_dim)
        # target belief embedding
        target_belief_embedding = torch.cat(
            [
                self.belief_embedding(
                    node_inputs[:, :, 2 + i * IPPArg.BELIEF_FEATURE_DIM:2 +
                                (1 + i) * IPPArg.BELIEF_FEATURE_DIM])
                for i in range(target_num)
            ],
            dim=1)  # (batch, graph_size  * target_num, embedding_dim)

        # embedded_feature
        embedded_feature: torch.Tensor = self.target_encoder(
            node_coord_embedding, target_belief_embedding)
        embedded_feature = embedded_feature.reshape(batch_size, graph_size,
                                                    IPPArg.EMBEDDING_DIM)

        return embedded_feature

    def select_next_node(self, embedding_feature, edge_inputs, budget_inputs,
                         current_index, LSTM_h, LSTM_c, mask):
        LSTM_h = LSTM_h.permute(1, 0, 2)
        LSTM_c = LSTM_c.permute(1, 0, 2)

        batch_size = edge_inputs.size()[0]
        sample_size = edge_inputs.size()[1]
        k_size = edge_inputs.size()[2]
        current_edge = torch.gather(edge_inputs, 1,
                                    current_index.repeat(1, 1, k_size))
        # print(current_edge)
        current_edge = current_edge.permute(0, 2, 1)
        embedding_dim = embedding_feature.size()[2]

        th = torch.FloatTensor([IPPArg.ADAPTIVE_TH
                                ]).unsqueeze(0).unsqueeze(0).repeat(
                                    batch_size, sample_size,
                                    1).to(embedding_feature.device)

        embedding_feature = self.budget_embedding(
            torch.cat((embedding_feature, budget_inputs, th), dim=-1))
        connected_nodes_feature = torch.gather(
            embedding_feature, 1, current_edge.repeat(1, 1, embedding_dim))

        connected_nodes_budget = torch.gather(budget_inputs, 1, current_edge)
        # print(embedding_feature)
        # print(connected_nodes_feature)
        current_node_feature = torch.gather(
            embedding_feature, 1, current_index.repeat(1, 1, embedding_dim))
        current_node_feature, (LSTM_h,
                               LSTM_c) = self.LSTM(current_node_feature,
                                                   (LSTM_h, LSTM_c))

        # end_node_feature = embedding_feature[:,0,:].unsqueeze(1)
        # current_node_feature = torch.cat((current_node_feature,end_node_feature),dim=-1)
        current_node_feature = self.current_embedding(current_node_feature)
        # print(current_node_feature)
        if mask is not None:
            current_mask = torch.gather(mask, 1,
                                        current_index.repeat(1, 1, k_size)).to(
                                            embedding_feature.device)
            # print(current_mask)
        else:
            current_mask = None
            current_mask = torch.zeros(
                (batch_size, 1, k_size),
                dtype=torch.int64).to(embedding_feature.device)
        one = torch.ones_like(current_mask,
                              dtype=torch.int64).to(embedding_feature.device)
        #print(connected_nodes_budget)
        current_mask = torch.where(
            connected_nodes_budget.permute(0, 2, 1) > 0, current_mask, one)
        current_mask[:, :, 0] = 1  # don't stay at current position
        # assert 0 in current_mask
        #print(current_mask)

        # connected_nodes_feature = self.encoder(connected_nodes_feature, current_mask)
        current_feature_prime = self.decoder(current_node_feature,
                                             connected_nodes_feature,
                                             current_mask)
        logp_list = self.pointer(current_feature_prime,
                                 connected_nodes_feature, current_mask)
        logp_list = logp_list.squeeze(1)
        value = self.value_output(current_feature_prime)

        LSTM_h = LSTM_h.permute(1, 0, 2)
        LSTM_c = LSTM_c.permute(1, 0, 2)

        return logp_list, value, LSTM_h, LSTM_c

    def spatio_attention(self,
                         embedded_feature: torch.Tensor,
                         edge_inputs: torch.Tensor,
                         curr_index: torch.Tensor,
                         pos_encoding,
                         spatio_mask=None):
        '''
        Args:
            embedded_feature : (batch, graph_size, embedding_dim)
            edge_inputs: (batch, graph_size, k_size), k_size for KNN , 连接关系
            pos_encoding: (batch. graph_size, num_eigen_value ), 图的 laplace 矩阵的 eigen_value 
            curr_index: (batch, 1 , 1) curr_index in range(0,graph_size)
        '''
        batch_size, graph_size, knn_size = edge_inputs.shape
        current_edge = torch.gather(
            edge_inputs, dim=1, index=curr_index.repeat(
                1, 1, knn_size))  # 仅在 knn_size 维度repeat , (batch,1, knn_size)
        current_edge = current_edge.permute(
            0, 2, 1)  # (batch, knn_size , 1) ,1 is for current node

        if spatio_mask is None:
            mask = torch.zeros((batch_size, 1, knn_size), dtype=torch.bool)
        else:
            pass
        # eigen_value 矩阵的 fature
        embedded_feature += self.spatio_pos_embedding(pos_encoding)
        # self attention for embedded featute
        embedded_feature = self.spatio_encoder(embedded_feature)

        ## dist_inputs todo, 此处未添加

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
        logp_list = logp_list.squeeze(dim=1)  # 去除冗余维度
        value = self.value_output(embedded_spatio_feature)

        return logp_list, value

    def forward(self,
                node_inputs,
                edge_inputs,
                current_index: torch.Tensor,
                pos_encoding,
                mask=None):
        """
        Args:
            node_inputs: (batch, graph_size, num_target * feature) , feature(mean,std)
            edge_inputs: (batch, graph_size, k_size), k_size for KNN , 连接关系
            pos_encoding: (batch, graph_size, num_eigen_value) , 图的 laplace 矩阵的 eigen_value 
            current_index: (batch, 1 , 1) curr_index in range(0,graph_size)
        """
        # 此处为 attention_net的 forward, PPO 部分在哪里？
        current_index = current_index.to(torch.int64)
        with autocast():
            embedded_feature = self.graph_embedding(node_inputs, mask=mask)
            logp_list, value = self.spatio_attention(embedded_feature,
                                                     edge_inputs,
                                                     current_index,
                                                     pos_encoding,
                                                     spatio_mask=None)
        return logp_list, value


def padding_inputs(inputs):
    seq = pad_sequence(inputs, batch_first=False, padding_value=1)
    seq = seq.permute(2, 1, 0)
    mask = torch.zeros_like(seq, dtype=torch.int64)
    ones = torch.ones_like(seq, dtype=torch.int64)
    mask = torch.where(seq != 1, mask, ones)
    # print(mask)
    # print(seq.size())
    return seq, mask


if __name__ == '__main__':
    model = AttentionNet(2, 8, greedy=True)
    node_inputs = torch.torch.rand((128, 10, 2))
    # print(node_inputs)
    edge_inputs = torch.randint(0, 10, (128, 10, 5))
    edge_inputs_list = []
    # for i in range(edge_inputs.size()[1]):
    #     edge_inputs_list.append(edge_inputs[:,i].permute(1,0))
    # edge_inputs_list.append(torch.randint(0, 10, (8, 1)))
    # edge_inputs, mask = padding_inputs(edge_inputs_list)
    current_index = torch.ones(size=(128, 1, 1), dtype=torch.int64)
    next_node, logp_list, value = model(node_inputs, edge_inputs,
                                        current_index)
    print(next_node.size())
    print(logp_list.size())
    print(value.size())

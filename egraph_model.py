import re
import os
import json
import logging
from collections import defaultdict
from egraph_data import EGraphData
from dag_greedy import greedy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor, t


class BatchedLinear(nn.Module):

    def __init__(self, batch_size, in_features, out_features):
        super().__init__()
        self.batch_size = batch_size
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(
            torch.rand(batch_size, in_features, out_features))
        self.bias = nn.Parameter(torch.rand(batch_size, out_features))

    def forward(self, x):
        return torch.bmm(x, self.weight) + self.bias


class BaseEGraph(nn.Module, EGraphData):

    def __init__(self,
                 input_file,
                 hidden_dim=32,
                 num_attributes=5,
                 gumbel_tau=1.0,
                 dropout=0.0,
                 eps=0.5,
                 batch_size=None,
                 soft=False,
                 embedding_type='lookup',
                 aggregate_type='mean',
                 device='cuda',
                 load_cost=False,
                 greedy_ini=False,
                 compress=False,
                 drop_self_loops=False,
                 share_proj=True):
        nn.Module.__init__(self)
        EGraphData.__init__(
            self,
            input_file,
            hidden_dim,
            load_cost,
            compress,
            drop_self_loops,
            device,
        )
        self.hidden_dim = hidden_dim
        self.num_attributes = num_attributes
        self.gumbel_tau = gumbel_tau
        self.dropout = dropout
        self.eps = eps
        self.embedding_type = embedding_type
        self.aggregate_type = aggregate_type
        self.device = device
        self.minimal_cost = None
        self.soft = soft
        self.p_number = np.e
        self.quadratic_cost = None
        self.greedy_ini = greedy_ini
        self.input_file = input_file
        self.share_proj = share_proj

        # BSC batch_size夹在 1 - 512 之间，但是现在已经采取了固定值64，猜测是这里的问题，打算看一下
        if batch_size is None:
            # self.batch_size = 100_000_000 / len(self.eclasses)**2
            # print(f'eclass size: {len(self.eclasses)}')
            num_class = len(self.eclasses)
            if num_class > 5000:
                batch_size = 40000 / num_class
            else:
                batch_size = 80000 / num_class
            self.batch_size = int(np.clip(2**int(np.log2(batch_size)), 1, 512))
            # print(f'Auto set batch size to {self.batch_size}')

        self.set_to_matrix()
        self.init_embedding()
        self.init_params()
        torch.cuda.empty_cache()

    def set_adj(self):
        # [N, M]
        # Enode adjacency matrix points from enode to eclass
        self.node_adj = torch.zeros((len(self.enodes), len(self.eclasses)),
                                    dtype=torch.bool,
                                    device=self.device)

        # [M, N]
        # eclass adjacency matrix contains the enodes in the eclass
        self.class_adj = torch.zeros((len(self.eclasses), len(self.enodes)),
                                     dtype=torch.bool,
                                     device=self.device)

        for enode_id in self.enodes:
            eclass_id = [i for i in self.enodes[enode_id].eclass_id]
            self.node_adj[enode_id, eclass_id] = 1

        for eclass_id in self.eclasses:
            self.class_adj[eclass_id, self.eclasses[eclass_id].enode_id] = 1

    from torch_sparse import SparseTensor
    # 主要改变了不识别的dtype
    def set_to_matrix(self):
        import torch
        device = self.device
        print("[CHECK] enter set_to_matrix file=", __file__, flush=True)

        # —— 1) 固定顺序 & 连续映射（防止形状/对齐问题） ——
        enode_ids  = list(self.enodes.keys())
        eclass_ids = list(self.eclasses.keys())

        print("[DEBUG] self.root_eclasses =", getattr(self, "root_eclasses", None), flush=True)
        print("[DEBUG] self.eclasses keys =", list(self.eclasses.keys())[:10], flush=True)
        print("[DEBUG] self.enodes keys =", list(self.enodes.keys())[:10], flush=True)
        # 如你的 key 是 "129.0" 这种字符串，也能被 as_tensor+long 吃掉，这里不强行排序
        N, M = len(enode_ids), len(eclass_ids)

        node_old2new = {nid: i for i, nid in enumerate(enode_ids)}
        cls_old2new  = {cid: j for j, cid in enumerate(eclass_ids)}

        # —— 2A) node→class（孩子一对多：训练/传播用） ——
        rows_many, cols_many = [], []
        for nid in enode_ids:
            row = node_old2new[nid]
            cids = getattr(self.enodes[nid], "eclass_id", [])
            # 兼容 int / list
            if isinstance(cids, (int, float, str)):
                cids = [cids]
            for cid in cids:
                # 某些数据里是 "12.0" / numpy 标量：统一转为 int 再映射
                try:
                    j = cls_old2new.get(cid)
                    if j is None:
                        j = cls_old2new.get(int(cid))
                    if j is None:
                        j = cls_old2new.get(int(float(cid)))
                except Exception:
                    j = cls_old2new.get(cid)
                if j is not None:
                    rows_many.append(row)
                    cols_many.append(j)

        if len(rows_many) == 0:
            row_many = torch.empty(0, dtype=torch.long, device=device)
            col_many = torch.empty(0, dtype=torch.long, device=device)
            val_many = torch.empty(0, dtype=torch.float32, device=device)
        else:
            row_many = torch.as_tensor(rows_many, dtype=torch.long, device=device)
            col_many = torch.as_tensor(cols_many, dtype=torch.long, device=device)
            val_many = torch.ones(row_many.numel(), dtype=torch.float32, device=device)

        node2class = SparseTensor(row=row_many,
                                col=col_many,
                                value=val_many,
                                sparse_sizes=(N, M)).coalesce()

        # —— 2B) belong（节点隶属的一对一：set_index 用） ——
        rows_belong, cols_belong = [], []
        for nid in enode_ids:
            row = node_old2new[nid]
            belong = getattr(self.enodes[nid], "belong_eclass_id", None)
            if belong is None:
                continue
            # 同样宽松解析为 int
            j = None
            try:
                j = cls_old2new.get(belong)
                if j is None:
                    j = cls_old2new.get(int(belong))
                if j is None:
                    j = cls_old2new.get(int(float(belong)))
            except Exception:
                j = cls_old2new.get(belong)
            if j is not None:
                rows_belong.append(row)
                cols_belong.append(j)

        if len(rows_belong) == 0:
            row_belong = torch.empty(0, dtype=torch.long, device=device)
            col_belong = torch.empty(0, dtype=torch.long, device=device)
            val_belong = torch.empty(0, dtype=torch.float32, device=device)
        else:
            row_belong = torch.as_tensor(rows_belong, dtype=torch.long, device=device)
            col_belong = torch.as_tensor(cols_belong, dtype=torch.long, device=device)
            val_belong = torch.ones(row_belong.numel(), dtype=torch.float32, device=device)

        class2node = SparseTensor(row=col_belong,  # 转置
                                col=row_belong,
                                value=val_belong,
                                sparse_sizes=(M, N)).coalesce()

        # —— 3) 回写 ——
        self.node2class = node2class
        self.class2node = class2node
        self.node2classT = node2class.t()


        # print(f"[DBG] node2class nnz={self.node2class.nnz()}", flush=True)
        # print(f"[DBG] class2node nnz={self.class2node.nnz()}", flush=True)
        # # 若已知 129.0 -> nid
        # nid = node_old2new.get("129.0", None)
        # if nid is not None:
        #     # 取该行的切片（node→class）
        #     row, col, val = self.node2class.coo()
        #     cols = col[row == nid].tolist()
        #     print(f"[DBG] node2class row for nid={nid} (key='129.0'): target classes={cols}", flush=True)

        print("[CHECK] leave set_to_matrix", flush=True)


    def init_embedding(self):
        self.embedding = torch.rand(self.batch_size,
                                    len(self.enodes),
                                    self.hidden_dim,
                                    device=self.device)
        self.embedding = self.embedding / np.sqrt(self.hidden_dim)
        if self.greedy_ini:
            self.bias = torch.zeros(len(self.enodes)).to(self.device)
            greedy_idx_list = greedy(self, method="faster", ini_greedy=True)
            self.bias[greedy_idx_list] += self.eps
        self.embedding = nn.Parameter(self.embedding, requires_grad=True)

    def init_params(self):
        if self.embedding_type == 'projection':
            self.emb_layer = torch.nn.Linear(in_features=self.num_attributes,
                                             out_features=self.hidden_dim)
        elif self.embedding_type == 'lookup':
            self.emb_layer = torch.nn.Embedding(num_embeddings=len(
                self.enodes),
                                                embedding_dim=self.hidden_dim)
        if self.share_proj:
            self.enode_proj = torch.nn.Linear(in_features=self.hidden_dim,
                                              out_features=self.hidden_dim)
            self.output_proj = torch.nn.Linear(in_features=self.hidden_dim,
                                               out_features=1)
        else:
            self.node_proj = BatchedLinear(self.batch_size, self.hidden_dim,
                                           self.hidden_dim)
            self.output_proj = BatchedLinear(self.batch_size, self.hidden_dim,
                                             1)
        self.activation = torch.nn.Sequential(
            torch.nn.LayerNorm(self.hidden_dim),
            torch.nn.ReLU(),
        )
        self.dropout = torch.nn.Dropout(p=self.dropout)
        self.set_attr(self.num_attributes)

    def set_temperature_schedule(self, steps, schedule='constant'):
        if schedule == 'constant':
            self.temperature_schedule = np.ones(steps)
        if schedule == 'linear':
            self.temperature_schedule = np.linspace(1, 1e-3, steps)
        elif schedule == 'log':
            self.temperature_schedule = np.logspace(0, -2, steps)
        self.step_count = 0

    def dense(self, enode_embedding, context_embedding):
        # enode_embedding: [n, hidden_dim]
        # context_embedding: [1, hidden_dim]
        # return: [n, hidden_dim]
        return self.enode_proj(enode_embedding) + self.context_proj(
            context_embedding)

    def projection(self, enode_embedding):
        # enode_embedding: [n, hidden_dim]
        # return: [n]
        return self.output_proj(enode_embedding).squeeze(-1)

    def forward_embedding(self, embedding):
        logit = self.activation(self.enode_proj(embedding))
        logit = self.projection(self.dropout(logit))
        if self.greedy_ini and not self.training and self.step_count == 0:
            logit[0] = logit[0] + self.bias
        return logit

    def set_attr(self, num_attributes):
        if self.embedding_type == 'projection':
            self.enode_attr = torch.randn(len(self.enodes),
                                          num_attributes).to(self.device)
        elif self.embedding_type == 'lookup':
            self.enode_attr = torch.arange(len(self.enodes)).to(self.device)


class AdhocEGraph(BaseEGraph):

    def step(self):
        for eclass in self.eclasses.values():
            eclass.visited_in_nodes = defaultdict(int)
            eclass.included = False
            eclass.predecessor_embeddings = []
            eclass.predecessor_enodes = []
            # 就是选中的class中的node被选中的概率归一化为1
            eclass.normalized_prob = 1
        for enode in self.enodes.values():
            enode.normalized_prob = 0
        self.gumbel_tau = self.temperature_schedule[self.step_count]
        self.step_count += 1

    def get_enode_embedding(self, enode_ids):
        # Sum the embedded attribute over the categories
        # TODO: Consider changing simple sum to projection
        if self.embedding_type == 'projection':
            return self.emb_layer(self.enode_attr[enode_ids])
        elif self.embedding_type == 'lookup':
            return self.emb_layer(self.enode_attr[enode_ids])

    def aggregate_eclass_context(self, eclass):
        if eclass.predecessor_embeddings == []:
            return self.start_embedding
        if self.aggregate_type == 'sum':
            return torch.stack(
                eclass.predecessor_embeddings).sum(dim=0).reshape(1, -1)
        elif self.aggregate_type == 'mean':
            return torch.stack(
                eclass.predecessor_embeddings).mean(dim=0).reshape(1, -1)
        elif self.aggregate_type == 'max':
            return torch.stack(eclass.predecessor_embeddings).max(
                dim=0).values.reshape(1, -1)
        elif self.aggregate_type == 'none':
            return self.start_embedding
        else:
            raise NotImplementedError

    def sample(self):
        if self.soft:
            return self.soft_sample()
        else:
            return self.hard_sample()

    # TODO: make the sampling process batched
    def hard_sample(self):
        # implement depth optimization for hard
        selected = {}
        to_visit = []
        visited = set()

        # initialize to_visit with all source eclasses
        for eclass_id in self.eclasses:
            # 这个innode就是说的入度，也就是说所有的root class
            if len(self.eclasses[eclass_id].in_nodes) == 0:
                to_visit.append(eclass_id)
                self.eclasses[eclass_id].included = True

        while to_visit:
            included = [
                eclass for eclass in self.eclasses
                if self.eclasses[eclass].included
            ]
            # logging.info(f'To visit: {to_visit}')
            # logging.info(f'Included: {included}')
            eclass_id = to_visit.pop()
            if eclass_id in visited:
                continue

            eclass = self.eclasses[eclass_id]
            # TODO: wrap the following in a function
            # 得到node对应的隐向量
            enode_embedding = self.get_enode_embedding(eclass.enode_id)
            # 得到pre node对应的上下文向量
            context_embedding = self.aggregate_eclass_context(eclass)
            # activation和dropout得到更新后的node表示
            updated_embedding = self.dropout(
                self.activation(self.dense(enode_embedding,
                                           context_embedding)))
            # 未归一化的enode cost
            logits = self.projection(updated_embedding)
            logging.info(f'visiting eclass {eclass_id} with logits {logits}')
            # 这里的softmax不明
            if self.training:
                choice = F.gumbel_softmax(logits,
                                          tau=self.gumbel_tau,
                                          hard=True)
            else:
                # 不训练就直接用argmax得到one-hot向量
                choice = torch.eq(logits, logits.max()).float()
            selected[eclass_id] = choice
            # 找到全局索引
            sg_choice = eclass.enode_id[choice.argmax()]

            # 标记为已访问
            visited.add(eclass_id)
            for out_eclass in self.enodes[sg_choice].eclass_id:
                self.eclasses[out_eclass].included = True

                # 更新累计激活概率，但是硬编码是不变的
                self.eclasses[out_eclass].normalized_prob *= choice.max()

                # propagate with gumbel softmax signal
                # embedding 加入下游的上下文信息
                self.eclasses[out_eclass].predecessor_embeddings.append(
                    choice.matmul(updated_embedding))

                # propagate without gumbel softmax signal
                # self.eclasses[out_eclass].predecessor_embeddings.append(
                #     updated_embedding[choice.argmax()])

            # BFS，但是下一层的node只有所有入度都被遍历完才能进入新的遍历队列
            for enode in eclass.enode_id:
                for out_eclass in self.enodes[enode].eclass_id:
                    self.eclasses[out_eclass].add_visited_in_node(enode)
                    # If all in_nodes are visited, add to to_visit
                    if self.eclasses[out_eclass].in_nodes == self.eclasses[
                            out_eclass].visited_in_nodes and self.eclasses[
                                out_eclass].included and out_eclass not in visited:
                        to_visit.append(out_eclass)

            if len(to_visit) == 0:
                # Find the an included eclass with the least number of unvisited in_nodes
                min_unvisited_in_nodes = float('inf')
                candidate_eclasses = None
                for eclass_id in self.eclasses:
                    if self.eclasses[eclass_id].included:
                        # 在这里更新没有遍历到的node
                        unvisited_in_nodes = len(
                            self.eclasses[eclass_id].in_nodes) - len(
                                self.eclasses[eclass_id].visited_in_nodes)
                        if unvisited_in_nodes < min_unvisited_in_nodes and unvisited_in_nodes > 0 and eclass_id not in visited:
                            min_unvisited_in_nodes = unvisited_in_nodes
                            candidate_eclasses = eclass_id
                if candidate_eclasses is not None:
                    to_visit.append(candidate_eclasses)

        logging.info(f'Selected: {selected}')
        return selected

    def soft_sample(self):
        selected = {}
        to_visit = []
        visited = set()

        # initialize to_visit with all source eclasses
        # same to the former
        for eclass_id in self.eclasses:
            if len(self.eclasses[eclass_id].in_nodes) == 0:
                to_visit.append(eclass_id)
                self.eclasses[eclass_id].included = True

        while to_visit:
            included = [
                eclass for eclass in self.eclasses
                if self.eclasses[eclass].included
            ]
            # logging.info(f'To visit: {to_visit}')
            # logging.info(f'Included: {included}')
            eclass_id = to_visit.pop()
            if eclass_id in visited:
                continue

            eclass = self.eclasses[eclass_id]
            # TODO: wrap the following in a function
            enode_embedding = self.get_enode_embedding(eclass.enode_id)
            context_embedding = self.aggregate_eclass_context(eclass)
            # use context embedding to update enode embedding
            updated_embedding = self.dropout(
                self.activation(self.dense(enode_embedding,
                                           context_embedding)))
            logits = self.projection(updated_embedding)
            logging.info(f'visiting eclass {eclass_id} with logits {logits}')
            if self.training:
                choice = F.gumbel_softmax(logits,
                                          tau=self.gumbel_tau,
                                          hard=False)

                # using softmax instead of gumbel softmax
                # choice = F.softmax(logits, dim=-1)
            else:
                # softmax = F.softmax(logits, dim=-1)
                # print(f'max logits: {softmax.max()}')
                choice = torch.eq(logits, logits.max()).float()
            selected[eclass_id] = choice
            # if isinstance(eclass.normalized_prob, torch.Tensor):
            #     eclass.normalized_prob = 1 - eclass.normalized_prob
            class_depth = 0
            class_prob = 1
            # 有入度才需要继承父节点的信息
            if len(eclass.predecessor_enodes) != 0:
                for enode in eclass.predecessor_enodes:
                    # 计算所有前驱都没激活的概率
                    class_prob *= 1 - self.enodes[enode].normalized_prob
                    class_depth += self.enodes[enode].depth
                # 更新成 1 - 所有前驱都趋势的情况， 1 - ∏(1 - p_i)，这样用来表示至少有一个前驱被选中的概率
                class_prob = 1 - class_prob
                # 深度的平均值，不知道目前深度作用是什么
                class_depth /= len(eclass.predecessor_enodes)
            #真正被选中的enode 
            sg_choice = eclass.enode_id[choice.argmax()]

            # 防止重复访问
            visited.add(eclass_id)

            for i, enode in enumerate(eclass.enode_id):
                if self.enodes[enode].normalized_prob == 0:
                    self.enodes[enode].normalized_prob = choice[i] * class_prob
                    self.enodes[enode].depth = class_depth + 1
                else:
                    raise ValueError
                for out_eclass in self.enodes[enode].eclass_id:
                    # 加入遍历队列
                    self.eclasses[out_eclass].included = True
                    # 加入前驱节点
                    self.eclasses[out_eclass].predecessor_enodes.append(enode)
                    # 把上下文信息均加入predecessor
                    # propagate with gumbel softmax signal
                    self.eclasses[out_eclass].predecessor_embeddings.append(
                        choice.matmul(updated_embedding))
            # 后面逻辑是完全相同的
            for enode in eclass.enode_id:
                for out_eclass in self.enodes[enode].eclass_id:
                    self.eclasses[out_eclass].add_visited_in_node(enode)
                    # If all in_nodes are visited, add to to_visit
                    if self.eclasses[out_eclass].in_nodes == self.eclasses[
                            out_eclass].visited_in_nodes and self.eclasses[
                                out_eclass].included and out_eclass not in visited:
                        to_visit.append(out_eclass)

            if len(to_visit) == 0:
                # Find the an included eclass with the least number of unvisited in_nodes
                min_unvisited_in_nodes = float('inf')
                candidate_eclasses = None
                for eclass_id in self.eclasses:
                    if self.eclasses[eclass_id].included:
                        unvisited_in_nodes = len(
                            self.eclasses[eclass_id].in_nodes) - len(
                                self.eclasses[eclass_id].visited_in_nodes)
                        if unvisited_in_nodes < min_unvisited_in_nodes and unvisited_in_nodes > 0 and eclass_id not in visited:
                            min_unvisited_in_nodes = unvisited_in_nodes
                            candidate_eclasses = eclass_id
                if candidate_eclasses is not None:
                    to_visit.append(candidate_eclasses)

        logging.info(f'Selected: {selected}')
        return selected

    def forward(self, selected, backward=True, optim_goal='sum', debug=False):
        assert not debug
        # use forward to compute loss
        cost = 0
        cost_per_node = self.cost_per_node
        # 如果是硬采样，也就是传统的{0,1}
        if not self.soft:
            for eclass_id in selected:
                enode_id = self.eclasses[eclass_id].enode_id
                enode_weight = cost_per_node[enode_id]
                # 硬采样中，一个类里面只能选中一个，选中的概率是1
                # chain在这里的作用是方便和后面soft_sample的逻辑统一,
                chain = self.eclasses[eclass_id].normalized_prob
                assert chain == 1 or chain.item() == 1
                # 先乘再求和，Σ node * cost_of_node 
                cost += chain * torch.matmul(selected[eclass_id], enode_weight)
        else:
            for enode in self.enodes:
                enode_weight = cost_per_node[enode]
                if optim_goal == 'depth':
                    enode_weight = self.p_number**self.enodes[enode].depth
                # 这里也是用的的归一化概率
                cost += enode_weight * self.enodes[enode].normalized_prob
        loss = cost
        if backward:
            # 反向传播
            loss.backward()
        # 张量变成float?标量，释放内存，便于logging
        loss = loss.item()
        return cost

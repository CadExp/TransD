#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import average_precision_score

from torch.utils.data import DataLoader

from dataloader import TestDataset

class KGEModel(nn.Module):
    def __init__(self, ent_tot, rel_tot, dim_e=50, dim_r=50, p_norm=1,
                 norm_flag=True, margin=1.0, epsilon=2.0):
        super(KGEModel, self).__init__()
        self.ent_tot = ent_tot
        self.rel_tot = rel_tot
        self.dim_e = dim_e
        self.dim_r = dim_r
        self.margin = margin
        self.epsilon = epsilon
        self.norm_flag = norm_flag
        self.p_norm = p_norm

        self.ent_embeddings = nn.Parameter(torch.zeros(ent_tot, self.dim_e))
        self.rel_embeddings = nn.Parameter(torch.zeros(rel_tot, self.dim_r))
        self.ent_transfer = nn.Parameter(torch.zeros(ent_tot, self.dim_e))
        self.rel_transfer = nn.Parameter(torch.zeros(rel_tot, self.dim_r))
        self.ent_embedding_range = nn.Parameter(
            torch.Tensor([(self.margin + self.epsilon) / self.dim_e]), requires_grad=False
        )
        self.rel_embedding_range = nn.Parameter(
            torch.Tensor([(self.margin + self.epsilon) / self.dim_r]), requires_grad=False
        )
        nn.init.uniform_(
            tensor=self.ent_embeddings,
            a=-self.ent_embedding_range.item(),
            b=self.ent_embedding_range.item()
        )
        nn.init.uniform_(
            tensor=self.rel_embeddings,
            a=-self.rel_embedding_range.item(),
            b=self.rel_embedding_range.item()
        )
        nn.init.uniform_(
            tensor=self.ent_transfer,
            a=-self.ent_embedding_range.item(),
            b=self.ent_embedding_range.item()
        )
        nn.init.uniform_(
            tensor=self.rel_transfer,
            a=-self.rel_embedding_range.item(),
            b=self.rel_embedding_range.item()
        )
        if margin != None:
            self.margin = nn.Parameter(torch.Tensor([margin]))
            self.margin.requires_grad = False
            self.margin_flag = True
        else:
            self.margin_flag = False

    def _resize(self, tensor, axis, size):
        shape = tensor.size()
        osize = shape[axis]
        if osize == size:
            return tensor
        if (osize > size):
            return torch.narrow(tensor, axis, 0, size)
        paddings = []
        for i in range(len(shape)):
            if i == axis:
                paddings = [0, size - osize] + paddings
            else:
                paddings = [0, 0] + paddings
        print(paddings)
        return F.pad(tensor, paddings=paddings, mode="constant", value=0)

    def _calc(self, h, t, r, mode):
        if self.norm_flag:
            h = F.normalize(h, 2, -1)
            r = F.normalize(r, 2, -1)
            t = F.normalize(t, 2, -1)
        if mode != 'normal':
            h = h.view(-1, r.shape[0], h.shape[-1])
            t = t.view(-1, r.shape[0], t.shape[-1])
            r = r.view(-1, r.shape[0], r.shape[-1])
        if mode == 'head_batch':
            score = h + (r - t)
        else:
            score = (h + r) - t
        score = torch.norm(score, self.p_norm, -1).flatten()
        return score

    def _transfer(self, e, e_transfer, r_transfer):
        if e.shape[0] != r_transfer.shape[0]:
            e = e.view(-1, r_transfer.shape[0], e.shape[-1])
            e_transfer = e_transfer.view(-1, r_transfer.shape[0], e_transfer.shape[-1])
            r_transfer = r_transfer.view(-1, r_transfer.shape[0], r_transfer.shape[-1])
            e = F.normalize(
                self._resize(e, -1, r_transfer.size()[-1]) + torch.sum(e * e_transfer, -1, True) * r_transfer,
                p=2,
                dim=-1
            )
            return e.view(-1, e.shape[-1])
        else:
            return F.normalize(
                self._resize(e, -1, r_transfer.size()[-1]) + torch.sum(e * e_transfer, -1, True) * r_transfer,
                p=2,
                dim=-1
            )

    def forward(self, sample, mode='normal'):
        if mode == 'normal':
            batch_size, negative_sample_size = sample.size(0), 1

            head = torch.index_select(
                self.ent_embeddings,
                dim=0,
                index=sample[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(
                self.rel_embeddings,
                dim=0,
                index=sample[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.ent_embeddings,
                dim=0,
                index=sample[:, 2]
            ).unsqueeze(1)
            h_transfer = torch.index_select(
                self.ent_transfer,
                dim=0,
                index=sample[:, 0]
            ).unsqueeze(1)

            r_transfer = torch.index_select(
                self.rel_transfer,
                dim=0,
                index=sample[:, 1]
            ).unsqueeze(1)

            t_transfer = torch.index_select(
                self.ent_transfer,
                dim=0,
                index=sample[:, 2]
            ).unsqueeze(1)

        elif mode == 'head-batch':
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)

            head = torch.index_select(
                self.ent_embeddings,
                dim=0,
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

            relation = torch.index_select(
                self.rel_embeddings,
                dim=0,
                index=tail_part[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.ent_embeddings,
                dim=0,
                index=tail_part[:, 2]
            ).unsqueeze(1)
            h_transfer = torch.index_select(
                self.ent_transfer,
                dim=0,
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

            r_transfer = torch.index_select(
                self.rel_transfer,
                dim=0,
                index=tail_part[:, 1]
            ).unsqueeze(1)

            t_transfer = torch.index_select(
                self.ent_transfer,
                dim=0,
                index=tail_part[:, 2]
            ).unsqueeze(1)
        elif mode == 'tail-batch':

            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)
            head = torch.index_select(
                self.ent_embeddings,
                dim=0,
                index=head_part[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(
                self.rel_embeddings,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.ent_embeddings,
                dim=0,
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)
            h_transfer = torch.index_select(
                self.ent_transfer,
                dim=0,
                index=head_part[:, 0]
            ).unsqueeze(1)

            r_transfer = torch.index_select(
                self.rel_transfer,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)

            t_transfer = torch.index_select(
                self.ent_transfer,
                dim=0,
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

        elif mode == 'relation-batch':

            entity_part, relation_part = sample
            batch_size, negative_sample_size = relation_part.size(0), relation_part.size(1)

            head = torch.index_select(
                self.ent_embeddings,
                dim=0,
                index=entity_part[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(
                self.rel_embeddings,
                dim=0,
                index=relation_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

            tail = torch.index_select(
                self.ent_embeddings,
                dim=0,
                index=entity_part[:, 2]
            ).unsqueeze(1)
            h_transfer = torch.index_select(
                self.ent_transfer,
                dim=0,
                index=entity_part[:, 0]
            ).unsqueeze(1)

            r_transfer = torch.index_select(
                self.rel_transfer,
                dim=0,
                index=relation_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

            t_transfer = torch.index_select(
                self.ent_transfer,
                dim=0,
                index=entity_part[:, 2]
            ).unsqueeze(1)

        else:
            raise ValueError('mode %s not supported' % mode)
        '''batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        h_transfer = self.ent_transfer(batch_h)
        t_transfer = self.ent_transfer(batch_t)
        r_transfer = self.rel_transfer(batch_r)'''
        head = self._transfer(head, h_transfer, r_transfer)
        tail = self._transfer(tail, t_transfer, r_transfer)
        score = self._calc(head, tail, relation, mode)
        if self.margin_flag:
            return self.margin - score
        else:
            return score

    '''def regularization(self, data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        h_transfer = self.ent_transfer(batch_h)
        t_transfer = self.ent_transfer(batch_t)
        r_transfer = self.rel_transfer(batch_r)
        regul = (torch.mean(h ** 2) +
                 torch.mean(t ** 2) +
                 torch.mean(r ** 2) +
                 torch.mean(h_transfer ** 2) +
                 torch.mean(t_transfer ** 2) +
                 torch.mean(r_transfer ** 2)) / 6
        return regul'''

    @staticmethod
    def train_step(model, optimizer, train_iterator, isCUDA):

        model.train()
        optimizer.zero_grad()
        positive_sample, negative_sample, subsampling_weight, mode = next(train_iterator)

        positive_sample = positive_sample.cuda()
        negative_sample = negative_sample.cuda()
        subsampling_weight = subsampling_weight.cuda()
        negative_score = model((positive_sample, negative_sample), mode=mode)
        negative_score = F.logsigmoid(-negative_score).mean(dim=-1)
        positive_score = model(positive_sample)
        positive_score = F.logsigmoid(positive_score).squeeze(dim=-1)
        positive_sample_loss = - (subsampling_weight * positive_score).sum() / subsampling_weight.sum()
        negative_sample_loss = - (subsampling_weight * negative_score).sum() / subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss) / 2
        loss.backward()
        optimizer.step()

        return loss.item()

    @staticmethod
    def test_step(model, test_triples, all_true_triples, nentity, nrelation, isCUDA):

        model.eval()
        test_dataset_head = TestDataset(
            test_triples,
            all_true_triples,
            nentity,
            nrelation,
            'head-batch'
        )
        test_dataset_tail = TestDataset(
            test_triples,
            all_true_triples,
            nentity,
            nrelation,
            'tail-batch'
        )
        test_dataloader_head = DataLoader(
            test_dataset_head,
            batch_size=1,
            num_workers=max(1, 1 // 2),
            collate_fn=TestDataset.collate_fn
        )

        test_dataloader_tail = DataLoader(
            test_dataset_tail,
            batch_size=1,
            num_workers=max(1, 1 // 2),
            collate_fn=TestDataset.collate_fn
        )

        test_dataset_list = [test_dataloader_head, test_dataloader_tail]

        metrics = []
        step = 0
        total_steps = sum([len(dataset) for dataset in test_dataset_list])

        with torch.no_grad():
            for test_dataset in test_dataset_list:
                logs = []
                tempstep = 0

                for positive_sample, negative_sample, filter_bias, mode in test_dataset:

                    if isCUDA == 1:
                        positive_sample = positive_sample.cuda()
                        negative_sample = negative_sample.cuda()
                        filter_bias = filter_bias.cuda()

                    batch_size = positive_sample.size(0)

                    score = model((positive_sample, negative_sample), mode)
                    '''print(score, score.size())
                    print(filter_bias, filter_bias.size())
                    score += filter_bias'''

                    # Explicitly sort all the entities to ensure that there is no test exposure bias
                    argsort = torch.argsort(score, dim=-1, descending=True)

                    if mode == 'head-batch':
                        positive_arg = positive_sample[:, 0]
                    elif mode == 'tail-batch':
                        positive_arg = positive_sample[:, 2]
                    else:
                        raise ValueError('mode %s not supported' % mode)
                    for i in range(batch_size):
                        # Notice that argsort is not ranking
                        ranking = (argsort[:] == positive_arg[i]).nonzero()
                        assert ranking.size(0) == 1

                        # ranking + 1 is the true ranking used in evaluation metrics
                        ranking = 1 + ranking.item()
                        logs.append({
                            'HITS@1': 1.0 if ranking <= 1 else 0.0,
                            'HITS@3': 1.0 if ranking <= 3 else 0.0,
                            'HITS@10': 1.0 if ranking <= 10 else 0.0,
                            'MRR': 1.0 / ranking,
                        })

                    # if step % 1000 == 0:
                    #    print('Evaluating the model... (%d/%d)' % (step, total_steps))

                    step += 1
                    tempstep += 1

                tempmetrics = {}
                for metric in logs[0].keys():
                    tempmetrics[metric] = sum([log[metric] for log in logs]) / len(logs)
                metrics.append(tempmetrics)

        return metrics

from os.path import join
import ujson as json
import numpy as np
from functools import lru_cache
import torch
import logging

logger = logging.getLogger(__name__)


class DwieEvaluator:
    def __init__(self, dataset_dir, ner2id, rel2id):
        self.dataset_dir = dataset_dir
        self.ner2id = ner2id
        self.rel2id = rel2id

        self.id2ner = {i: t for t, i in ner2id.items()}
        self.id2rel = {i: t for t, i in rel2id.items()}

    @lru_cache()
    def get_gold(self, partition):
        with open(join(self.dataset_dir, f'{partition}.json'), 'r') as f:
            truth = json.load(f)
        gold = {}
        tot_evidences = 0
        titleset = set([])
        title2vectexSet = {}
        for x in truth:
            title = x['title']
            titleset.add(title)
            vertexSet = x['vertexSet']
            title2vectexSet[title] = vertexSet
            for label in x['labels']:
                r = label['r']
                h_idx = label['h']
                t_idx = label['t']
                gold[(title, r, h_idx, t_idx)] = set(label['evidence'])
                tot_evidences += len(label['evidence'])
        return gold, tot_evidences, titleset, title2vectexSet

    def to_official(self, all_preds, features):
        all_entity_pair_h, all_entity_pair_t, all_titles = [], [], []
        for f in features:
            all_entity_pair_h.append(f['entity_pairs_h'])
            all_entity_pair_t.append(f['entity_pairs_t'])
            all_titles += [f['title']] * f['entity_pairs_h'].shape[0]
        all_entity_pair_h = torch.cat(all_entity_pair_h, dim=-1).tolist()
        all_entity_pair_t = torch.cat(all_entity_pair_t, dim=-1).tolist()
        ans = []
        for entity_pair_i in range(all_preds.shape[0]):
            if all_entity_pair_h[entity_pair_i] == all_entity_pair_t[entity_pair_i]:  # Skip identify pairs
                continue
            rel_ids = np.nonzero(all_preds[entity_pair_i])[0].tolist()
            for rel_id in rel_ids:
                if rel_id != 0:
                    ans.append({
                        'title': all_titles[entity_pair_i],
                        'h_idx': all_entity_pair_h[entity_pair_i],
                        't_idx': all_entity_pair_t[entity_pair_i],
                        'r': self.id2rel[rel_id]
                    })
        return ans

    def official_evaluate(self, ans, partition='dev'):
        gold, tot_evidences, titleset, title2vectexSet = self.get_gold(partition)
        ans.sort(key=lambda x: (x['title'], x['h_idx'], x['t_idx'], x['r']))
        if ans:
            submission_answer = [ans[0]]
            for i in range(1, len(ans)):
                x = ans[i]
                y = ans[i - 1]
                if (x['title'], x['h_idx'], x['t_idx'], x['r']) != (y['title'], y['h_idx'], y['t_idx'], y['r']):
                    submission_answer.append(ans[i])
        else:
            submission_answer = []

        correct_re = 0
        for x in submission_answer:
            title = x['title']
            h_idx = x['h_idx']
            t_idx = x['t_idx']
            r = x['r']
            if title not in title2vectexSet:
                continue
            if (title, r, h_idx, t_idx) in gold:
                correct_re += 1

        re_p = 1.0 * correct_re / len(submission_answer) * 100 if submission_answer else 0
        re_r = 1.0 * correct_re / len(gold) * 100
        if re_p + re_r == 0:
            re_f1 = 0
        else:
            re_f1 = 2.0 * re_p * re_r / (re_p + re_r)
        return (re_p, re_r, re_f1), None, None, None

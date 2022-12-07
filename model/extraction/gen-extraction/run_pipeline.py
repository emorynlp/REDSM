import logging
import util
from run_base import BaseRunner
import torch
from copy import deepcopy

logger = logging.getLogger(__name__)


class PipelineRunner(BaseRunner):

    def __init__(self, config_name, gpu_id):
        super(PipelineRunner, self).__init__(config_name, gpu_id)

    @classmethod
    def build_index_mapping(cls, features, coref_predictions):
        predidx2goldidx = {}
        for doc_i, (_, predicted_clusters_subtok, _, _) in enumerate(coref_predictions):
            title = features[doc_i]['title']
            gold2idx = {util.tuplize_cluster(entity): entity_i
                        for entity_i, entity in enumerate(features[doc_i]['entities'])}
            num_nongold = 0
            for entity_i, pred_entity in enumerate(predicted_clusters_subtok):
                mapped = gold2idx.get(util.tuplize_cluster(pred_entity), None)
                if mapped is None:
                    mapped = len(gold2idx) + num_nongold
                    num_nongold += 1
                predidx2goldidx[(title, entity_i)] = mapped
        return predidx2goldidx

    @classmethod
    def convert_features(cls, features, coref_predictions):
        converted_features = []
        for feature, (_, predicted_clusters_subtok, _, predicted_types) in zip(features, coref_predictions):
            feature = deepcopy(feature)
            feature['entities'] = predicted_clusters_subtok
            feature['entity_types'] = predicted_types
            feature['entity_pairs_h'] = []
            feature['entity_pairs_t'] = []
            for h in range(len(predicted_clusters_subtok)):
                for t in range(len(predicted_clusters_subtok)):
                    if h != t:
                        feature['entity_pairs_h'].append(h)
                        feature['entity_pairs_t'].append(t)
            feature['entity_pairs_h'] = torch.tensor(feature['entity_pairs_h'], dtype=torch.long)
            feature['entity_pairs_t'] = torch.tensor(feature['entity_pairs_t'], dtype=torch.long)
            feature['rel_labels'] = None
            converted_features.append(feature)
        return converted_features

    @classmethod
    def convert_official_re_predictions(self, re_pred_official, predidx2goldidx):
        for inst in re_pred_official:
            inst['h_idx'] = predidx2goldidx[(inst['title'], inst['h_idx'])]
            inst['t_idx'] = predidx2goldidx[(inst['title'], inst['t_idx'])]

    def initialize_model(self, init_suffix=None):
        pass

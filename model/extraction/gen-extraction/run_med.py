import torch
from run_re import DocReRunner, main
from model_med import MedJointModel
import numpy as np
import math
import logging
from tqdm import tqdm
from run_pipeline import PipelineRunner
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

logger = logging.getLogger(__name__)


class MedRunner(DocReRunner):
    def __init__(self, config_name, gpu_id, **kwargs):
        super(MedRunner, self).__init__(config_name, gpu_id, **kwargs)

        ner2id, rel2id = self.data.get_label_types(self.dataset_name)
        self.id2ner = {i: t for t, i in ner2id.items()}
        self.id2rel = {i: t for t, i in rel2id.items()}

    def initialize_model(self, init_suffix=None):
        model = MedJointModel(self.config, num_entity_types=len(self.id2ner))
        if init_suffix:
            self.load_model_checkpoint(model, init_suffix)
        elif self.config["model_init_config"]:
            self.load_model_checkpoint(
                model,
                self.config["model_init_suffix"],
                self.config["model_init_config"],
            )
        return model

    def evaluate_pseudo_coref(
        self,
        model,
        dataset_name,
        partition,
        docs,
        features,
        tb_writer=None,
        step=0,
        do_eval=True,
    ):
        conf = model.config
        assert len(docs) == len(features)

        logger.info(f"Predicting on {len(features)} features...")
        eval_dataloader = DataLoader(
            features,
            sampler=SequentialSampler(features),
            batch_size=conf["eval_batch_size"],
            collate_fn=self.collator,
        )
        coref_predictions, re_predictions = [], []
        model.to(self.device)
        model.eval()
        for batch_i, batch in tqdm(enumerate(eval_dataloader)):
            batch.pop("mention_cluster_id", None)
            with torch.no_grad():
                doc_returns = model(**batch)
            for doc_return in doc_returns:
                doc_i = len(coref_predictions)
                (
                    span_starts,
                    span_ends,
                    span_mention_scores,
                    span_type_logits,
                ) = doc_return[:4]
                span_starts, span_ends = span_starts.tolist(), span_ends.tolist()
                span_mention_scores = span_mention_scores.tolist()
                span_types = (
                    model.get_span_types(span_type_logits.cpu().numpy())
                    if span_type_logits is not None
                    else None
                )
                predicted_clusters, predicted_types, mention_to_cluster_id = [], [], {}
                predicted_clusters_subtok = []
                predicted_clusters_idx = []
                model.get_predicted_pseudo_clusters(
                    span_starts,
                    span_ends,
                    span_mention_scores,
                    span_types,
                    [
                        (sent_id, tok_id)
                        for sent_id, tok_id in docs[doc_i]["subtok_to_tok"]
                    ],
                    predicted_clusters,
                    predicted_clusters_subtok,
                    predicted_clusters_idx,
                    predicted_types,
                )

                (
                    predicted_clusters,
                    predicted_clusters_subtok,
                    predicted_clusters_idx,
                    predicted_types,
                ) = self.sort_clusters(
                    predicted_clusters,
                    predicted_clusters_subtok,
                    predicted_clusters_idx,
                    predicted_types,
                )
                coref_predictions.append(
                    (
                        predicted_clusters,
                        predicted_clusters_subtok,
                        predicted_clusters_idx,
                        predicted_types,
                    )
                )
                if len(doc_return) > 4:
                    re_pair_logits = doc_return[4].cpu().numpy()
                    re_predictions.append(re_pair_logits)
                if do_eval:
                    docs[doc_i]["clusters"] = [
                        tuple((tuple(m[0]), tuple(m[1])) for m in cluster)
                        for cluster in docs[doc_i]["clusters"]
                    ]
        results = (
            coref_predictions
            if not re_predictions
            else (coref_predictions, re_predictions)
        )
        if not do_eval:
            return results

        metric_fct = (
            self.get_ner_metrics if conf["use_span_type"] else self.get_me_metrics
        )
        eval_score, metrics = metric_fct(
            dataset_name, partition, coref_predictions, docs
        )
        self.log_metrics(metrics, tb_writer, step)
        return eval_score, metrics, results

    def evaluate(
        self,
        model,
        dataset_name,
        partition,
        docs,
        features,
        tb_writer=None,
        step=0,
        do_eval=True,
        re_agg_union=False,
    ):
        coref_return = self.evaluate_pseudo_coref(
            model, dataset_name, partition, docs, features, tb_writer, step, do_eval
        )
        if do_eval:
            ner_eval_score, metrics, (coref_predictions, re_predictions) = coref_return
        else:
            coref_predictions, re_predictions = coref_return

        re_pred_official = []
        for doc_i, ((_, _, cluster_idx, _), re_pair_logits) in enumerate(
            zip(coref_predictions, re_predictions)
        ):
            doc_re_pred = []
            num_spans = math.isqrt(re_pair_logits.shape[0])
            re_pair_logits = re_pair_logits.view()
            re_pair_logits.shape = (num_spans, num_spans, -1)
            for h in range(len(cluster_idx)):
                h_m = cluster_idx[h][0]
                for t in range(len(cluster_idx)):
                    if h == t:
                        continue
                    t_m = cluster_idx[t][0]
                    pair_logits = re_pair_logits[h_m, t_m]
                    doc_re_pred.append(pair_logits)
            if not doc_re_pred:
                continue
            doc_re_pred = np.stack(doc_re_pred)
            doc_re_pred = model.get_re_labels(torch.from_numpy(doc_re_pred)).numpy()
            pair_i = 0
            for h in range(len(cluster_idx)):
                for t in range(len(cluster_idx)):
                    if h == t:
                        continue
                    rel_ids = np.nonzero(doc_re_pred[pair_i])[0].tolist()
                    for rel_id in rel_ids:
                        if rel_id != 0:
                            re_pred_official.append(
                                {
                                    "title": features[doc_i]["title"],
                                    "h_idx": h,
                                    "t_idx": t,
                                    "r": self.id2rel[rel_id],
                                }
                            )
                    pair_i += 1
            assert pair_i == doc_re_pred.shape[0]
        predidx2goldidx = PipelineRunner.build_index_mapping(
            features, coref_predictions
        )
        PipelineRunner.convert_official_re_predictions(
            re_pred_official, predidx2goldidx
        )

        results = coref_predictions, re_pred_official
        if not do_eval:
            return results

        re_eval_score, re_metrics = self.get_re_metrics(
            dataset_name, partition, re_pred_official
        )
        self.log_metrics(re_metrics)
        metrics.update(re_metrics)
        return re_eval_score, metrics, results


if __name__ == "__main__":
    main(MedRunner)

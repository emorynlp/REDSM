import logging
from data import DataProcessor
from run_base import BaseRunner
from metrics_re import DwieEvaluator
from metrics_coref import CorefEvaluator
from metrics_ner import NerEvaluator, MeEvaluator
from collator import FeatureCollator
from os.path import join
from torch.utils.tensorboard import SummaryWriter
import torch
import time
from functools import cached_property, lru_cache
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import sys
import util
import torch.cuda.amp as amp

logger = logging.getLogger(__name__)


class DocReRunner(BaseRunner):
    def __init__(self, config_name, gpu_id=None, **kwargs):
        super(DocReRunner, self).__init__(config_name, gpu_id, **kwargs)
        logger.info(self.config)

    @cached_property
    def data(self):
        return DataProcessor(self.config)

    @cached_property
    def collator(self):
        return FeatureCollator(self.data.tokenizer, device=self.device)

    def initialize_model(self, init_suffix=None):
        pass

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
    ):
        pass

    def train(self, model, use_amp=True, save_threshold=None):
        conf = self.config
        epochs, grad_accum = conf["num_epochs"], conf["gradient_accumulation_steps"]

        model.to(self.device)
        tb_path = join(conf["tb_dir"], self.name + "_" + self.name_suffix)
        tb_writer = SummaryWriter(tb_path, flush_secs=30)

        train_docs, train_features = self.data.get_data(
            self.dataset_name, self.partition["train"]
        )
        dev_docs, dev_features = self.data.get_data(
            self.dataset_name, self.partition["dev"]
        )
        train_dataloader = DataLoader(
            train_features,
            sampler=RandomSampler(train_features),
            batch_size=conf["batch_size"],
            collate_fn=self.collator,
        )
        total_update_steps = len(train_dataloader) * epochs // grad_accum
        eval_after_step = int(total_update_steps * conf["start_eval_after_ratio"])
        optimizer = self.get_optimizer(
            model,
            bert_lr=conf["bert_learning_rate"],
            task_lr=conf["task_learning_rate"],
            bert_wd=conf["bert_wd"],
            task_wd=conf["task_wd"],
            eps=conf["adam_eps"],
        )
        scheduler = self.get_scheduler(
            optimizer, total_update_steps, conf["warmup_ratio"]
        )
        clipping_param = [p for p in model.parameters() if p.requires_grad]
        scaler = amp.GradScaler(enabled=use_amp)

        logger.info("*******************Training*******************")
        logger.info("Num features: %d" % len(train_features))
        logger.info("Num epochs: %d" % epochs)
        logger.info("Batch size: %d" % conf["batch_size"])
        logger.info("Gradient accumulation steps: %d" % grad_accum)
        logger.info("Total update steps: %d" % total_update_steps)

        loss_during_accum = []
        loss_during_report = 0.0
        loss_history = []
        eval_scores = []
        start_time = time.time()
        model.zero_grad()
        for epo in range(epochs):
            for batch_i, batch in enumerate(train_dataloader):
                model.train()
                with amp.autocast(enabled=use_amp):
                    loss, _ = model(**batch)
                    loss /= grad_accum
                scaler.scale(loss).backward()
                loss_during_accum.append(loss.item())

                # Update
                if len(loss_during_accum) % grad_accum == 0:
                    if conf["max_grad_norm"]:
                        scaler.unscale_(optimizer)
                        norm = torch.nn.utils.clip_grad_norm_(
                            clipping_param, conf["max_grad_norm"]
                        )
                    scaler.step(optimizer)
                    scaler.update()
                    model.zero_grad()
                    scheduler.step()

                    effective_loss = sum(loss_during_accum)
                    loss_during_accum = []
                    loss_during_report += effective_loss
                    loss_history.append(effective_loss)
                    if len(loss_history) % conf["report_frequency"] == 0:
                        avg_loss = loss_during_report / conf["report_frequency"]
                        loss_during_report = 0.0
                        end_time = time.time()
                        logger.info(
                            "Step %d: avg loss %.4f; steps/sec %.2f"
                            % (
                                len(loss_history),
                                avg_loss,
                                conf["report_frequency"] / (end_time - start_time),
                            )
                        )
                        start_time = end_time
                        tb_writer.add_scalar(
                            "Training_Loss", avg_loss, len(loss_history)
                        )
                        tb_writer.add_scalar(
                            "Learning_Rate_Bert",
                            scheduler.get_last_lr()[0],
                            len(loss_history),
                        )
                        tb_writer.add_scalar(
                            "Learning_Rate_Task",
                            scheduler.get_last_lr()[-1],
                            len(loss_history),
                        )

                    if (
                        len(loss_history) > eval_after_step
                        and len(loss_history) % conf["eval_frequency"] == 0
                    ):
                        eval_score, _, _ = self.evaluate(
                            model,
                            self.dataset_name,
                            self.partition["dev"],
                            dev_docs,
                            dev_features,
                            tb_writer,
                            len(loss_history),
                        )
                        if not eval_scores or eval_score > max(eval_scores):
                            if save_threshold is None or eval_score > save_threshold:
                                self.save_model_checkpoint(model, len(loss_history))
                        eval_scores.append(eval_score)
                        logger.info(f"Best eval score: {max(eval_scores):.2f}")
                        start_time = time.time()

        logger.info("**********Finished training**********")
        logger.info("Actual update steps: %d" % len(loss_history))
        model.zero_grad()

        eval_score, _, _ = self.evaluate(
            model,
            self.dataset_name,
            self.partition["dev"],
            dev_docs,
            dev_features,
            tb_writer,
            len(loss_history),
        )
        if not eval_scores or eval_score > max(eval_scores):
            if save_threshold is None or eval_score > save_threshold:
                self.save_model_checkpoint(model, len(loss_history))
        eval_scores.append(eval_score)
        logger.info(f"All eval scores: {eval_scores}")

        tb_writer.close()
        return loss_history, eval_scores

    @lru_cache()
    def get_re_evaluator(self, dataset_name):
        ner2id, rel2id = self.data.get_label_types(dataset_name)
        return DwieEvaluator(self.data.get_data_dir(dataset_name), ner2id, rel2id)

    def get_re_metrics(self, dataset_name, partition, pred_official):
        re_prf, _, re_ign_prf, _ = self.get_re_evaluator(
            dataset_name
        ).official_evaluate(pred_official, partition)
        metrics = {f"{partition}_re_f1": re_prf[-1], f"{partition}_re_prf": re_prf}
        if re_ign_prf is not None:
            metrics.update(
                {
                    f"{partition}_re_f1_ign": re_ign_prf[-1],
                    f"{partition}_re_ign_prf": re_ign_prf,
                }
            )
        return re_prf[-1], metrics

    def evaluate_coref(
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

        coref_evaluator = CorefEvaluator()
        coref_predictions, re_predictions = [], []
        model.to(self.device)
        model.eval()
        for batch_i, batch in enumerate(eval_dataloader):
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
                    antecedent_idx,
                    antecedent_scores,
                ) = doc_return[:6]
                span_starts, span_ends = span_starts.tolist(), span_ends.tolist()
                span_mention_scores = span_mention_scores.tolist()
                antecedent_idx, antecedent_scores = (
                    antecedent_idx.tolist(),
                    antecedent_scores.tolist(),
                )
                span_types = (
                    model.get_span_types(span_type_logits.cpu().numpy())
                    if span_type_logits is not None
                    else None
                )
                predicted_clusters, predicted_types, mention_to_cluster_id = [], [], {}
                predicted_clusters_subtok = []
                predicted_clusters_idx = []
                model.get_predicted_clusters(
                    span_starts,
                    span_ends,
                    span_mention_scores,
                    span_types,
                    antecedent_idx,
                    antecedent_scores,
                    [
                        (sent_id, tok_id)
                        for sent_id, tok_id in docs[doc_i]["subtok_to_tok"]
                    ],
                    predicted_clusters,
                    predicted_clusters_subtok,
                    predicted_clusters_idx,
                    predicted_types,
                    mention_to_cluster_id,
                    allow_singletons=conf["allow_singletons"],
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
                if len(doc_return) > 6:
                    re_pair_logits = doc_return[6].cpu().numpy()
                    re_predictions.append(re_pair_logits)
                if do_eval:
                    docs[doc_i]["clusters"] = [
                        tuple((tuple(m[0]), tuple(m[1])) for m in cluster)
                        for cluster in docs[doc_i]["clusters"]
                    ]
                    self.update_coref_evaluator(
                        predicted_clusters,
                        mention_to_cluster_id,
                        docs[doc_i]["clusters"],
                        coref_evaluator,
                    )

        results = (
            coref_predictions
            if not re_predictions
            else (coref_predictions, re_predictions)
        )
        if not do_eval:
            return results

        eval_score, metrics = self.get_coref_metrics(partition, coref_evaluator)
        metric_fct = (
            self.get_ner_metrics if conf["use_span_type"] else self.get_me_metrics
        )
        ner_eval_score, ner_metrics = metric_fct(
            dataset_name, partition, coref_predictions, docs
        )
        metrics.update(ner_metrics)
        self.log_metrics(metrics, tb_writer, step)
        return eval_score, metrics, results

    @classmethod
    def sort_clusters(cls, clusters, clusters_subtok, clusters_idx, cluster_types):
        (
            sorted_clusters,
            sorted_clusters_subtok,
            sorted_clusters_idx,
            sorted_cluster_types,
        ) = ([], [], [], [])
        for cluster, cluster_subtok, cluster_idx, types in zip(
            clusters, clusters_subtok, clusters_idx, cluster_types
        ):
            indices = util.argsort(cluster_subtok)
            sorted_clusters.append(tuple(cluster[idx] for idx in indices))
            sorted_clusters_subtok.append(tuple(cluster_subtok[idx] for idx in indices))
            sorted_clusters_idx.append(tuple(cluster_idx[idx] for idx in indices))
            sorted_cluster_types.append(tuple(types[idx] for idx in indices))
        return (
            tuple(sorted_clusters),
            tuple(sorted_clusters_subtok),
            tuple(sorted_clusters_idx),
            tuple(sorted_cluster_types),
        )

    @classmethod
    def update_coref_evaluator(
        cls, predicted_clusters, mention_to_cluster_id, gold_clusters, evaluator
    ):
        mention_to_predicted = {
            m: predicted_clusters[cluster_idx]
            for m, cluster_idx in mention_to_cluster_id.items()
        }
        mention_to_gold = {m: cluster for cluster in gold_clusters for m in cluster}
        evaluator.update(
            predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold
        )

    @classmethod
    def get_coref_metrics(cls, partition, coref_evaluator):
        p, r, f = coref_evaluator.get_prf()
        p, r, f = p * 100, r * 100, f * 100
        metrics = {
            f"{partition}_coref_precision": p,
            f"{partition}_coref_recall": r,
            f"{partition}_coref_f1": f,
        }
        metrics.update(
            {
                name: coref_evaluator.evaluators[i].get_prf()
                for i, name in enumerate(["muc", "b_cubed", "ceafe"])
            }
        )
        return f, metrics

    def get_ner_metrics(self, dataset_name, partition, coref_predictions, docs):
        predicted_spans, predicted_types, gold_spans, gold_types = [], [], [], []
        for doc_i, (doc_pred_spans, _, _, doc_pred_types) in enumerate(
            coref_predictions
        ):
            doc_gold_spans, doc_gold_types = (
                docs[doc_i]["clusters"],
                docs[doc_i]["cluster_types"],
            )
            predicted_spans += [(doc_i, val) for val in util.flatten(doc_pred_spans)]
            predicted_types += util.flatten(doc_pred_types)
            gold_spans += [(doc_i, val) for val in util.flatten(doc_gold_spans)]
            gold_types += util.flatten(doc_gold_types)
        ner2id, _ = self.data.get_label_types(dataset_name)
        ner_evaluator = NerEvaluator(ner2id)
        total_prf, type2prf = ner_evaluator.evaluate(
            predicted_spans, predicted_types, gold_spans, gold_types
        )
        ner_metrics = {
            f"{partition}_ner_total_precision": total_prf[0],
            f"{partition}_ner_total_recall": total_prf[1],
            f"{partition}_ner_total_f1": total_prf[2],
        }
        for ner_type, prf in type2prf.items():
            ner_metrics[f"{partition}_ner_{ner_type}_prf"] = prf
        return total_prf[2], ner_metrics

    def get_me_metrics(self, dataset_name, partition, coref_predictions, docs):
        predicted_spans, gold_spans = [], []
        for doc_i, (doc_pred_spans, _, _, _) in enumerate(coref_predictions):
            doc_gold_spans = docs[doc_i]["clusters"]
            predicted_spans += [(doc_i, val) for val in util.flatten(doc_pred_spans)]
            gold_spans += [(doc_i, val) for val in util.flatten(doc_gold_spans)]

        ner_evaluator = MeEvaluator()
        total_prf = ner_evaluator.evaluate(predicted_spans, gold_spans)
        me_metrics = {
            f"{partition}_me_precision": total_prf[0],
            f"{partition}_me_recall": total_prf[1],
            f"{partition}_me_f1": total_prf[2],
        }
        return total_prf[2], me_metrics


def main(runner_class, partition_config="test", do_eval=True, save_json_target=None):
    if len(sys.argv) in (2, 3):
        # Train
        config_name, gpu_id = (
            sys.argv[1],
            int(sys.argv[2]) if len(sys.argv) > 2 else None,
        )
        runner = runner_class(config_name, gpu_id)
        model = runner.initialize_model()
        runner.train(model, use_amp=runner.config["use_amp"])
    elif len(sys.argv) == 4:
        # Predict test set
        config_name, suffix, gpu_id = sys.argv[1], sys.argv[2], int(sys.argv[3])
        runner = runner_class(config_name, gpu_id)
        model = runner.initialize_model(init_suffix=suffix)
        do_eval = runner.config["do_eval"]

        dataset_name, partition = (
            runner.dataset_name,
            runner.partition[partition_config],
        )
        test_docs, test_features = runner.data.get_data(dataset_name, partition)
        results = runner.evaluate(
            model, dataset_name, partition, test_docs, test_features, do_eval=do_eval
        )
        if do_eval:
            eval_score, metrics, results = results

        runner.save_results(dataset_name, partition, suffix, "bin", results)
        if save_json_target:
            runner.save_results(
                dataset_name, partition, suffix, "json", save_json_target(results)
            )
    else:
        raise RuntimeError(f"Usage: python run.py config_name [suffix] gpu")

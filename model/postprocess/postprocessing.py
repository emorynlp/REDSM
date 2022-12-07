import json, pickle, os, csv, sys
from pprint import pprint
from typing import Counter
import logging
from nervaluate import Evaluator
from resultviewer import ResultsViewer

logger = logging.getLogger(__name__)


class PostProcessor(ResultsViewer):
    def __init__(self, config_name, config_path, save_log=True) -> None:
        super().__init__(config_name, config_path, save_log)

    def counts_spurious(self):
        all_posts, gold_ents, gold_rels, pred_ents, pred_rels = self.get_results(
            self.gold_file, self.pred_file
        )
        all_spurious = []
        for i, doc_ents in enumerate(pred_ents):
            spurious = self.print_spurious(gold_ents[i], doc_ents)
            doc_spurious = []
            for ent in spurious:
                # doc_spurious += ent[0]
                doc_spurious.append(" ".join(ent[0]))
            all_spurious += doc_spurious
            # all_spurious += [ent[1] for ent in spurious]
        logger.info(Counter(all_spurious))

    def filter_entity(self, pred_ents, pred_rels):
        filtered_ents = []
        for i, doc_ents in enumerate(pred_ents):
            ent_in_rel = set(
                [post_rel["h_idx"] for post_rel in pred_rels[i]]
                + [post_rel["t_idx"] for post_rel in pred_rels[i]]
            )
            filtered = [
                ent
                for ent in doc_ents
                if (doc_ents.index(ent) in ent_in_rel)
                or (ent[-1]) not in self.config["ent_types"]
            ]

            filtered_ents.append(filtered)
        return filtered_ents

    def filter_rel(self, pred_ents, pred_rels):
        filtered_rels = []
        attr_parents = self.config["condition"] + self.config["treatment"]
        for i, doc_rels in enumerate(pred_rels):
            filtered = []
            for rel in doc_rels:
                try:
                    parent_type = pred_ents[i][rel["t_idx"]][-1]
                    child_type = pred_ents[i][rel["h_idx"]][-1]
                    if rel["r"] == "Treatment" and (
                        parent_type in self.config["condition"]
                        and child_type in self.config["treatment"]
                    ):
                        filtered.append(rel)
                    if rel["r"] == "Attribute" and (
                        parent_type in attr_parents
                        and child_type in self.config["attributes"]
                    ):
                        filtered.append(rel)
                except:
                    print(rel["title"])
                    print(len(pred_ents[i]))
                    print(rel["t_idx"], rel["h_idx"])
            filtered_rels.append(filtered)

        return filtered_rels

    def eval_ner(self, gold_ents, pred_ents, pred_rels):
        filtered_pred_ents = self.filter_entity(pred_ents, pred_rels)
        gold_ent_ls, pred_ent_ls = [], []
        for i, doc_ents in enumerate(gold_ents):
            gold_ent_ls.append(self.format_ner(doc_ents))
            pred_ent_ls.append(self.format_ner(filtered_pred_ents[i]))
            # pred_ent_ls.append(self.format_ner(pred_ents[i]))

        evaluator = Evaluator(gold_ent_ls, pred_ent_ls, tags=list(self.id2ner.values()))
        results, results_per_tag = evaluator.evaluate()
        logger.info(results["strict"])
        logger.info({tag: value["strict"] for tag, value in results_per_tag.items()})

        return results, results_per_tag

    def main(self):
        all_posts, gold_ents, gold_rels, pred_ents, pred_rels = self.get_results(
            self.gold_file, self.pred_file
        )
        ner_results, ner_results_per_tag = self.eval_ner(
            gold_ents, pred_ents, pred_rels
        )
        filtered_pred_rels = self.filter_rel(pred_ents, pred_rels)
        rel_p, rel_r, rel_f1 = self.evaluate_relation(gold_rels, filtered_pred_rels)


if __name__ == "__main__":

    processor = PostProcessor(
        "filter_basic_rel",
        config_path="/local/scratch/stu9/RLS/relation/postprocess/postprocess.conf",
    )
    # processor.eval_ner()
    processor.main()


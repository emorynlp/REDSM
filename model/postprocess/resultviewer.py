import jsonlines, json, os
import pickle
import logging
from util import get_config
from nervaluate import Evaluator
from pprint import pprint
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm

logger = logging.getLogger()


class ResultsViewer:
    def __init__(self, config_name, config_path, save_log=True) -> None:
        self.config = get_config(config_name, config_path)
        self.gold_file = os.path.join(
            self.config["dataset_dir"], self.config["dataset_name"], "test.json"
        )
        self.pred_file = os.path.join(
            self.config["model_result_dir"], self.config["pred_file"]
        )
        self.id_file = os.path.join(
            self.config["dataset_dir"], self.config["dataset_name"], "meta.json"
        )
        self.log_dir = os.path.join(
            self.config["save_dir"],
            self.config["dataset_name"],
            self.config["pretrained"],
        )
        self.compare_file = os.path.join(self.log_dir, "compare.txt")
        self.results_file = os.path.join(
            self.log_dir, f"{self.config['pred_file'].split('.')[0]}.json"
        )
        self.id2ner, self.id2rel = self.id2label(self.id_file)
        if save_log:
            log_path = os.path.join(
                self.log_dir, f"{self.config['pred_file'].split('.')[0]}.txt"
            )
            logger.addHandler(logging.FileHandler(log_path, "a"))
            logger.info(f"Log file path: {log_path}")
            logger.info(self.config)

    def file_ext(self, file):
        return file.split(".")[-1]

    # load file based on the extension
    def load_file(self, file):
        file_ext = self.file_ext(file)
        if file_ext == "bin":
            with open(
                file,
                "rb",
            ) as f:
                results = pickle.load(f)
        elif file_ext == "jsonlines":
            with jsonlines.open(file, "r") as f:
                results = [doc for doc in f]
        elif file_ext == "json":
            with open(file, "r") as f:
                results = json.load(f)
        else:
            logger.error("Check input file type.")
        return results

    def id2label(self, id_file):
        dicts = self.load_file(id_file)
        # print(dicts)
        id2ner = {n: i for i, n in dicts["ner2id"].items()}
        id2rel = {r: i for i, r in dicts["rel2id"].items()}
        return id2ner, id2rel

    def get_predicted(self, file):
        results = self.load_file(file)
        pred_entities = results[0]
        pred_rels = results[1]
        return pred_entities, pred_rels

    def get_gold_entities(self, vertexset):
        entities = []
        for vertex in vertexset:
            entities.append(
                (
                    vertex[0]["name"].split(),
                    vertex[0]["sent_id"],
                    tuple(vertex[0]["pos"]),
                    vertex[0]["type"],
                )
            )
        return entities

    def get_pred_entities(self, clusters, cluster_types, sents, id2ner):
        entities = []
        for i, cluster in enumerate(clusters):
            ent = cluster[0]
            sent_id = ent[0][0]
            ent_start = ent[0][1]
            ent_end = ent[-1][1] + 1
            entity = [
                sents[sent_id][ent_start:ent_end],
                sent_id,
                (ent_start, ent_end),
                id2ner[cluster_types[i][0]],
            ]
            # pprint(entity)
            entities.append(entity)
        return entities

    # get the gold ents, rels and pred ents, rels for per doc (also works for multiple docs)
    def get_results(self, gold_file, pred_file):
        logger.info("Loading file...")
        gold_data = self.load_file(gold_file)
        pred_ents, pred_rels = self.get_predicted(pred_file)
        (
            all_posts,
            formatted_gold_ents,
            formatted_gold_rels,
            formatted_pred_ents,
            formatted_pred_rels,
        ) = ([], [], [], [], [])
        logger.info("Formatting entities and relations...")
        for i, doc in tqdm(enumerate(gold_data)):
            doc_id, sents = doc["title"], doc["sents"]
            if self.config["print_gold"]:
                gold_doc_rels = doc["labels"]
                gold_doc_ents = self.get_gold_entities(doc["vertexSet"])
                formatted_gold_ents.append(gold_doc_ents)
                formatted_gold_rels.append(gold_doc_rels)
            pred_doc_ents = self.get_pred_entities(
                pred_ents[i][0], pred_ents[i][3], sents, self.id2ner
            )
            pred_doc_rels = [rel for rel in pred_rels if rel["title"] == doc_id]
            all_posts.append((doc_id, sents))
            formatted_pred_ents.append(pred_doc_ents)
            formatted_pred_rels.append(pred_doc_rels)
        return (
            all_posts,
            formatted_gold_ents,
            formatted_gold_rels,
            formatted_pred_ents,
            formatted_pred_rels,
        )

    def format_ner(self, doc_ent_ls, ent_name=False):
        sent_ids = sorted(set(map(lambda x: x[1], doc_ent_ls)))
        if ent_name:
            doc_ents = {
                x: [
                    (ent[0], ent[-1], x, (ent[2][0], ent[2][1]))
                    for ent in doc_ent_ls
                    if ent[1] == x
                ]
                for x in sent_ids
            }
        else:
            doc_ents = [
                {"label": ent[-1], "start": ent[2][0], "end": ent[2][1]}
                for x in sent_ids
                for ent in doc_ent_ls
                if ent[1] == x
            ]
        return doc_ents

    def find_overlap(self, trues, preds):
        overlap_dict = {}
        for pred in preds:
            pred_set = set((pred[0], pred[1] - 1))
            overlap_dict[pred] = False
            for gold in trues:
                true_set = set((gold[0], gold[1] - 1))
                if true_set.intersection(pred_set):
                    overlap_dict[pred] = True
                    break
        return [key for key, value in overlap_dict.items() if value is False]

    def print_spurious(self, doc_gold, doc_pred):
        formated_gold = self.format_ner(doc_gold, ent_name=True)
        formated_pred = self.format_ner(doc_pred, ent_name=True)
        # spurious_sent_id = set(formated_pred.keys()) - set(formated_gold.keys())
        spurious = []
        for sent_id, ents in formated_pred.items():
            pred_ent_ranges = [ent[3] for ent in ents]
            gold_ent_ranges = (
                [ent[3] for ent in formated_gold[sent_id]]
                if sent_id in formated_gold
                else []
            )
            if gold_ent_ranges:
                spurious_ents = self.find_overlap(gold_ent_ranges, pred_ent_ranges)
                spurious += [ent for ent in ents if ent[3] in spurious_ents]
            else:
                spurious += ents

        return spurious

    def print_diff(self, gold_ent_ls, pred_ent_ls):
        count = 0
        for doc_gold, doc_pred in zip(gold_ent_ls, pred_ent_ls):
            difft = [ent for ent in doc_pred if ent not in doc_gold]
            count += len(difft)
            print(difft)

        print(count)

    # list of sent lists of entity dicts
    def evaluate_ner(self, gold_ents, pred_ents):
        gold_ent_ls, pred_ent_ls = [], []
        for i, doc_ents in enumerate(gold_ents):
            # gold_ent_ls += self.format_ner(doc_ents)
            # pred_ent_ls += self.format_ner(pred_ents[i])
            gold_ent_ls.append(self.format_ner(doc_ents))
            pred_ent_ls.append(self.format_ner(pred_ents[i]))

        evaluator = Evaluator(gold_ent_ls, pred_ent_ls, tags=list(self.id2ner.values()))
        results, results_per_tag = evaluator.evaluate()

        # pprint(results["exact"])
        # pprint(results_per_tag)
        all_tag = results["strict"]
        per_tag = {tag: result["strict"] for tag, result in results_per_tag.items()}

        return all_tag, per_tag

    def rel_by_tag(self, doc_rel, rel_type=""):
        return [rel for rel in doc_rel if rel["r"] == rel_type]

    def rel_score(self, correct, actual, possible):
        precision = correct / actual
        recall = correct / possible
        f1 = 2 * (precision * recall) / (precision + recall)

        return precision, recall, f1

    def evaluate_relation(self, gold_rels, pred_rels):
        keymap = {"h_idx": "h", "t_idx": "t", "r": "r"}
        correct, actual, possible = 0, 0, 0
        attr_correct, attr_actual, attr_possible = 0, 0, 0
        treat_correct, treat_actural, treat_possible = 0, 0, 0
        for i, doc_rel in enumerate(pred_rels):
            formatted_rels = []
            for j, rel in enumerate(doc_rel):
                formatted_pred_rel = {
                    keymap[key]: value for (key, value) in rel.items() if key in keymap
                }
                formatted_pred_rel["evidence"] = []
                formatted_rels.append(formatted_pred_rel)
            correct += len([rel for rel in formatted_rels if rel in gold_rels[i]])
            actual += len(gold_rels[i])
            possible += len(formatted_rels)
            attr_correct += len(
                [
                    rel
                    for rel in self.rel_by_tag(formatted_rels, "Attribute")
                    if rel in gold_rels[i]
                ]
            )
            attr_actual += len(self.rel_by_tag(gold_rels[i], "Attribute"))
            attr_possible += len(self.rel_by_tag(formatted_rels, "Attribute"))
            treat_correct += len(
                [
                    rel
                    for rel in self.rel_by_tag(formatted_rels, "Treatment")
                    if rel in gold_rels[i]
                ]
            )
            treat_actural += len(self.rel_by_tag(gold_rels[i], "Treatment"))
            treat_possible += len(self.rel_by_tag(formatted_rels, "Treatment"))

        precision, recall, f1 = self.rel_score(correct, actual, possible)
        attr_p, attr_r, attr_f1 = self.rel_score(
            attr_correct, attr_actual, attr_possible
        )
        treat_p, treat_r, treat_f1 = self.rel_score(
            treat_correct, treat_actural, treat_possible
        )

        logger.info(
            f"Relation F1 score: {f1}\nRelation Precision: {precision}\nRelation Recall: {recall}\n"
        )
        logger.info(
            f"Attr F1 score: {attr_f1}\nAttr Precision: {attr_p}\nAttr Recall: {attr_r}\n"
        )
        logger.info(
            f"Treat F1 score: {treat_f1}\nTreat Precision: {treat_p}\nTreat Recall: {treat_r}\n"
        )

        return precision, recall, f1

    def write_results(self, ner_scores, ner_by_tag, rel_scores):
        with open(self.results_file, "w") as f:
            json.dump([ner_scores, ner_by_tag, rel_scores], f)

    def write_comparison(
        self, docs, gold_ents, gold_relations, pred_ents, pred_relations
    ):
        with open(self.compare_file, "w") as f:
            for i, (doc_id, sents) in enumerate(docs):
                f.write(f"{doc_id}\n")
                for j, sent in enumerate(sents):
                    s = " ".join(sent)
                    f.write(f"[{j}]\t{s}\n")
                if self.config["print_spurious"]:
                    f.write("\nSpurious entities:\n")
                    for ent in self.print_spurious(gold_ents[i], pred_ents[i]):
                        f.write(f"{ent}\t")
                if self.config["print_gold"]:
                    f.write(f"\n\nGold entities:\n")
                    for ent in gold_ents[i]:
                        f.write(f"{ent}\t")
                    f.write(f"\n\nGold relations:\n")
                    for rel in gold_relations[i]:
                        f.write(
                            f"Type: {rel['r']}\thead: {gold_ents[i][rel['h']][0]}\ttail: {gold_ents[i][rel['t']][0]}\n"
                        )
                f.write(f"\n\nPred entities:\n")
                for ent in pred_ents[i]:
                    f.write(f"{ent}\t")
                f.write(f"\n\nPred relations:\n")
                for rel in pred_relations[i]:
                    try:
                        f.write(
                            f"Type: {rel['r']}\thead: {pred_ents[i][rel['h_idx']][0]}\ttail: {pred_ents[i][rel['t_idx']][0]}\n"
                        )
                    except:
                        print(doc_id)
                f.write("\n==================================\n")

    def write_csv(self, docs, pred_ents):
        ent_ls = []
        for i, (doc_id, _) in tqdm(enumerate(docs)):
            if len(pred_ents[i]) > 0:
                for ent in pred_ents[i]:
                    if ent[-1] in self.config["ent_types"]:
                        ent_text = " ".join(ent[0])
                        ent_start = f"{ent[1]}-{ent[2][0]}"
                        ent_end = f"{ent[1]}-{ent[2][1]}"
                        ent_ls.append((doc_id, ent_text, ent_start, ent_end, ent[-1]))

        df = pd.DataFrame(
            ent_ls, columns=["post_id", "condition", "ent_start", "ent_end", "label"]
        )
        df.to_csv(self.config["csv_file"])

    def make_confusion_matrix(self, all_posts, gold_entities, pred_entities):
        gold, pred = [], []
        for i, post in enumerate(all_posts):
            doc_gold = gold_entities[i]
            doc_pred = pred_entities[i]
            for j, sent in enumerate(post[1]):
                gold_sent = ["O"] * len(sent)
                pred_sent = ["O"] * len(sent)
                sent_gold_ents = [ent for ent in doc_gold if ent[1] == j]
                sent_pred_ents = [ent for ent in doc_pred if ent[1] == j]
                for ent in sent_gold_ents:
                    for k in range(ent[2][0], ent[2][1]):
                        gold_sent[k] = ent[-1] if "Condition" in ent[-1] else "O"
                for ent in sent_pred_ents:
                    for l in range(ent[2][0], ent[2][1]):
                        pred_sent[l] = ent[-1] if "Condition" in ent[-1] else "O"
                # print(gold_sent)
                gold += gold_sent
                pred += pred_sent
        labels = [
            "Patient Condition",
            "Caregiver Condition",
            "Unspecified Condition",
        ]
        cm = confusion_matrix(gold, pred, labels=labels)

        print(cm)
        cm_df = pd.DataFrame(cm, index=labels, columns=labels)
        plt.figure(figsize=(10, 10))
        sns.heatmap(cm_df, annot=True, cmap="Blues", fmt="g")
        plt.title("Confusion Matrix")
        plt.ylabel("Actal Values")
        plt.xlabel("Predicted Values")
        plt.savefig("cm_condition.png")
        plt.show()

    def main(self):
        (
            all_posts,
            gold_entities,
            gold_relations,
            pred_entities,
            pred_relations,
        ) = self.get_results(self.gold_file, self.pred_file)
        # self.print_diff(gold_entities, pred_entities)

        # ner_scores, ner_by_tag = self.evaluate_ner(gold_entities, pred_entities)
        # rel_p, rel_r, rel_f = self.evaluate_relation(gold_relations, pred_relations)
        # rel_scores = {"rel_p": rel_p, "rel_r": rel_r, "rel_f": rel_f}

        # self.write_results(ner_scores, ner_by_tag, rel_scores)
        # self.write_comparison(
        #     all_posts, gold_entities, gold_relations, pred_entities, pred_relations
        # )
        # self.make_confusion_matrix(all_posts, gold_entities, pred_entities)
        self.write_csv(all_posts, pred_entities)


if __name__ == "__main__":
    viewer = ResultsViewer(
        "adhd",
        config_path="/local/scratch/stu9/RLS/relation/postprocess/postprocess.conf",
    )
    viewer.main()

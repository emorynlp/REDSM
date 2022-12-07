import logging
import util
import io_util
from functools import cached_property, lru_cache
from os.path import join, exists
import os
from data_util import (
    get_all_docs,
    convert_docs_to_features,
)

logger = logging.getLogger(__name__)


class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.task_desc = None

    @cached_property
    def tokenizer(self):
        return util.get_transformer_tokenizer(self.config)

    @lru_cache()
    def get_label_types(self, dataset_name):
        if dataset_name in ("docred", "dwie"):
            meta_dir = join(self.get_data_dir(dataset_name), "meta")
            ner2id = io_util.read_json(join(meta_dir, "ner2id.json"))
            rel2id = io_util.read_json(join(meta_dir, "rel2id.json"))
            return ner2id, rel2id
        elif dataset_name.startswith("med_"):
            meta_path = join(self.get_data_dir(dataset_name), "meta.json")
            meta = io_util.read_json(meta_path)
            return meta["ner2id"], meta["rel2id"]
        else:
            raise ValueError(f"Unknown dataset {dataset_name}")

    @classmethod
    def is_training(cls, partition):
        return "train" in partition

    def get_data(self, dataset_name, partition):
        doc_path = self.get_data_doc_path(dataset_name, partition)
        feat_path = self.get_data_feature_path(dataset_name, partition)
        conf, is_training = self.config, self.is_training(partition)

        if exists(doc_path):
            docs = io_util.read_jsonlines(doc_path)
        else:
            raw_path = self.get_data_raw_path(dataset_name, partition)
            if dataset_name in ("docred", "dwie") or dataset_name.startswith("med_"):
                ner2id, rel2id = self.get_label_types(dataset_name)
                docs = get_all_docs(
                    dataset_name,
                    raw_path,
                    self.tokenizer,
                    ner2id=ner2id,
                    rel2id=rel2id,
                    is_training=is_training,
                    with_root=conf["with_root"],
                )
            else:
                raise ValueError(f"Unknown dataset {dataset_name}")
            io_util.write_jsonlines(doc_path, docs)

        if exists(feat_path):
            features = io_util.read_pickle(feat_path)
        else:
            features = convert_docs_to_features(
                dataset_name,
                docs,
                self.tokenizer,
                max_seq_len=conf["max_seq_len"],
                overlapping=conf["overlapping"],
                is_training=is_training,
                max_training_seg=conf["max_training_seg"],
                show_example=True,
            )
            io_util.write_pickle(feat_path, features)

        return docs, features

    def get_data_dir(self, dataset_name):
        return join(self.config["dataset_dir"], dataset_name)

    def get_data_raw_path(self, dataset_name, partition):
        if dataset_name in ("docred", "dwie") or dataset_name.startswith("med_"):
            file_path = join(self.get_data_dir(dataset_name), f"{partition}.json")
            return file_path
        else:
            raise ValueError(f"Unknown dataset {dataset_name}")

    def get_data_doc_path(self, dataset_name, partition):
        save_dir = join(self.config["data_dir"], "processed")
        os.makedirs(save_dir, exist_ok=True)
        tokenizer = self.config["model_type"]
        save_path = join(
            save_dir, f"doc_{dataset_name}_{partition}_{tokenizer}.jsonlines"
        )
        return save_path

    def get_data_feature_path(self, dataset_name, partition):
        save_dir = join(self.config["data_dir"], "processed")
        os.makedirs(save_dir, exist_ok=True)
        t = self.config["model_type"]
        msl = self.config["max_seq_len"]
        is_training = self.is_training(partition)

        ol = self.config["overlapping"]
        mts = f'_mts{self.config["max_training_seg"]}' if is_training else ""
        save_path = join(
            save_dir, f"feat_{dataset_name}_{partition}_{t}_max{msl}_ol{ol}{mts}.bin"
        )
        return save_path

    def get_results_path(self, dataset_name, partition, suffix, ext="json"):
        save_dir = join(self.config["log_dir"], "results")
        os.makedirs(save_dir, exist_ok=True)
        return join(save_dir, f"results_{dataset_name}_{partition}_{suffix}.{ext}")

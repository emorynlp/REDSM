import os, json
import numpy as np
import pandas as pd
from elit.component.tokenizer import EnglishTokenizer
import logging
from tqdm import tqdm
import string

# print(f"max post len: {max(toks_nums)}")  # 467
# print(f"avg post len: {sum(toks_nums)/len(toks_nums)}")  # 53.464
# print(f"std of post len: {np.std(toks_nums)}")  # 45.455

logger = logging.getLogger(__name__)


class Processor:
    def __init__(self, file, dataset, root_dir):
        self.file = file
        self.dataset_path = os.path.join(root_dir, dataset)
        self.tokenizer = EnglishTokenizer()

    def load_data(self, file):
        with open(file, "r", encoding="utf-8") as f:
            file = f.read()
            data = file.strip().split("\n")
            id_text = []
            for line in tqdm(data):
                seg = line.split("\t", 1)
                idx = seg[0]
                post = "".join(filter(lambda x: x in string.printable, seg[1]))
                if post != seg[1]:
                    print("================")
                    print(idx, "\n", post, "\n", seg[1])
                id_text.append((idx, post))

            # toks_nums = [text.split() for text in raw_texts]
            # print(idx, text)
        return id_text

    def segment(self, post, idx):
        try:
            toked_sent = self.tokenizer.tokenize(post)
            sentences = self.tokenizer.segment(toked_sent[0], toked_sent[1])
            if len(sentences) > 0:
                sent_ls = [s["tok"] for s in sentences if len(s) > 0]
                return sent_ls
            else:
                logger.info(f"{idx}\n{post}")
                return []
        except:
            logger.exception("may be recursion error")
            logger.info(f"{idx}\n{post}")
            return []

    def write_file(self, posts):
        if not os.path.exists(self.dataset_path):
            os.makedirs(self.dataset_path)

        with open(os.path.join(self.dataset_path, "test.json"), "w") as f:
            json.dump(posts, f)

    def generate_single(self, post, idx):
        post_dic = {}
        post_dic["vertexSet"], post_dic["labels"] = [], []
        post_dic["title"] = idx
        post_sent = self.segment(post, idx)
        post_dic["sents"] = post_sent

        return post_dic

    def check(self):
        with open(os.path.join(self.dataset_path, "test.json"), "r") as f:
            data = json.load(f)
            for i, doc in tqdm(enumerate(data)):
                sents = doc["sents"]
                sent_len = [len(sent) for sent in sents]
                if 0 in sent_len:
                    print(doc["title"])
                    print(sents)

    def main(self):
        posts = []
        raw_texts = self.load_data(self.file)
        for (post_id, text) in tqdm(raw_texts):
            single_post = self.generate_single(text, post_id)
            if single_post["sents"]:
                posts.append(single_post)

        self.write_file(posts)


if __name__ == "__main__":
    file = "/local/scratch/stu9/RLS/relation/RLS_ADHD_posts_Emory_full.tsv"
    root_dir = "/local/scratch/stu9/RLS/relation/extraction/dataset"
    dataset_name = "med_adhd_all"

    processor = Processor(file, dataset_name, root_dir)
    processor.main()
    # processor.check()

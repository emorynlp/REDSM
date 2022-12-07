from collections import defaultdict
import util
import numpy as np
import io_util
from seq_encoding import tokenize_sentences, encode_long_sequence
import torch
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


def print_cluster_mentions(cluster, subtoks=None):
    cluster_mentions = []
    for m_s, m_e in cluster:
        cluster_mentions.append(
            " ".join(subtoks[m_s : m_e + 1]) if subtoks else f"({m_s}, {m_e})"
        )
    entity_mentions = " | ".join(cluster_mentions)
    return entity_mentions


def merge_clusters(clusters, subtoks=None):
    merged_clusters = []
    merged2orig = {}
    for orig_i, cluster in enumerate(clusters):
        existing_i = None
        for mention in cluster:
            for merged_i, merged_cluster in enumerate(merged_clusters):
                if mention in merged_cluster:
                    existing_i = merged_i
                    break
            if existing_i is not None:
                break
        if existing_i is not None:
            cluster_str = print_cluster_mentions(cluster, subtoks)
            existing_str = print_cluster_mentions(merged_clusters[existing_i], subtoks)
            logger.info(f"Merging clusters: [{cluster_str}] ==> [{existing_str}]")
            merged2orig[existing_i].append(orig_i)
            merged_clusters[existing_i].update(cluster)
        else:
            merged2orig[len(merged2orig)] = [orig_i]
            merged_clusters.append(set(cluster))

    return [tuple(cluster) for cluster in merged_clusters], merged2orig


def from_clusters_to_mentions(merged_clusters, tok_to_subtok, mention2types=None):
    mention_starts, mention_ends, mention_cluster_id, mention_type = [], [], [], []
    merged_cluster_types = []

    for cluster_i, cluster in enumerate(merged_clusters):
        cluster_types = []
        for m_s_orig, m_e_orig in cluster:
            m_e_orig = m_e_orig[0], m_e_orig[1] + 1
            m_s, m_e = tok_to_subtok[m_s_orig], tok_to_subtok[m_e_orig] - 1  # Inclusive
            mention_starts.append(m_s)
            mention_ends.append(m_e)
            mention_type.append(
                -1 if mention2types is None else mention2types[(m_s, m_e)]
            )
            mention_cluster_id.append(cluster_i)
            cluster_types.append(mention_type[-1])
        merged_cluster_types.append(tuple(cluster_types))
    mentions = [(m_s, m_e) for m_s, m_e in zip(mention_starts, mention_ends)]
    sorted_indices = util.argsort(mentions)
    mention_starts = np.array(mention_starts)[sorted_indices].tolist()
    mention_ends = np.array(mention_ends)[sorted_indices].tolist()
    mention_type = np.array(mention_type)[sorted_indices].tolist()
    mention_cluster_id = np.array(mention_cluster_id)[sorted_indices].tolist()
    return (
        mention_starts,
        mention_ends,
        mention_type,
        mention_cluster_id,
    ), merged_cluster_types


def _get_doc_docred(inst, tokenizer, ner2id, rel2id, is_training, with_root=False):
    if with_root:
        inst["sents"].insert(0, ["root"])
        root_count = 0
        for entity_i, entity in enumerate(inst["vertexSet"]):
            for m in entity:
                if m["pos"][0] == -1:
                    m["pos"] = [0, 1]
                    m["sent_id"] = 0
                    m["name"] = "root"
                    root_count += 1
                    if root_count >= 2:
                        raise RuntimeError(f'{inst["title"]}: more than one root node')
                else:
                    m["sent_id"] += 1

    (
        subtoks,
        (subtok_tok_end, subtok_sent_end),
        (subtok_to_tok, tok_to_subtok),
    ) = tokenize_sentences(tokenizer, inst["sents"])
    entities, entity_types = [], []
    mention2types = {}
    for entity_i, entity in enumerate(inst["vertexSet"]):
        mentions, types = [], []
        for m in entity:
            m_s = tok_to_subtok[(m["sent_id"], m["pos"][0])]
            m_e = tok_to_subtok[(m["sent_id"], m["pos"][1])] - 1
            if (m_s, m_e) not in mentions:
                mentions.append((m_s, m_e))
                types.append(ner2id[m["type"]])
                mention2types[(m_s, m_e)] = types[-1]
        sorted_indices = util.argsort(mentions)  # Sort mentions
        entities.append([mentions[idx] for idx in sorted_indices])
        entity_types.append([types[idx] for idx in sorted_indices])

    pos_pairs = defaultdict(lambda: [0] * len(rel2id))
    if "labels" in inst:
        for rel in inst["labels"]:
            pos_pairs[(rel["h"], rel["t"])][rel2id[rel["r"]]] = 1

    def create_neg_rel_label():
        rel_label = [0] * len(rel2id)
        rel_label[rel2id["Na"]] = 1
        return rel_label

    entity_pairs, rel_labels = [], []
    num_pos_pairs, num_neg_pairs = 0, 0
    for h in range(len(entities)):
        for t in range(len(entities)):
            if True or h != t:
                if (h, t) in pos_pairs:
                    rel_label = pos_pairs[(h, t)]
                    num_pos_pairs += 1
                else:
                    rel_label = create_neg_rel_label()
                    num_neg_pairs += 1
                entity_pairs.append((h, t))
                rel_labels.append(rel_label)
    assert len(entity_pairs) == len(entities) ** 2

    merged_clusters = [
        [(subtok_to_tok[m_s], subtok_to_tok[m_e]) for m_s, m_e in entity]
        for entity in entities
    ]
    merged_clusters, merged2orig = merge_clusters(merged_clusters)
    (
        mention_starts,
        mention_ends,
        mention_type,
        mention_cluster_id,
    ), merged_cluster_types = from_clusters_to_mentions(
        merged_clusters, tok_to_subtok, mention2types
    )

    flattened_rel_labels = []
    for merged_h in range(len(merged2orig) + 1):
        for merged_t in range(len(merged2orig) + 1):
            if merged_h == 0 or merged_t == 0:
                rel_label = create_neg_rel_label()
            else:
                orig_h, orig_t = (
                    merged2orig[merged_h - 1][0],
                    merged2orig[merged_t - 1][0],
                )
                if (orig_h, orig_t) in pos_pairs:
                    rel_label = pos_pairs[(orig_h, orig_t)]
                else:
                    rel_label = create_neg_rel_label()
            flattened_rel_labels.append(rel_label)

    doc = {
        "title": inst["title"].strip(),
        "sents": inst["sents"],
        "subtoks": subtoks,
        "subtok_tok_end": subtok_tok_end,
        "subtok_sent_end": subtok_sent_end,
        "subtok_to_tok": subtok_to_tok,
        "tok_to_subtok": tok_to_subtok,
        "entities": entities,
        "entity_types": entity_types,
        "entity_pairs": entity_pairs,
        "rel_labels": rel_labels,
        "mention_starts": mention_starts,
        "mention_ends": mention_ends,
        "mention_type": mention_type,
        "mention_cluster_id": mention_cluster_id,
        "clusters": merged_clusters,
        "cluster_types": merged_cluster_types,
        "flattened_rel_labels": flattened_rel_labels,
    }
    return doc, (num_pos_pairs, num_neg_pairs)


def get_all_docs(
    dataset_name,
    file_path,
    tokenizer,
    ner2id=None,
    rel2id=None,
    is_training=False,
    with_root=False,
):
    """Interface for reading datasets into a unified format."""
    if dataset_name in ["docred", "dwie"] or dataset_name.startswith("med_"):
        assert ner2id is not None
        assert rel2id is not None
        instances = io_util.read_json(file_path)
        docs, total_pos_pairs, total_neg_pairs = [], 0, 0
        for inst in tqdm(instances, desc="Docs"):
            doc, (num_pos, num_neg) = _get_doc_docred(
                inst, tokenizer, ner2id, rel2id, is_training, with_root
            )
            docs.append(doc)
            total_pos_pairs += num_pos
            total_neg_pairs += num_neg
        logger.info(
            f"# total pos pairs: {total_pos_pairs}; # total neg pairs: {total_neg_pairs}"
        )
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")
    return docs


def convert_docs_to_features(
    dataset_name,
    docs,
    tokenizer,
    max_seq_len,
    overlapping,
    is_training,
    max_training_seg,
    show_example=False,
):
    short_example_shown, long_example_shown = False, False
    features = []

    for doc_i, doc in enumerate(tqdm(docs, desc="Features")):
        feature, is_max_context, _ = encode_long_sequence(
            doc["title"],
            tokenizer,
            doc["subtoks"],
            max_seq_len=max_seq_len,
            overlapping=overlapping,
            constraints=(doc["subtok_sent_end"], doc["subtok_tok_end"]),
        )
        num_seg = len(feature["input_ids"])
        if is_training and num_seg > max_training_seg:
            is_max_context = is_max_context[:max_training_seg]
            num_training_subtoks = sum(util.flatten(is_max_context))
            assert num_training_subtoks > 0
            feature["input_ids"] = feature["input_ids"][:max_training_seg]
            feature["attention_mask"] = feature["attention_mask"][:max_training_seg]
            feature["token_type_ids"] = feature["token_type_ids"][:max_training_seg]
            m_i_valid = (np.array(doc["mention_starts"]) < num_training_subtoks) & (
                np.array(doc["mention_ends"]) < num_training_subtoks
            )
            truncated_entities = []
            for entity in doc["entities"]:
                truncated_entity = [
                    (m_s, m_e) for m_s, m_e in entity if m_e < num_training_subtoks
                ]
                truncated_entities.append(truncated_entity)
            truncated_entity_pairs, truncated_rel_labels = [], []
            for (h, t), rel_label in zip(doc["entity_pairs"], doc["rel_labels"]):
                if truncated_entities[h] and truncated_entities[t]:
                    truncated_entity_pairs.append((h, t))
                    truncated_rel_labels.append(rel_label)

            feature["entities"] = truncated_entities
            feature["entity_pairs"] = truncated_entity_pairs
            feature["rel_labels"] = truncated_rel_labels
            feature["subtok_tok_end"] = doc["subtok_tok_end"][:num_training_subtoks]
            feature["subtok_sent_end"] = doc["subtok_sent_end"][:num_training_subtoks]
            feature["mention_starts"] = np.array(doc["mention_starts"])[
                m_i_valid
            ].tolist()
            feature["mention_ends"] = np.array(doc["mention_ends"])[m_i_valid].tolist()
            feature["mention_type"] = np.array(doc["mention_type"])[m_i_valid].tolist()
            feature["mention_cluster_id"] = np.array(doc["mention_cluster_id"])[
                m_i_valid
            ].tolist()
        else:
            feature["entities"] = doc["entities"]
            feature["entity_pairs"] = doc["entity_pairs"]
            feature["rel_labels"] = doc["rel_labels"]
            feature["subtok_tok_end"] = doc["subtok_tok_end"]
            feature["subtok_sent_end"] = doc["subtok_sent_end"]
            feature["mention_starts"] = doc["mention_starts"]
            feature["mention_ends"] = doc["mention_ends"]
            feature["mention_type"] = doc["mention_type"]
            feature["mention_cluster_id"] = doc["mention_cluster_id"]

        feature["is_max_context"] = is_max_context
        feature["entity_types"] = doc["entity_types"]
        feature["entity_pairs_h"] = [pair[0] for pair in feature["entity_pairs"]]
        feature["entity_pairs_t"] = [pair[1] for pair in feature["entity_pairs"]]
        del feature["entity_pairs"]
        feature["flattened_rel_labels"] = doc["flattened_rel_labels"]

        def get_sent_map(is_end):
            mapping, offset = [], 0
            for idx_is_end in is_end:
                mapping.append(offset)
                if idx_is_end:
                    offset += 1
            return mapping

        def get_tok_start_or_end(tok_end):
            start_or_end = tok_end[:]
            for i in range(len(start_or_end) - 1)[::-1]:
                if start_or_end[i]:
                    start_or_end[i + 1] = 1
            start_or_end[0] = 1
            return start_or_end

        feature["doc_len"] = sum(util.flatten(is_max_context))
        feature["tok_start_or_end"] = get_tok_start_or_end(
            feature.pop("subtok_tok_end")
        )
        feature["sent_map"] = get_sent_map(feature.pop("subtok_sent_end"))
        feature = {
            "title": doc["title"],
            "input_ids": feature["input_ids"],
            "attention_mask": feature["attention_mask"],
            "token_type_ids": feature["token_type_ids"],
            "is_max_context": feature["is_max_context"],
            "doc_len": feature["doc_len"],
            "tok_start_or_end": torch.tensor(
                feature["tok_start_or_end"], dtype=torch.long
            ),
            "sent_map": torch.tensor(feature["sent_map"], dtype=torch.long),
            "mention_starts": torch.tensor(feature["mention_starts"], dtype=torch.long),
            "mention_ends": torch.tensor(feature["mention_ends"], dtype=torch.long),
            "mention_type": torch.tensor(feature["mention_type"], dtype=torch.long),
            "mention_cluster_id": torch.tensor(
                feature["mention_cluster_id"], dtype=torch.long
            ),
            "flattened_rel_labels": torch.tensor(
                feature["flattened_rel_labels"], dtype=torch.long
            ),
            "entities": feature["entities"],
            "entity_types": feature["entity_types"],
            "entity_pairs_h": torch.tensor(feature["entity_pairs_h"], dtype=torch.long),
            "entity_pairs_t": torch.tensor(feature["entity_pairs_t"], dtype=torch.long),
            "rel_labels": torch.tensor(feature["rel_labels"], dtype=torch.long),
        }
        features.append(feature)
        if show_example and not short_example_shown and num_seg == 1:
            short_example_shown = True
            print(f'\nShort example title: {doc["title"]}')
            show_feature(tokenizer, feature, is_max_context)
        elif show_example and not long_example_shown and num_seg > 1:
            long_example_shown = True
            print(f'\nLong example title: {doc["title"]}')
            show_feature(tokenizer, feature, is_max_context)

    return features


def show_feature(tokenizer, feature, is_max_context):
    encoded_text, non_special_subtoks = [], []
    for seg_i, (seg_input_ids, seg_is_max_context) in enumerate(
        zip(feature["input_ids"], is_max_context)
    ):
        seg_subtoks = tokenizer.convert_ids_to_tokens(seg_input_ids)
        non_special_subtoks += [
            subtok
            for subtok, subtok_is_max_context in zip(seg_subtoks, seg_is_max_context)
            if subtok_is_max_context
        ]
        seg_text = [
            (f"*{subtok}*" if subtok_is_max_context else subtok)
            for subtok, subtok_is_max_context in zip(seg_subtoks, seg_is_max_context)
        ]
        encoded_text.append(f"SEGMENT_{seg_i}: " + " ".join(seg_text))
    print("\n".join(encoded_text))

    for e_i, (entity, entity_type) in enumerate(
        zip(feature["entities"], feature["entity_types"])
    ):
        mentions_str = print_cluster_mentions(entity, non_special_subtoks)
        print(f"ENTITY_{e_i} (TYPE {entity_type}): [{mentions_str}]")

    all_mentions = [
        (m_s, m_e)
        for m_s, m_e in zip(
            feature["mention_starts"].tolist(), feature["mention_ends"].tolist()
        )
    ]
    print(f"MENTIONS: {print_cluster_mentions(all_mentions, non_special_subtoks)}")
    print()

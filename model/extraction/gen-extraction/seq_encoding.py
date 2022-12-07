""" General encoding scheme for long sequence with Transformers. """
import util
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)


non_start_punct = {',', '.', '!', '?', '%', ':', '>', ')', ']', '}', '_'}  # Usually no space at front
follow_punct = {'(', '[', '{'}  # If follow these token, don't add space to current token


def tokenize_sentences(tokenizer, sents):
    """ Tokenize sentences of pre-split tokens to a sequence of subtokens.
    For BPE tokenizer, use heuristic to handle space.
    """
    subtoks, subtok_tok_end, subtok_sent_end = [], [], []
    subtok_to_tok, tok_to_subtok = [], {}
    is_bpe = util.is_bpe_tokenizer(tokenizer)

    for sent_i, sent in enumerate(sents):
        if not sent:
            continue
        for token_i, token in enumerate(sent):
            token = token.strip()
            tok_to_subtok[(sent_i, token_i)] = len(subtoks)  # (sent idx, token idx) -> first subtok idx w/o sentences
            if not token:
                continue

            is_added_vocab = token.startswith(f'<{util.FLAG_SPL_TOK}')

            if is_bpe:
                # Heuristic for adding space
                add_space = True
                if is_added_vocab:
                    add_space = False
                elif sent_i == 0 and token_i == 0:
                    add_space = False
                elif token_i > 0 and token in non_start_punct:
                    add_space = False
                elif token_i > 0 and sent[token_i - 1] in follow_punct:
                    add_space = False
                if add_space:
                    token = ' ' + token
            subs = tokenizer.tokenize(token)

            subtoks += subs
            subtok_tok_end += [0] * len(subs)
            subtok_tok_end[-1] = 1
            subtok_sent_end += [0] * len(subs)
            subtok_to_tok += [(sent_i, token_i)] * len(subs)
        tok_to_subtok[(sent_i, len(sent))] = len(subtoks)  # For last token; e.g. case for [left, right) span
        subtok_sent_end[-1] = 1
    return subtoks, (subtok_tok_end, subtok_sent_end), (subtok_to_tok, tok_to_subtok)


def encode_long_sequence(seq_id, tokenizer, subtoks, max_seq_len, overlapping=0, maximize_last_segment=True,
                         constraints=None):
    """
    Unlike MRC, here we set overlapping < max_seq_len // 2, so that each subtok appears at most two segments;
    thus, the max context of each subtok is always its first segment.
    Ignore the non-spaced first subtoken case in a segment for BPE tokenization.
    :param seq_id:
    :param tokenizer:
    :param subtoks:
    :param max_seq_len: max sequence length for a segment
    :param overlapping: overlapping context by at least # subtokens; can be more for the last segment
    :param maximize_last_segment:
    :param constraints: do not split segments at certain subtok; [num constraints, num subtoks]
    :return: segments, is_max_context
    """
    assert max_seq_len <= tokenizer.model_max_length
    assert overlapping < max_seq_len // 2, f'Currently only support overlapping < {max_seq_len // 2}'
    num_spl_added = tokenizer.model_max_length - tokenizer.max_len_single_sentence

    # Perform segmentation with constraints and overlapping
    segments, is_max_context, seg_subtok_range = [], [], []
    till_idx = 0  # Exclusive
    while till_idx < len(subtoks):
        seg_start = max(0, till_idx - overlapping)  # Segment start idx considering overlapping context
        seg_end = min(len(subtoks), seg_start + max_seq_len - num_spl_added)  # Exclusive

        # Adjust seg end idx considering constraints
        if constraints:
            for c_i, constraint in enumerate(constraints):
                while seg_end > seg_start and not constraint[seg_end - 1]:
                    seg_end -= 1

                if seg_end <= seg_start and c_i != len(constraints) - 1:
                    logger.info(f'{seq_id}: try to segment by next constraint {c_i + 1}')
                    seg_end = min(len(subtoks), seg_start + max_seq_len - num_spl_added)
                elif seg_end <= seg_start:
                    raise RuntimeError(f'{seq_id}: cannot split segment by neither constraints')
                else:
                    break

        # Adjust seg start idx for last segment if maximize
        if seg_end == len(subtoks) and maximize_last_segment:
            seg_start = max(0, len(subtoks) - max_seq_len + num_spl_added)

        # Set segmentation
        segments.append(subtoks[seg_start: seg_end])
        is_max_context.append([0] * (till_idx - seg_start) + [1] * (seg_end - till_idx))
        seg_subtok_range.append((seg_start, seg_end))

        till_idx = seg_end

    # Encode segments
    longest_seg_len = max([len(seg) for seg in segments]) + num_spl_added
    encoded_segments = []
    for seg_i, segment in enumerate(segments):
        encoded = tokenizer.prepare_for_model(tokenizer.convert_tokens_to_ids(segment),
                                              padding='max_length', max_length=longest_seg_len,
                                              return_attention_mask=True, return_token_type_ids=True,
                                              return_special_tokens_mask=True)
        encoded_segments.append(encoded)

        # Adjust is_max_context for special tokens: CLS, SEP, PAD, etc.
        # Assume special tokens appear left or right
        special_tokens_mask = encoded.pop('special_tokens_mask')
        num_left_special = special_tokens_mask.index(0)
        num_right_special = longest_seg_len - util.rindex(special_tokens_mask, 0) - 1
        is_max_context[seg_i] = [0] * num_left_special + is_max_context[seg_i] + [0] * num_right_special
        assert len(encoded['input_ids']) == len(is_max_context[seg_i])

    # Re-arrange encoded_segments
    encoded_segments = {
        'input_ids': [encoded_seg['input_ids'] for encoded_seg in encoded_segments],
        'attention_mask': [encoded_seg['attention_mask'] for encoded_seg in encoded_segments],
        'token_type_ids': [encoded_seg['token_type_ids'] for encoded_seg in encoded_segments],
    }

    # Sanity check
    assert len(subtoks) == sum(util.flatten(is_max_context))
    return encoded_segments, is_max_context, seg_subtok_range

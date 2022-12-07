import propagation
from model_coref import CorefModel
import torch
import logging
import torch.nn.functional as F
import losses
import math

logger = logging.getLogger(__name__)


class MedJointModel(CorefModel):
    def __init__(self, config, num_entity_types=0):
        super(MedJointModel, self).__init__(config, num_entity_types)

        self.num_re_labels = config["num_re_labels"]
        self.max_re_labels = config["max_re_labels"]
        self.rel_emb_hidden_size = 32

        self.re_mention_hidden_size = (
            config["re_mention_hidden_size"]
            if config["re_transform_mention"]
            else self.span_hidden_size
        )
        self.re_head_transform = self.make_linear(
            self.span_hidden_size, self.re_mention_hidden_size
        )
        self.re_tail_transform = self.make_linear(
            self.span_hidden_size, self.re_mention_hidden_size
        )
        self.re_head_prior = self.make_linear(
            self.re_mention_hidden_size, self.num_re_labels, bias=False
        )
        self.re_tail_prior = self.make_linear(
            self.re_mention_hidden_size, self.num_re_labels, bias=False
        )
        self.re_fast_bilinear = self.make_linear(
            self.re_mention_hidden_size,
            self.re_mention_hidden_size * self.num_re_labels,
        )
        self.re_fast_bilinear_propagation = self.make_linear(
            self.re_mention_hidden_size,
            self.re_mention_hidden_size * self.num_re_labels,
        )
        self.re_num_blocks = config["re_num_blocks"]
        self.re_block_size = self.re_mention_hidden_size // self.re_num_blocks
        self.re_slow_bilinear = self.make_linear(
            self.re_mention_hidden_size * self.re_block_size, self.num_re_labels
        )
        self.re_scoring = self.get_re_logits_fast
        self.re_loss_fct = losses.ATLoss()
        self.dygie_transform = self.make_linear(
            self.num_re_labels, self.re_mention_hidden_size, bias=False
        )
        self.dygie_gate = self.make_linear(
            self.re_mention_hidden_size * 2, self.re_mention_hidden_size
        )
        self.rel_dependent_transform = self.make_linear(
            self.re_mention_hidden_size,
            self.re_mention_hidden_size * self.num_re_labels,
        )
        self.rel_independent_transform = self.make_linear(
            self.re_mention_hidden_size + self.rel_emb_hidden_size,
            self.re_mention_hidden_size,
        )
        self.rel_emb = self.make_embedding(self.num_re_labels, self.rel_emb_hidden_size)
        self.re_rel_attn = self.make_ffnn(2 * self.re_mention_hidden_size, 0, 1)

        self.debug = True

    def get_re_labels(self, re_logits):
        return self.re_loss_fct.get_label(re_logits, self.max_re_labels)

    def forward_single(
        self,
        tokens,
        tok_start_or_end=None,
        sent_map=None,
        speaker_ids=None,
        mention_starts=None,
        mention_ends=None,
        mention_type=None,
        mention_cluster_id=None,
        flattened_rel_labels=None,
        **kwargs,
    ):
        conf, num_tokens = self.config, tokens.size()[0]
        gold_starts, gold_ends, gold_types, gold_cluster_map = (
            mention_starts,
            mention_ends,
            mention_type,
            mention_cluster_id,
        )
        device = tokens.device

        (
            (
                top_span_starts,
                top_span_ends,
                top_span_emb,
                top_span_mention_scores,
                top_span_type_logits,
                top_span_cluster_ids,
            ),
            (
                _,
                _,
                _,
                candidate_mention_scores,
                candidate_type_logits,
                candidate_cluster_ids,
                candidate_type_labels,
            ),
            selected_idx,
            num_top_spans,
        ) = self.extract_spans(
            tokens,
            tok_start_or_end,
            sent_map,
            gold_starts,
            gold_ends,
            gold_types,
            gold_cluster_map,
        )
        if conf["re_transform_mention"]:
            if conf["re_distinguish_ht"]:
                re_head_emb = self.dropout(
                    torch.tanh(self.re_head_transform(top_span_emb))
                )
                re_tail_emb = self.dropout(
                    torch.tanh(self.re_tail_transform(top_span_emb))
                )
            else:
                re_head_emb = re_tail_emb = self.dropout(
                    torch.tanh(self.re_head_transform(top_span_emb))
                )
        else:
            re_head_emb = re_tail_emb = top_span_emb
        re_pair_logits = self.re_scoring(
            self.re_fast_bilinear, re_head_emb, re_tail_emb
        )
        if conf["re_add_prior"]:
            head_priors = self.re_head_prior(re_head_emb)
            tail_priors = self.re_tail_prior(re_tail_emb)
            re_priors = (head_priors.unsqueeze(1) + tail_priors.unsqueeze(0)).view(
                -1, tail_priors.size()[-1]
            )
            re_pair_logits += re_priors
        if conf["re_propagation"]:
            new_re_pair_logits, (
                re_head_emb,
                re_tail_emb,
            ) = self.do_re_propagation_rel_dependent(
                re_head_emb, re_tail_emb, re_pair_logits
            )
            if conf["re_propagation_update_scores"]:
                if conf["re_propagation_only_last"]:
                    re_pair_logits = new_re_pair_logits
                    if conf["re_add_prior"]:
                        re_pair_logits += re_priors
                else:
                    re_pair_logits += new_re_pair_logits

        outputs = (
            top_span_starts,
            top_span_ends,
            top_span_mention_scores,
            top_span_type_logits,
            re_pair_logits,
        )
        if gold_cluster_map is None:
            return outputs

        if conf["use_span_type"]:
            loss_gold_mention, loss_non_gold_mention = losses.get_mention_type_loss(
                candidate_type_logits, candidate_type_labels
            )
            loss_ner = conf["mention_loss_coef"] * (
                loss_gold_mention + loss_non_gold_mention
            )
        else:
            loss_gold_mention, loss_non_gold_mention = losses.get_mention_score_loss(
                candidate_mention_scores, candidate_cluster_ids
            )
            loss_non_gold_mention *= 0.5
            loss_ner = conf["mention_loss_coef"] * (
                loss_gold_mention + loss_non_gold_mention
            )
        re_pair_labels = self.adapt_re_gold_labels(
            top_span_cluster_ids, flattened_rel_labels
        )
        loss_re = self.re_loss_fct(re_pair_logits, re_pair_labels.to(torch.float)).sum()
        loss = loss_ner + loss_re

        if self.debug:
            if self.forward_steps % (conf["report_frequency"] * 2) == 0:
                logger.info(f"---------debug step: {self.forward_steps}---------")
                logger.info(
                    f"NER / RE Loss: {loss_ner.item():.4f} / {loss_re.item():.4f}"
                )
        self.forward_steps += 1

        return loss, outputs

    def get_re_logits_fast(self, bilinear, re_head_emb, re_tail_emb):
        intermediate = (
            bilinear(re_head_emb)
            .view(-1, self.re_mention_hidden_size, self.num_re_labels)
            .permute(2, 0, 1)
        )
        target = torch.transpose(re_tail_emb, 0, 1).unsqueeze(0)
        re_logits = torch.matmul(intermediate, target)
        re_logits = re_logits.permute(1, 2, 0).contiguous()
        re_logits = re_logits.view(-1, re_logits.size()[-1])
        return re_logits

    def do_re_propagation_rel_dependent(self, head_emb, tail_emb, re_logits):
        conf = self.config
        for prop_i in range(conf["re_propagation"]):
            mention_transform, rel_emb = None, None
            if conf["re_propagation_transform"]:
                if conf["re_propagation_rel_emb"]:
                    (
                        mention_transform,
                        rel_emb,
                    ) = self.rel_independent_transform, self.dropout(
                        self.rel_emb.weight
                    )
                else:
                    mention_transform = self.rel_dependent_transform
            attended_head_emb, attended_tail_emb = propagation.propagate_re(
                head_emb,
                tail_emb,
                re_logits,
                mention_transform,
                rel_emb,
                void_negative=conf["re_propagation_void_negative"],
                with_atloss=True,
            )
            rel_reduction = (
                self.apply_re_rel_attention
                if conf["re_propagation_rel_attention"]
                else self.apply_re_rel_reduce
            )
            new_head_emb = rel_reduction(
                head_emb, self.dropout(attended_tail_emb), self.re_rel_attn
            )
            new_tail_emb = rel_reduction(
                tail_emb, self.dropout(attended_head_emb), self.re_rel_attn
            )
            head_emb, tail_emb = self.dropout(new_head_emb), self.dropout(new_tail_emb)
            if conf["re_propagation_update_scores"]:
                bilinear = (
                    self.re_fast_bilinear
                    if self.config["re_propagation_same_bilinear"]
                    else self.re_fast_bilinear_propagation
                )
                re_logits = self.get_re_logits_fast(bilinear, head_emb, tail_emb)
                re_logits = re_logits.view(-1, re_logits.size()[-1])
        return re_logits, (head_emb, tail_emb)

    @classmethod
    def apply_re_rel_attention(cls, span_emb, span_rel_emb, rel_attn):
        (num_labels, num_spans), device = span_rel_emb.size()[0:2], span_rel_emb.device
        span_rel_emb = span_rel_emb.transpose(0, 1)
        attn_emb = torch.cat(
            [span_rel_emb, span_emb.unsqueeze(1).repeat(1, num_labels, 1)], dim=-1
        )
        attentions = rel_attn(attn_emb).squeeze(-1)
        attentions = torch.cat(
            [torch.ones(num_spans, 1, device=device), attentions], dim=-1
        )
        attentions += torch.log((attentions > 0).to(torch.float))
        attentions = F.softmax(attentions, dim=-1)
        attn_emb = torch.cat([span_emb.unsqueeze(1), span_rel_emb], dim=1)
        attended_emb = attn_emb * attentions.unsqueeze(-1)
        attended_emb = attended_emb.sum(dim=1, keepdims=False)
        return attended_emb

    @classmethod
    def apply_re_rel_reduce(cls, span_emb, span_rel_emb, placeholder):
        reduced_emb = span_emb + span_rel_emb.mean(dim=0)
        return reduced_emb

    @classmethod
    def adapt_re_gold_labels(cls, span_cluster_ids, flattened_rel_labels):
        num_spans = span_cluster_ids.size()[0]
        num_clusters = math.isqrt(flattened_rel_labels.size()[0])
        matrix_rel_labels = flattened_rel_labels.view(num_clusters, num_clusters, -1)
        pair_cluster_ids_x = span_cluster_ids.repeat_interleave(num_spans, dim=0)
        pair_cluster_ids_y = span_cluster_ids.repeat(num_spans)
        re_pair_labels = matrix_rel_labels[pair_cluster_ids_x, pair_cluster_ids_y]
        return re_pair_labels

import torch
import torch.nn.functional as F


def _propagate_re_uni(span_emb, re_weights, mention_transform=None, rel_emb=None, do_softmax=False):
    num_labels, num_spans = re_weights.size()[0:2]
    if do_softmax:
        re_weights = F.softmax(re_weights, dim=-1)
    if mention_transform is None:
        transformed = span_emb.unsqueeze(0)
    else:
        if rel_emb is None:
            transformed = torch.tanh(mention_transform(span_emb))
            transformed = transformed.view(span_emb.size()[:2] + (num_labels,))
            transformed = transformed.permute(2, 0, 1).contiguous()
        else:
            transformed = torch.cat([
                span_emb.unsqueeze(0).repeat(num_labels, 1, 1),
                rel_emb.unsqueeze(1).repeat(1, num_spans, 1)], dim=-1)
            transformed = mention_transform(transformed)
    attended_emb = torch.matmul(re_weights, transformed)
    return attended_emb


def propagate_re(head_emb, tail_emb, re_logits, mention_transform=None, rel_emb=None,
                 with_atloss=True, void_negative=False, do_softmax=False):
    num_spans = tail_emb.size()[0]

    if with_atloss:
        th_logits = re_logits[:, 0:1]
        re_weights = re_logits - th_logits
        re_weights = F.leaky_relu(re_weights, negative_slope=0.1) if void_negative else re_weights
    else:
        re_weights = F.leaky_relu(re_logits, negative_slope=0.1) if void_negative else re_logits
    re_weights = re_weights.view(num_spans, num_spans, -1)
    re_h2t_weights = re_weights.permute(2, 0, 1)
    re_t2h_weights = re_weights.permute(2, 1, 0)
    attended_tail_emb = _propagate_re_uni(tail_emb, re_h2t_weights, mention_transform, rel_emb, do_softmax)
    attended_head_emb = _propagate_re_uni(head_emb, re_t2h_weights, mention_transform, rel_emb, do_softmax)
    return attended_head_emb, attended_tail_emb

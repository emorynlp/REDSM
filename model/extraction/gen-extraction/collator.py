import util
from dataclasses import dataclass
import torch
from torch import Tensor


@dataclass
class FeatureCollator:
    tokenizer: None
    device: torch.device('cpu')

    def __post_init__(self):
        self.ignored_keys = {'title'}
        self.addl_batched_attrs = {'is_max_context': 0}

    @classmethod
    def _right_pad_batched_attr(cls, attr_segs, pad_to_len, pad_val):
        return [(attr_segs[seg_i] + [pad_val] * (pad_to_len - len(attr_segs[seg_i])))
                for seg_i in range(len(attr_segs))]

    def __call__(self, features):
        all_keys = set(features[0].keys())
        collated = {
            'input_ids': util.flatten([f['input_ids'] for f in features]),
            'attention_mask': util.flatten([f['attention_mask'] for f in features])
        }
        if 'token_type_ids' in all_keys:
            collated['token_type_ids'] = util.flatten([f['token_type_ids'] for f in features])
        collated = self.tokenizer.pad(collated, padding=True, pad_to_multiple_of=8)
        num_seg, seg_len = len(collated['input_ids']), len(collated['input_ids'][0])

        for attr_to_batch, pad_val in self.addl_batched_attrs.items():
            if attr_to_batch in all_keys:
                collated[attr_to_batch] = self._right_pad_batched_attr(
                    util.flatten([f[attr_to_batch] for f in features]), pad_to_len=seg_len, pad_val=pad_val)
                assert len(collated[attr_to_batch]) == num_seg
        collated = {k: torch.tensor(v, device=self.device) for k, v in collated.items()}
        others = {feat_attr: [(f[feat_attr].to(self.device) if isinstance(f[feat_attr], Tensor) else f[feat_attr])
                              for f in features]
                  for feat_attr in all_keys if feat_attr not in collated and feat_attr not in self.ignored_keys}
        collated.update(others)

        return collated

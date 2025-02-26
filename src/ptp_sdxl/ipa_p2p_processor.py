from __future__ import annotations

import abc
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
from diffusers.models.attention import Attention

from diffusers.utils import deprecate
from diffusers.image_processor import IPAdapterMaskProcessor

class IPAP2PCrossAttnProcessor:
    def __init__(self, controller, place_in_unet, ipa_processor):
        super().__init__()
        self.controller = controller
        self.place_in_unet = place_in_unet

        # copy params from ipa_processor
        self.hidden_size = ipa_processor.hidden_size
        self.cross_attention_dim = ipa_processor.cross_attention_dim
        self.num_tokens = ipa_processor.num_tokens
        self.scale = ipa_processor.scale
        self.to_k_ip = ipa_processor.to_k_ip
        self.to_v_ip = ipa_processor.to_v_ip

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        scale: float = 1.0,
        ip_adapter_masks: Optional[torch.Tensor] = None,
    ):
        # separate ip_hidden_states from encoder_hidden_states
        if encoder_hidden_states is not None:
            if isinstance(encoder_hidden_states, tuple):
                encoder_hidden_states, ip_hidden_states = encoder_hidden_states
            else:
                deprecation_message = (
                    "You have passed a tensor as `encoder_hidden_states`. This is deprecated and will be removed in a future release."
                    " Please make sure to update your script to pass `encoder_hidden_states` as a tuple to suppress this warning."
                )
                deprecate("encoder_hidden_states not a tuple", "1.0.0", deprecation_message, standard_warn=False)
                end_pos = encoder_hidden_states.shape[1] - self.num_tokens[0]
                encoder_hidden_states, ip_hidden_states = (
                    encoder_hidden_states[:, :end_pos, :],
                    [encoder_hidden_states[:, end_pos:, :]],
                )

        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        query = attn.to_q(hidden_states)

        is_cross = encoder_hidden_states is not None
        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        # one line change
        self.controller(attention_probs, is_cross, self.place_in_unet)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        if ip_adapter_masks is not None:
            if not isinstance(ip_adapter_masks, List):
                # for backward compatibility, we accept `ip_adapter_mask` as a tensor of shape [num_ip_adapter, 1, height, width]
                ip_adapter_masks = list(ip_adapter_masks.unsqueeze(1))
            if not (len(ip_adapter_masks) == len(self.scale) == len(ip_hidden_states)):
                raise ValueError(
                    f"Length of ip_adapter_masks array ({len(ip_adapter_masks)}) must match "
                    f"length of self.scale array ({len(self.scale)}) and number of ip_hidden_states "
                    f"({len(ip_hidden_states)})"
                )
            else:
                for index, (mask, scale, ip_state) in enumerate(zip(ip_adapter_masks, self.scale, ip_hidden_states)):
                    if not isinstance(mask, torch.Tensor) or mask.ndim != 4:
                        raise ValueError(
                            "Each element of the ip_adapter_masks array should be a tensor with shape "
                            "[1, num_images_for_ip_adapter, height, width]."
                            " Please use `IPAdapterMaskProcessor` to preprocess your mask"
                        )
                    if mask.shape[1] != ip_state.shape[1]:
                        raise ValueError(
                            f"Number of masks ({mask.shape[1]}) does not match "
                            f"number of ip images ({ip_state.shape[1]}) at index {index}"
                        )
                    if isinstance(scale, list) and not len(scale) == mask.shape[1]:
                        raise ValueError(
                            f"Number of masks ({mask.shape[1]}) does not match "
                            f"number of scales ({len(scale)}) at index {index}"
                        )
        else:
            ip_adapter_masks = [None] * len(self.scale)

        # for ip-adapter
        for current_ip_hidden_states, scale, to_k_ip, to_v_ip, mask in zip(
            ip_hidden_states, self.scale, self.to_k_ip, self.to_v_ip, ip_adapter_masks
        ):
            skip = False
            if isinstance(scale, list):
                if all(s == 0 for s in scale):
                    skip = True
            elif scale == 0:
                skip = True
            if not skip:
                if mask is not None:
                    if not isinstance(scale, list):
                        scale = [scale] * mask.shape[1]

                    current_num_images = mask.shape[1]
                    for i in range(current_num_images):
                        ip_key = to_k_ip(current_ip_hidden_states[:, i, :, :])
                        ip_value = to_v_ip(current_ip_hidden_states[:, i, :, :])

                        ip_key = attn.head_to_batch_dim(ip_key)
                        ip_value = attn.head_to_batch_dim(ip_value)

                        ip_attention_probs = attn.get_attention_scores(query, ip_key, None)
                        _current_ip_hidden_states = torch.bmm(ip_attention_probs, ip_value)
                        _current_ip_hidden_states = attn.batch_to_head_dim(_current_ip_hidden_states)

                        mask_downsample = IPAdapterMaskProcessor.downsample(
                            mask[:, i, :, :],
                            batch_size,
                            _current_ip_hidden_states.shape[1],
                            _current_ip_hidden_states.shape[2],
                        )

                        mask_downsample = mask_downsample.to(dtype=query.dtype, device=query.device)

                        hidden_states = hidden_states + scale[i] * (_current_ip_hidden_states * mask_downsample)
                else:
                    ip_key = to_k_ip(current_ip_hidden_states)
                    ip_value = to_v_ip(current_ip_hidden_states)

                    ip_key = attn.head_to_batch_dim(ip_key)
                    ip_value = attn.head_to_batch_dim(ip_value)

                    ip_attention_probs = attn.get_attention_scores(query, ip_key, None)
                    current_ip_hidden_states = torch.bmm(ip_attention_probs, ip_value)
                    current_ip_hidden_states = attn.batch_to_head_dim(current_ip_hidden_states)

                    hidden_states = hidden_states + scale * current_ip_hidden_states

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states
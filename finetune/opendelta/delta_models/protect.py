from typing import Optional, Union

from opendelta.utils.signature import get_arg_names, get_arg_names_inside_func
from opendelta.utils.name_based_addressing import *
from opendelta.basemodel import DeltaBase
import torch.nn as nn
from opendelta import BaseDeltaConfig
import math
from dataclasses import dataclass, field

class LowRankLinear(nn.Module):
    #  ------------------------------------------------------------------------------------------
    #  Copyright (c) Microsoft Corporation. All rights reserved.
    #  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
    #  ------------------------------------------------------------------------------------------
    #  copy from loralib and do some refactor
    def __init__(self,
        in_features,
        out_features,
        weight,
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
    ):
        super().__init__()
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        if r > 0:
            self.lora_A = nn.Parameter(weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def forward(self, x):
        return (self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling

@dataclass
class PretectArguments:
    r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0

class PretectConfig(BaseDeltaConfig):
    r"""
    This is the configuration class to store the configuration of a :py:class:`~LoraModel`

    """
    def __init__(
        self,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        arg_names = get_arg_names_inside_func(self.__init__)
        for arg_name in arg_names:
            if not hasattr(self, arg_name): # the arg has not been registered in parent config
                setattr(self, arg_name, locals()[arg_name])


class ProtectModel(DeltaBase):
    config_class = PretectConfig
    delta_type = "protect"
    default_modified_modules = ['attn@.q@', 'attn@.v@']
    _supported_backends = ['hf', 'bmt']
    _need_pseudo_data = False

    def __init__(self,
                 backbone_model: nn.Module,
                 lora_r=8,
                 lora_alpha=16,
                 lora_dropout=0.0,
                 modified_modules: Optional[List[str]] = None,
                 unfrozen_modules: Optional[List[str]] = None,
                 exclude_modules: Optional[List[str]] = None,
                 common_structure: Optional[bool] = None,
                 interactive_modify: Optional[Union[bool, int]] = False,
                 backend: Optional[str] = "hf",
                 ):
        DeltaBase.__init__(self,
                           backbone_model,
                           modified_modules=modified_modules,
                           unfrozen_modules=unfrozen_modules,
                           common_structure=common_structure,
                           interactive_modify=interactive_modify,
                           backend=backend,
                           )
        arg_names = get_arg_names_inside_func(self.__init__)
        for arg_name in arg_names:
            if not hasattr(self, arg_name): # not registered in parent class
                setattr(self, arg_name, locals()[arg_name])

        self.delta_modules = nn.ModuleList()

        self.add_all_delta_to_backbone(self.backbone_model,
                                       self.modified_modules,
                                       )

        # Add projection layers for prefix and text representations
        self.prefix_projection = nn.Linear(self.backbone_model.config.hidden_size, 128)
        self.text_projection = nn.Linear(self.backbone_model.config.hidden_size, 128)

    def update_module(self, module: nn.Module, key: str):
        parent_ref, child_name, child_ref = self.find_module(module, key)
        parallel_module = self.new_module_like(child_module=child_ref)
        self.insert_parallel_module(child_ref, delta_module=parallel_module, delta_name="lora")

    def _pseudo_data_to_instantiate(self, module):
        # no need to pass pseudo input, so overwrite it
        pass

    def new_module_like(self, child_module):
        in_features, out_features = child_module.in_features, child_module.out_features
        new_module = LowRankLinear(in_features=in_features,
                                   out_features=out_features,
                                   weight=child_module.weight,
                                   r=self.lora_r,
                                   lora_alpha=self.lora_alpha,
                                   lora_dropout=self.lora_dropout)
        if self.backend == "bmt":
            import bmtrain as bmt
            new_module = bmt.BMTrainModelWrapper(new_module)
        
        self.delta_modules.append(new_module)
        return new_module

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.backbone_model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_states = outputs[0]

        # Obtain the representations for the prefix and text
        prefix_representation = self.prefix_projection(hidden_states[:, 0, :])
        text_representation = self.text_projection(hidden_states[:, 1:, :].mean(dim=1))

        # Calculate InfoNCE loss
        loss = self.infonce_loss(prefix_representation, text_representation)

        return outputs, loss

    def infonce_loss(self, prefix_representation, text_representation, temperature=0.1):
        batch_size = prefix_representation.size(0)
        representations = torch.cat([prefix_representation, text_representation], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

        labels = torch.cat([torch.arange(batch_size) for _ in range(2)]).to(prefix_representation.device)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

        logits = similarity_matrix / temperature
        logits_max, _ = logits.max(dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        exp_logits = logits.exp()
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True))

        mean_log_prob_pos = (labels * log_prob).sum(dim=1) / labels.sum(dim=1)
        loss = -mean_log_prob_pos.mean()

        return loss
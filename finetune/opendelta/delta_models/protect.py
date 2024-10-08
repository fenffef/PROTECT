from typing import Optional, Union, List
import torch
import torch.nn as nn
import math
from opendelta.utils.signature import get_arg_names_inside_func
from opendelta.utils.name_based_addressing import *
from opendelta.basemodel import DeltaBase
from opendelta import BaseDeltaConfig
from dataclasses import dataclass

class SLALinear(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 weight,
                 r=8,
                 sla_alpha=16,
                 sla_dropout=0.0):
        super().__init__()
        self.r = r
        self.sla_alpha = sla_alpha
        self.sla_dropout = sla_dropout
        if sla_dropout > 0.:
            self.sla_dropout = nn.Dropout(p=sla_dropout)
        else:
            self.sla_dropout = lambda x: x
        if r > 0:
            self.sla_A = nn.Parameter(weight.new_zeros((r, in_features)))
            self.sla_B = nn.Parameter(weight.new_zeros((out_features, r)))
            self.scaling = self.sla_alpha / self.r
            nn.init.kaiming_uniform_(self.sla_A, a=math.sqrt(5))
            nn.init.zeros_(self.sla_B)

    def forward(self, x):
        return (self.sla_dropout(x) @ self.sla_A.T @ self.sla_B.T) * self.scaling


@dataclass
class ProtectArguments:
    r: int = 8
    sla_alpha: int = 16
    sla_dropout: float = 0.0


class ProtectConfig(BaseDeltaConfig):
    def __init__(self,
                 sla_r=8,
                 sla_alpha=16,
                 sla_dropout=0.0,
                 **kwargs):
        super().__init__(**kwargs)
        arg_names = get_arg_names_inside_func(self.__init__)
        for arg_name in arg_names:
            if not hasattr(self, arg_name):  # the arg has not been registered in parent config
                setattr(self, arg_name, locals()[arg_name])


# MINE (Mutual Information Neural Estimator) for minimizing mutual information between sla and cap representations
class MINE(nn.Module):
    def __init__(self, input_size):
        super(MINE, self).__init__()
        self.fc1 = nn.Linear(input_size * 2, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x, y):
        combined = torch.cat((x, y), dim=-1)
        h = torch.relu(self.fc1(combined))
        return self.fc2(h)

    def mutual_information(self, original_rep, sla_rep, batch_size):
        joint = self.forward(original_rep, sla_rep)
        sla_rep_perm = sla_rep[torch.randperm(batch_size)]
        marginal = self.forward(original_rep, sla_rep_perm)
        mi_estimate = torch.mean(joint) - torch.log(torch.mean(torch.exp(marginal)))
        return mi_estimate


class ProtectModel(DeltaBase):
    config_class = ProtectConfig
    delta_type = "protect"
    default_modified_modules = ['attn@.q@', 'attn@.v@']
    _supported_backends = ['hf', 'bmt']
    _need_pseudo_data = False

    def __init__(self,
                 backbone_model: nn.Module,
                 sla_r=8,
                 sla_alpha=16,
                 sla_dropout=0.0,
                 mine_input_size=768,  # 假设表征维度为 768
                 modified_modules: Optional[List[str]] = None,
                 unfrozen_modules: Optional[List[str]] = None,
                 exclude_modules: Optional[List[str]] = None,
                 common_structure: Optional[bool] = None,
                 interactive_modify: Optional[Union[bool, int]] = False,
                 backend: Optional[str] = "hf"):
        DeltaBase.__init__(self,
                           backbone_model,
                           modified_modules=modified_modules,
                           unfrozen_modules=unfrozen_modules,
                           common_structure=common_structure,
                           interactive_modify=interactive_modify,
                           backend=backend)
        arg_names = get_arg_names_inside_func(self.__init__)
        for arg_name in arg_names:
            if not hasattr(self, arg_name):  # not registered in parent class
                setattr(self, arg_name, locals()[arg_name])

        self.delta_modules = nn.ModuleList()
        self.mine = MINE(input_size=mine_input_size)  # Initialize MINE
        self.add_all_delta_to_backbone(self.backbone_model, self.modified_modules)

    def update_module(self, module: nn.Module, key: str):
        parent_ref, child_name, child_ref = self.find_module(module, key)
        parallel_module = self.new_module_like(child_module=child_ref)
        self.insert_parallel_module(child_ref, delta_module=parallel_module, delta_name="protect")

    def _pseudo_data_to_instantiate(self, module):
        # No need to pass pseudo input, so overwrite it
        pass

    def new_module_like(self, child_module):
        in_features, out_features = child_module.in_features, child_module.out_features
        new_module = SLALinear(in_features=in_features,
                                   out_features=out_features,
                                   weight=child_module.weight,
                                   r=self.sla_r,
                                   sla_alpha=self.sla_alpha,
                                   sla_dropout=self.sla_dropout)
        if self.backend == "bmt":
            import bmtrain as bmt
            new_module = bmt.BMTrainModelWrapper(new_module)

        self.delta_modules.append(new_module)
        return new_module

    def forward(self, inputs, compute_loss=True):
        original_output = self.backbone_model(inputs)

        SLA_output = self.apply_SLA(inputs)

        batch_size = original_output.size(0)
        CAP_loss = self.mine.mutual_information(original_output, SLA_output, batch_size)

        if compute_loss:
            SLA_loss = self.compute_SLA_loss(SLA_output, inputs)
            total_loss = SLA_loss + CAP_loss
            return total_loss

        return SLA_output

    def apply_SLA(self, inputs):
        modified_output = self.backbone_model(inputs)
        return modified_output

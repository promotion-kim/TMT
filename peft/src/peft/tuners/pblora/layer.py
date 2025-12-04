# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import math
import warnings
from typing import Any, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils.other import transpose

class PBLoraLayer(BaseTunerLayer):
    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names = ("pblora_A", "pblora_B", "pref_vec", "pblora_W1", "pblora_W2")
    # All names of other parameters that may contain adapter-related parameters
    other_param_names = ("r1", "r2", "lora_alpha", "scaling", "lora_dropout")

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        self.base_layer = base_layer
        self.r1 = {}
        self.r2 = {}
        self.obj_num = {}
        # self.lora_num = {}
        self.lora_alpha = {}
        self.scaling = {}
        self.lora_dropout = nn.ModuleDict({})
        # W: [n, m], obj_num: k
        self.pref_vec = nn.ParameterDict({}) # k
        self.pblora_A = nn.ParameterDict({}) # r1+r2, m
        self.pblora_B = nn.ParameterDict({}) # n, r1+r2
        self.pblora_W1 = nn.ParameterDict({}) # r1, r1
        self.pblora_W2 = nn.ParameterDict({}) # k, r2**2
        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []
        self.kwargs = kwargs

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        else:
            raise ValueError(f"Unsupported layer type {type(base_layer)}")

        self.in_features = in_features
        self.out_features = out_features

    def update_layer(self, adapter_name, r1, r2, obj_num, 
                    pref_vec_init, lora_alpha, lora_dropout):
        # This code works for linear layers, override for other layer types
        if r1 <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        self.r1[adapter_name] = r1
        self.r2[adapter_name] = r2
        self.obj_num[adapter_name] = obj_num
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        # Actual trainable parameters
        self.pblora_A[adapter_name] = nn.Parameter(
                torch.zeros(r1+r2, self.in_features) , requires_grad=True
        )

        self.pblora_B[adapter_name] = nn.Parameter(
                torch.zeros(self.out_features, r1+r2), requires_grad=True
        )

        if pref_vec_init is None:
            self.pref_vec[adapter_name] = nn.Parameter(
                    torch.zeros(obj_num,), requires_grad=False
            )
        else:
            assert len(pref_vec_init) == obj_num
            self.pref_vec[adapter_name] = nn.Parameter(
                    torch.tensor(pref_vec_init), requires_grad=False
            )

        self.pblora_W1[adapter_name] = nn.Parameter(
                torch.zeros(r1, r1), requires_grad=True
        )

        self.pblora_W2[adapter_name] = nn.Parameter(
                torch.zeros(obj_num, r2**2), requires_grad=True
        )

        self.scaling[adapter_name] = lora_alpha / r2

        self.reset_lora_parameters(adapter_name)

        self.set_adapter(self.active_adapters)

    def reset_lora_parameters(self, adapter_name):
        if adapter_name in self.pblora_A.keys():
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.pblora_A[adapter_name], a=math.sqrt(5))
                nn.init.zeros_(self.pblora_B[adapter_name])
                nn.init.kaiming_uniform_(self.pblora_W1[adapter_name], a=math.sqrt(5))
                nn.init.kaiming_uniform_(self.pblora_W2[adapter_name], a=math.sqrt(5))

    def _check_forward_args(self, x, *args, **kwargs):
        """Check if the arguments are compatible with the configs and state of the model"""
        adapter_names = kwargs.get("adapter_names", None)
        if adapter_names is None:
            return

        if len(x) != len(adapter_names):
            msg = (
                "Length of `adapter_names` should be the same as the number of inputs, but got "
                f"{len(adapter_names)} and {len(x)} respectively."
            )
            raise ValueError(msg)

        if self.merged:
            # It is unclear what would be the right thing to do if users pass adapter_names and there are merged
            # adapters. Therefore, it is better to raise an error in this case.
            msg = "Cannot pass `adapter_names` when there are merged adapters, please call `unmerge_adapter` first."
            raise ValueError(msg)


# Below code is based on https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
# and modified to work with PyTorch FSDP


#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------


class Linear(nn.Module, PBLoraLayer):
    # Lora implemented in a dense layer
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        r1: int = 0,
        r2: int = 0,
        obj_num: int = 0,
        pref_vec_init: list = None,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        **kwargs,
    ) -> None:
        super().__init__()
        PBLoraLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out

        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            r1=r1,
            r2=r2,
            obj_num=obj_num,
            pref_vec_init=pref_vec_init,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`list[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        for active_adapter in adapter_names:
            if active_adapter in self.pblora_A.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = base_layer.weight.data.clone()
                    delta_weight = self.get_delta_weight(active_adapter)
                    orig_weights = orig_weights + delta_weight

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = orig_weights
                else:
                    delta_weight = self.get_delta_weight(active_adapter)
                    base_layer.weight.data = base_layer.weight.data + delta_weight

                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.pblora_A.keys():
                weight = self.get_base_layer().weight
                delta_weight = self.get_delta_weight(active_adapter)
                weight.data -= delta_weight

    def _get_lora_matrices(self, adapter, cast_to_fp32=False) -> torch.Tensor:
        pblora_A = self.pblora_A[adapter]
        pblora_B = self.pblora_B[adapter]
        pblora_W1 = self.pblora_W1[adapter]
        pblora_W2 = self.pblora_W2[adapter]
        lora_pref_vec = self.pref_vec[adapter]
        lora_pref_vec = lora_pref_vec.to(pblora_A.dtype)
        obj_num = len(lora_pref_vec)

        r1, r2 = self.r1[adapter], self.r2[adapter] 

        # In case users wants to merge the adapter weights that are in
        # float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # float16 because the `@` and matmul operation in general is not supported in torch + cpu + fp16.

        if cast_to_fp32:
            pblora_A = pblora_A.float()
            pblora_B = pblora_B.float()
            pblora_W1 = pblora_W1.float()
            pblora_W2 = pblora_W2.float()
            lora_pref_vec = lora_pref_vec.float()

        W2 = (lora_pref_vec@pblora_W2).view(r2, r2)
        deltaw = pblora_B[:,:r1]@pblora_W1@pblora_A[:r1,:] + pblora_B[:,r1:]@W2@pblora_A[r1:,:]
        
        return deltaw

    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        device = self.pblora_B[adapter].device
        dtype = self.pblora_B[adapter].dtype
        cast_to_fp32 = device.type == "cpu" and dtype == torch.float16
        deltaw = self._get_lora_matrices(adapter, cast_to_fp32)
        output_tensor = transpose(deltaw, self.fan_in_fan_out) * self.scaling[adapter]
        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

        return output_tensor

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        self._check_forward_args(x, *args, **kwargs)
        # adapter_names = kwargs.pop("adapter_names", None)

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype
            for active_adapter in self.active_adapters:
                if active_adapter not in self.pblora_A.keys():
                    continue
                deltaw = self._get_lora_matrices(active_adapter).to(x.device)
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                x = x.to(self.pblora_A[active_adapter].dtype)

                result = result + F.linear(dropout(x), deltaw) * scaling

            result = result.to(torch_result_dtype)

        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "pblora." + rep
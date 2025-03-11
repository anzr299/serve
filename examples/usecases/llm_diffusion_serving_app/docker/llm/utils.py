import os
from typing import Dict

import openvino.torch
import time
import torch
import torch.fx
from optimum.modeling_base import OptimizedModel
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import GenerationConfig
from transformers import GenerationMixin
from transformers import PretrainedConfig
from transformers import PreTrainedModel
from transformers.cache_utils import StaticCacheConfig
from transformers.integrations.executorch import TorchExportableModuleWithStaticCache
from transformers.modeling_outputs import CausalLMOutputWithPast
from tabulate import tabulate
import datasets
import nncf
from nncf.torch.dynamic_graph.patch_pytorch import disable_patching

class FXAutoModelForCausalLM(OptimizedModel, GenerationMixin):
    def __init__(
        self,
        model: torch.fx.GraphModule,
        config: PretrainedConfig,
        device: str = "cpu",
        compile: bool = True,
        backend: str = None,
        dtype=torch.float32,
    ):
        super().__init__(model, config)
        self.generation_config = GenerationConfig.from_model_config(self.config)
        self.main_input_name = "input_ids"
        self.compile = compile
        self.backend = backend
        self._device = device.upper()
        self._dtype = dtype
        self._cached_prefill_input_ids = None
        self._cached_cache_position = None

        if self.compile:
            if backend is None or backend != "openvino":
                prefil_args = {"fullgraph": True, "dynamic": True}
                decode_one_token_args = {"fullgraph": True, "mode": "reduce-overhead"}
                if backend is not None:
                    prefil_args["backend"] = backend
                    decode_one_token_args["backend"] = backend
                self.prefill = torch.compile(self.model, **prefil_args)
                self.decode_one_token = torch.compile(self.model, **decode_one_token_args)
            else:
                self.prefill = None
                self.decode_one_token = None

    def get_openvino_backend_options(self) -> Dict:
        return {
            "device": self._device,
            "aot_autograd": True,
            # "disabled_ops": ["torch.ops.aten.copy_.default"]
        }

    def get_prefill(self, input_ids: torch.Tensor, cache_position: torch.Tensor):
        if (
            self.backend is not None
            and self.backend == "openvino"
            and (
                self.prefill is None
                or self._cached_prefill_input_ids.shape != input_ids.shape
                or self._cached_cache_position.shape != cache_position.shape
            )
        ):
            self._cached_prefill_input_ids = input_ids
            self._cached_cache_position = cache_position
            # self.prefill = openvino.torch.backend.openvino(
            #     self.model, (input_ids, cache_position), options=self.get_openvino_backend_options()
            # )
            self.prefill = self.model
            self.prefill.forward = torch.compile(
                self.prefill.forward,
                backend="openvino",
                options=self.get_openvino_backend_options(),
            )
        return self.prefill

    def get_decode_one_token(self, input_ids: torch.Tensor, cache_position: torch.Tensor):
        if self.backend is not None and self.backend == "openvino" and self.decode_one_token is None:
            # self.decode_one_token = openvino.torch.backend.openvino(
            #     self.model, (input_ids, cache_position), options=self.get_openvino_backend_options()
            # )
            self.decode_one_token = self.model
            self.decode_one_token.forward = torch.compile(
                self.decode_one_token.forward,
                backend="openvino",
                options=self.get_openvino_backend_options(),
            )
        return self.decode_one_token

    def infer_prefill(self, input_ids: torch.Tensor, cache_position: torch.Tensor):
        if self.compile:
            _ = self.get_prefill(input_ids, cache_position)(input_ids, cache_position)
        else:
            self.model(input_ids, cache_position)

    def infer_decode_one_token(self, input_ids: torch.Tensor, cache_position: torch.Tensor):
        if self.compile:
            _ = self.get_decode_one_token(input_ids, cache_position)(input_ids, cache_position)
        else:
            self.model(input_ids, cache_position)

    @property
    def device(self) -> torch.device:
        return torch.device(self._device.lower())

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        cache_position = kwargs["cache_position"]
        past_len = cache_position[0]
        if past_len < input_ids.shape[1]:
            input_ids = input_ids[:, past_len:]

        return {"input_ids": input_ids, "cache_position": cache_position}

    def _save_pretrained(self, save_directory):
        pass

    def forward(
        self,
        input_ids: torch.Tensor,
        cache_position: torch.Tensor,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        if self.compile:
            if input_ids.shape[1] == 1:
                logits = self.get_decode_one_token(input_ids, cache_position)(input_ids, cache_position)
            else:
                logits = self.get_prefill(input_ids, cache_position)(input_ids, cache_position)
        else:
            logits = self.model(input_ids, cache_position)

        return CausalLMOutputWithPast(logits=logits)

    def can_generate(self):
        return True

    def _supports_default_dynamic_cache(self) -> bool:
        return False


class TorchExportableModuleWithStaticCacheDynamicShape(TorchExportableModuleWithStaticCache):
    def forward(self, input_ids: torch.Tensor, cache_position: torch.Tensor):
        outs = self.model(
            input_ids=input_ids,
            position_ids=cache_position.unsqueeze(0),
            cache_position=cache_position,
            past_key_values=self.static_cache,
            use_cache=True,
        )
        return outs.logits


def convert_and_export_with_cache(model: PreTrainedModel, use_torch_export=True):
    """
    Convert a `PreTrainedModel` into an exportable module and export it using `torch.export`
    or `torch._export.capture_pre_autograd_graph`.
    """
    import torch.export._trace

    with torch.no_grad():
        example_input_ids = torch.ones(1, 8, dtype=torch.long)
        example_cache_position = torch.arange(0, 8, dtype=torch.long)
        model(example_input_ids)
        sequence_length = torch.export.Dim("sequence_length", min=1, max=128)
        dynamic_shapes = {"input_ids": {1: sequence_length}, "cache_position": {0: sequence_length}}

        exported_program = torch.export.export_for_training(
            TorchExportableModuleWithStaticCacheDynamicShape(model),
            args=(example_input_ids, example_cache_position),
            dynamic_shapes=dynamic_shapes
        ).run_decompositions(decomp_table={})
        return exported_program
import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfApi
from utils import convert_and_export_with_cache
from transformers.cache_utils import StaticCacheConfig
from nncf.torch import disable_patching
import nncf

def dir_path(path_str):
    try:
        if not os.path.isdir(path_str):
            os.makedirs(path_str)
            print(f"{path_str} did not exist, created the directory.")
            print("\nDownload will take few moments to start.. ")
        return path_str
    except Exception as e:
        raise NotADirectoryError(f"Failed to create directory {path_str}: {e}")


class HFModelNotFoundError(Exception):
    def __init__(self, model_str):
        super().__init__(f"HuggingFace model not found: '{model_str}'")


def hf_model(model_str):
    api = HfApi()
    models = [m.modelId for m in api.list_models()]
    if model_str in models:
        return model_str
    else:
        raise HFModelNotFoundError(model_str)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_path",
    "-o",
    type=dir_path,
    default="model",
    help="Output directory for downloaded model files",
)
parser.add_argument(
    "--model_name",
    "-m",
    required=True,
    help="HuggingFace model name",
    # type=hf_model,
)

args = parser.parse_args()

model = AutoModelForCausalLM.from_pretrained(
    args.model_name,
    torch_dtype=torch.float16,
    use_safetensors=True,
)
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
model.generation_config.cache_implementation = "static"
model.generation_config.cache_config = StaticCacheConfig(
    batch_size=1, max_cache_len=1024
)

model_config = model.config
generation_config = model.generation_config
with disable_patching():
    exported_model = convert_and_export_with_cache(model)
    graph_module = exported_model.module()
    graph_module = nncf.compress_weights(graph_module, mode=nncf.CompressWeightsMode.INT4_ASYM)
    exported_model = convert_and_export_with_cache(graph_module, re_export=True)
    model_config.save_pretrained(args.model_path)
    generation_config.save_pretrained(args.model_path)
    tokenizer.save_pretrained(args.model_path)
    torch.export.save(exported_model, f"{args.model_path}/exported_llama.pt2")

print(f"\nFiles for '{args.model_name}' are downloaded to '{args.model_path}'")

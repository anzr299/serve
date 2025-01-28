import os
import argparse
import torch
from diffusers import StableDiffusion3Pipeline, UNet2DConditionModel
from huggingface_hub import HfApi
from utils import save_fx_pipeline, export_models, init_pipeline, collect_calibration_data
import nncf
from nncf.quantization.range_estimator import RangeEstimatorParametersSet
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
    "--model_name", "-m", type=hf_model, required=True, help="HuggingFace model name"
)

args = parser.parse_args()

pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", text_encoder_3=None, tokenizer_3=None)
models_dict, configs_dict = export_models(pipe)

calibration_dataset = collect_calibration_data(pipe)

transformer = models_dict["transformer"]
models_dict['text_encoder'] = nncf.compress_weights(models_dict['text_encoder'])
models_dict['text_encoder_2'] = nncf.compress_weights(models_dict['text_encoder_2'])
models_dict['vae_encoder'] = nncf.compress_weights(models_dict['vae_encoder'])
models_dict['vae_decoder'] = nncf.compress_weights(models_dict['vae_decoder'])
quantized_transformer = nncf.quantize(transformer, calibration_dataset=nncf.Dataset(calibration_dataset),
                    subset_size=len(calibration_dataset),
                    model_type=nncf.ModelType.TRANSFORMER,
                    advanced_parameters=nncf.AdvancedQuantizationParameters(smooth_quant_alpha=0.7, 
                                                                            disable_bias_correction=True, 
                                                                            weights_range_estimator_params=RangeEstimatorParametersSet.MINMAX,
                                                                            activations_range_estimator_params=RangeEstimatorParametersSet.MINMAX,))
fx_pipe = init_pipeline(models_dict, configs_dict)
save_fx_pipeline(fx_pipe, args.model_path)

print(f"\nFiles for '{args.model_name}' is downloaded to '{args.model_path}'")

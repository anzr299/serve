import argparse
import os
import gc
import torch
from pathlib import Path
from sd.utils import load_fx_pipeline

def run_inference(prompt: str, save_path: str, pipeline_path: str):
    loaded_pipe = load_fx_pipeline(pipeline_path)
    loaded_pipe.transformer = torch.compile(loaded_pipe.transformer, backend="openvino")
    loaded_pipe.text_encoder = torch.compile(loaded_pipe.text_encoder, backend="openvino")
    loaded_pipe.text_encoder_2 = torch.compile(loaded_pipe.text_encoder_2, backend="openvino")
    loaded_pipe.vae.decoder = torch.compile(loaded_pipe.vae.decoder, backend="openvino")
    print(f"Compiling Pipeline")
    loaded_pipe("sample", num_inference_steps=1)
    print(f"Running inference on prompt: {prompt}")
    image = loaded_pipe(prompt).images[0]
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    image.save(save_path)
    print(f"Image saved at: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on a prompt and save the generated image.")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for image generation.")
    parser.add_argument("--pipeline_path", type=str, default="sd3-fx/", help="Path to the saved FX pipeline.")
    parser.add_argument("--save_path", type=str, default="output/generated_image.png", help="Path to save the generated image. Default is 'output/'.")
    args = parser.parse_args()
    run_inference(args.prompt, args.save_path, args.pipeline_path)
import os
import re
import json
import logging
import time
import torch
import openvino.torch  # noqa: F401  # Import to enable optimizations from OpenVINO
from transformers import AutoModelForCausalLM, AutoTokenizer

from ts.handler_utils.timer import timed
from ts.torch_handler.base_handler import BaseHandler
from utils import PretrainedConfig, GenerationConfig, FXAutoModelForCausalLM

logger = logging.getLogger(__name__)


class LlmHandler(BaseHandler):
    def __init__(self):
        super().__init__()

        self.model = None
        self.tokenizer = None
        self.context = None
        self.initialized = False
        self.device = "cpu"
        self.prompt_length = 0
        self.stream = False
        self.user_prompt = []
        self.prompt_template = ""

    def initialize(self, ctx):
        self.context = ctx
        self.manifest = ctx.manifest

        model_store_dir = ctx.model_yaml_config["handler"]["model_store_dir"]
        model_name_llm = os.environ["MODEL_NAME_LLM"].replace("/", "---")
        model_dir = os.path.join(model_store_dir, model_name_llm, "model")

        self.device = ctx.model_yaml_config["deviceType"]
        self.stream = ctx.model_yaml_config["handler"].get("stream", True)

        logger.info(f"ctx.model_yaml_config is {ctx.model_yaml_config}")
        logger.info(f"ctx.system_properties is {ctx.system_properties}")
        logger.info(f"Using device={self.device}")

        # Load tokenizer and model
        logger.info(f"Loading model {model_dir}...")
        t0 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model_config = PretrainedConfig.from_pretrained(model_dir)
        self.generation_config = GenerationConfig.from_pretrained(model_dir)
        exported_model = torch.export.load(f"{model_dir}/exported_llama.pt2")
        graph_module = exported_model.module()
        self.model = FXAutoModelForCausalLM(graph_module, model_config, compile=compile, backend='openvino')

        # Get backend for model-confil.yaml. Defaults to "openvino"
        compile_options = {}
        pt2_config = ctx.model_yaml_config.get("pt2", {})
        compile_options = {
            "backend": pt2_config.get("backend", "openvino"),
            "options": pt2_config.get("options", {}),
        }
        logger.info(f"Loading LLM model with PT2 compiler options: {compile_options}")

        self.model.to(self.device)

        logger.info(f"Time to load {model_dir}: {time.time() - t0:.02f} seconds")
        self.initialized = True

    @timed
    def preprocess(self, requests):
        assert len(requests) == 1, "Llama currently only supported with batch_size=1"

        req_data = requests[0]
        input_data = req_data.get("data") or req_data.get("body")

        if isinstance(input_data, (bytes, bytearray)):
            input_data = input_data.decode("utf-8")

        if isinstance(input_data, str):
            input_data = json.loads(input_data)

        self.user_prompt = input_data["user_prompt"]
        self.prompt_template = input_data["prompt_template"]
        encoded_prompt = self.tokenizer(self.prompt_template, return_tensors="pt")["input_ids"].to(
            self.device
        )

        input_data["encoded_prompt"] = encoded_prompt

        return input_data

    @timed
    def inference(self, input_data):
        generated_text = " "
        try:
            self.generation_config.do_sample=False
            self.generation_config.max_new_tokens=input_data["max_new_tokens"]
            self.generation_config.temperature=None
            self.generation_config.top_p=None
            self.generation_config.repetition_penalty=1.2
            self.generation_config.pad_token_id=self.tokenizer.pad_token_id
            self.generation_config.eos_token_id=self.tokenizer.eos_token_id

            with torch.no_grad():
                outputs = self.model.generate(
                    input_data["encoded_prompt"],
                    generation_config=self.generation_config
                )

            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            logger.error(f"An error occurred during LLM inference: {e}")

        return generated_text

    def postprocess(self, generated_text):
        logger.info(f"LLM Generated Output: {generated_text}")
        # Initialize with user prompt
        prompt_list = [self.user_prompt]
        try:
            logger.info("Parsing LLM Generated Output to extract prompts within []...")
            response_match = re.search(r"\[(.*?)\]", generated_text)
            # Extract the result if match is found
            if response_match:
                # Split the extracted string by semicolon and strip any leading/trailing spaces
                response_list = response_match.group(1)
                extracted_prompts = [item.strip() for item in response_list.split(";")]
                prompt_list.extend(extracted_prompts)
            else:
                logger.warning("No match found in the generated output text !!!")
        except Exception as e:
            logger.error(f"An error occurred while parsing the generated text: {e}")

        logger.info(f"Extracted prompt list: {prompt_list}")

        return [prompt_list]

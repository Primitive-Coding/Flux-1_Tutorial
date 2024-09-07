import os
from dotenv import load_dotenv

load_dotenv()

import torch

from huggingface_hub import login

login(token=os.getenv("API_KEY"))


from diffusers import FluxPipeline

# "black-forest-labs/FLUX.1-schnell"


class FluxModel:
    def __init__(self, name: str, save_path: str, model: str) -> None:
        self.name = name
        self.save_path = save_path
        self.model = model

    def load_model(self):
        pipeline = FluxPipeline.from_pretrained(
            self.save_path, torch_dtype=torch.float16
        )
        return pipeline

    def save_model(self):
        pipeline = FluxPipeline.from_pretrained(self.name, torch_dtype=torch.float16)
        pipeline.save_pretrained(self.save_path)

    def get_image(self, prompt: str, file_name: str):
        pipeline = self.load_model()
        pipeline.enable_sequential_cpu_offload()
        image = pipeline(
            prompt,
            guidance_scale=0.0,
            output_type="pil",
            num_inference_steps=15,
            max_sequence_length=512,
            generator=torch.Generator("cpu").manual_seed(0),
        ).images[0]
        image.save(f"./Images/{file_name}.png")

import torch
import random
import numpy as np
from PIL import Image
from diffusers.utils import load_image
import yaml

# Import parts of your original code
from diffusers.utils.torch_utils import randn_tensor
from transformers import CLIPVisionModelWithProjection

# These functions are assumed to be available in your project:
from src.config import RunConfig
from src.enums import Model_Type, Scheduler_Type
from src.utils.enums_utils import get_pipes, is_stochastic, model_type_to_size
from src.utils.images_utils import crop_center_square_and_resize
from editing_utils import setup_env, edit_demo

class TightInversionPipeline:
    """Pipeline for editing images using Tight Inversion."""
    
    def __init__(self, config_path="configs/run_configs/ddim/ddim_ipa_scale_0.4.yaml"):
        """Initialize the pipeline with models and config."""
        self.weight_name = "ip-adapter-plus_sdxl_vit-h.safetensors"
        
        # Set up environment once
        self.cfg, self.pipe_inversion, self.pipe_inference, self.device, self.dtype, self.seed = setup_env(config_path)
        
        # Load image encoder if needed
        self.image_encoder = None
        if self.cfg.inference_use_ipa:
            self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                "h94/IP-Adapter",
                subfolder="models/image_encoder",
                torch_dtype=self.dtype,
            ).to(self.device)
    
    def edit_image(self, img, source_prompt, edit_prompt, ipa_scale=0.4, 
                   guidance_scale=7.5, sharpening_factor=1.0, use_negative_prompt=False):
        """Edit an image using Tight Inversion."""
        result_image = edit_demo(
            img,
            source_prompt, 
            edit_prompt, 
            ipa_scale, 
            guidance_scale, 
            sharpening_factor, 
            use_negative_prompt,
            cfg=self.cfg, 
            pipe_inversion=self.pipe_inversion,
            pipe_inference=self.pipe_inference,
            weight_name = self.weight_name,
            device=self.device,
            seed = self.seed
        )
        
        # Save or display the result
        result_image.save("output_edited_image.jpg")
        
        return result_image

tight_inversion = TightInversionPipeline()
source_img = Image.open("sophia.png")
source_prompt = "a girl in a light pink dress with a funny expression"
edit_prompt = "a girl with a cartoon thought bubble over her head"
ipa_scale = 0.4
guidance_scale = 15
sharpening_factor = 1.0
use_negative_prompt = False
res = tight_inversion.edit_image(source_img, source_prompt, edit_prompt, ipa_scale, guidance_scale, sharpening_factor, use_negative_prompt)

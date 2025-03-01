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

# Use the same weight name as in your code
weight_name = "ip-adapter-plus_sdxl_vit-h.safetensors"

# A helper to create noise (duplicated from your code)
def create_noise_list(model_type, length, dtype, generator=None):
    img_size = model_type_to_size(model_type)
    VQAE_SCALE = 8
    latents_size = (1, 4, img_size[0] // VQAE_SCALE, img_size[1] // VQAE_SCALE)
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    return [randn_tensor(latents_size, dtype=dtype, device=device, generator=generator) for i in range(length)]

config_path = "configs/run_configs/ddim/ddim_ipa_scale_0.4.yaml"
with open(config_path, 'r') as file:
    cfg = RunConfig.from_yaml(yaml.safe_load(file))

# Set device and seed
device = "cuda" if torch.cuda.is_available() else "cpu"
seed = cfg.seed if cfg.seed is not None else 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

# Set torch dtype based on config
dtype = torch.float32 if cfg.use_float32 else torch.bfloat16

# Load image encoder if using IP-Adapter
image_encoder = None
if cfg.inference_use_ipa:
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        "h94/IP-Adapter",
        subfolder="models/image_encoder",
        torch_dtype=dtype,
    ).to(device)

# Get both inversion and inference (editing) pipelines.
pipe_inversion, pipe_inference = get_pipes(cfg, image_encoder=image_encoder, device=device)

def edit_demo(source_image, source_prompt, edit_prompt, ipa_scale, guidance_scale, sharpening_factor, use_negative_prompt):
    """
    Given a source image, a source prompt and an edit prompt, this function:
      1. Preprocesses the image.
      2. Uses the inversion pipeline (with the source prompt) to compute image latents
         using an IPA scale set by the user.
      3. Uses the editing pipeline (with the edit prompt) to generate the edited image
         using the user-provided guidance scale (while inversion guidance scale is fixed to 1)
         and applies a sharpening factor.
    """
    if source_image is None:
        return None

    pipe_inversion, pipe_inference = get_pipes(cfg, image_encoder=image_encoder, device=device)
    pipe_inversion.cfg = cfg
    pipe_inference.cfg = cfg

    # Preprocess the source image: convert to RGB and crop/resize
    input_image = source_image.convert("RGB")
    model_size = model_type_to_size(cfg.model_type)  # e.g. (512, 512)
    if input_image.size == (500, 500):
        # add noise to the image (magicbrush)
        input_image = input_image.resize(model_size)
        # # Convert image to numpy array
        image_array = np.array(input_image)

        # Define Gaussian noise parameters
        mean = 0
        stddev = 17.5  # Standard deviation (adjust for more/less noise)

        # Generate Gaussian noise
        gaussian_noise = np.random.normal(mean, stddev, image_array.shape).astype(np.float32)

        # Add noise to the image
        noisy_image_array = np.clip(image_array + gaussian_noise, 0, 255).astype(np.uint8)

        # Convert back to PIL Image
        noisy_image = Image.fromarray(noisy_image_array)
        input_image = noisy_image
    else:
        input_image = crop_center_square_and_resize(input_image, model_size)

    # Prepare IP adapter arguments using the user-specified IPA scale
    editing_ipa_args = {}
    inversion_ipa_args = {}
    reconstruction_ipa_args = {}
    if cfg.inference_use_ipa or cfg.inversion_use_ipa:
        # pipe_inversion.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name=weight_name)
        pipe_inference.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name=weight_name)
        pipe_inference.set_ip_adapter_scale(cfg.inference_ipa_scale)
        print("Preparing image embeds...")
        image_embeds = pipe_inference.prepare_ip_adapter_image_embeds(
            ip_adapter_image=input_image,
            ip_adapter_image_embeds=None,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
        )
        if cfg.use_image_embeds_for_null_prompt:
            image_embeds = [torch.stack([image_embeds[0][1], image_embeds[0][1]])]
        inference_image_embeds = image_embeds
        inversion_image_embeds = image_embeds
        reconstruction_image_embeds = image_embeds
        if cfg.guidance_scale == 1.0 or cfg.guidance_scale == 0.0:
            inference_image_embeds = [image_embeds[0][None, 1]]
        if cfg.inversion_guidance_scale == 1.0 or cfg.inversion_guidance_scale == 0.0:
            inversion_image_embeds = [image_embeds[0][None, 1]]
        if cfg.reconstruction_guidance_scale == 1.0 or cfg.reconstruction_guidance_scale == 0.0:
            reconstruction_image_embeds = [image_embeds[0][None, 1]]
        if cfg.inference_use_ipa:
            editing_ipa_args = {"ip_adapter_image_embeds": inference_image_embeds}
            reconstruction_ipa_args = {"ip_adapter_image_embeds": reconstruction_image_embeds}
        if cfg.inversion_use_ipa:
            inversion_ipa_args = {"ip_adapter_image_embeds": inversion_image_embeds}
        pipe_inference.unload_ip_adapter()

    # Create a generator with the seed
    generator = torch.Generator(device=device).manual_seed(seed)
    if is_stochastic(cfg.scheduler_type):
        noise = create_noise_list(cfg.model_type, cfg.num_inversion_steps, dtype, generator=generator)
        pipe_inversion.scheduler.set_noise_list(noise)
        pipe_inference.scheduler.set_noise_list(noise)

    # --- Inversion Stage ---
    # Use the source prompt to invert the image into latents.
    print("Performing inversion...")
    if cfg.inversion_use_ipa:
        pipe_inversion.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name=weight_name)
        pipe_inversion.set_ip_adapter_scale(ipa_scale)  # user-specified IPA scale
    inversion_result = pipe_inversion(
        prompt=source_prompt if source_prompt != "" else None,
        negative_prompt=cfg.negative_prompt,
        num_inversion_steps=cfg.num_inversion_steps,
        num_inference_steps=cfg.num_inversion_steps,
        generator=generator,
        image=input_image,
        guidance_scale=1.0,  # inversion guidance scale is fixed to 1
        use_cfgpp=cfg.use_cfgpp_inversion,
        strength=cfg.inversion_max_step,
        denoising_start=1.0 - cfg.inversion_max_step,
        num_gd_steps=cfg.num_gd_steps,
        gd_step_size=cfg.gd_step_size,
        optimization_start=cfg.optimization_start,
        normalize=cfg.normalize,
        **inversion_ipa_args
    )
    # The inversion result returns a tuple; we take the first latent.
    inv_latent = inversion_result[0][0]

    # --- Editing Stage ---
    print("Performing editing...")
    negative_prompt = cfg.negative_prompt if not use_negative_prompt else source_prompt
    if cfg.inference_use_ipa:
        pipe_inference.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name=weight_name)
        pipe_inference.set_ip_adapter_scale(ipa_scale)  # user-specified IPA scale
    edited_image = pipe_inference(
        image=inv_latent,
        prompt=edit_prompt if edit_prompt != "" else None,
        negative_prompt=negative_prompt if negative_prompt != "" else None,
        denoising_start=1.0 - cfg.inversion_max_step,
        strength=cfg.inversion_max_step,
        num_inference_steps=cfg.num_inference_steps,
        guidance_scale=guidance_scale,  # use the user-specified editing guidance scale
        guidance_rescale=cfg.guidance_rescale,
        use_cfgpp=cfg.use_cfgpp_inference,
        sharpening_factor=sharpening_factor,
        **editing_ipa_args
    ).images[0]

    pipe_inference.unload_ip_adapter()
    pipe_inversion.unload_ip_adapter()

    return edited_image

# Your custom inputs
source_image = Image.open("juliana.jpg")
source_prompt = "a girl in a red coat looks into the distance"
edit_prompt = "a girl with a red hat"
ipa_scale = 0.4
guidance_scale = 7.5
sharpening_factor = 1.0
use_negative_prompt = False

# Call the edit function directly
result_image = edit_demo(
    source_image, 
    source_prompt, 
    edit_prompt, 
    ipa_scale, 
    guidance_scale, 
    sharpening_factor, 
    use_negative_prompt
)

# Save or display the result
result_image.save("output_edited_image.jpg")
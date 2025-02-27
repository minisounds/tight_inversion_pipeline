import spaces
import gradio as gr
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
# pipe_inversion.cfg = cfg
# pipe_inference.cfg = cfg

@spaces.GPU
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

# Build Gradio Interface with additional inputs for IPA scale, guidance scale, and sharpening factor.
demo = gr.Interface(
    fn=edit_demo,
    inputs=[
        gr.Image(type="pil", label="Source Image"),
        gr.Textbox(lines=2, placeholder="Enter source prompt here", label="Source Prompt"),
        gr.Textbox(lines=2, placeholder="Enter edit prompt here", label="Edit Prompt"),
        gr.Slider(minimum=0.1, maximum=1.0, step=0.05, value=0.4, label="IPA Scale", info="Adjust to balance faithfulness and editability"),
        gr.Slider(minimum=1.0, maximum=20.0, step=0.1, value=7.5, label="Guidance Scale", info="Adjust to control the strength of the edit"),
        gr.Slider(minimum=1.0, maximum=3.0, step=0.1, value=1.5, label="Sharpening Factor", info="Makes CFG effects more localized"),
        gr.Checkbox(label="Use negative prompt", value=False, info="Use the source prompt as a negative prompt for editing"),
    ],
    outputs=gr.Image(type="pil", label="Edited Image"),
    title="Tight Inversion SDXL Demo",
    description=(
        "Upload an image, provide a source prompt (for inversion) and an edit prompt, "
        "set the IPA scale (for both inversion and editing), the guidance scale for editing "
        "and the guidance scale sharpening factor, "
        "then view the edited image. You can start with the provided examples."
    ),
    examples=[
        # Example template:
        # ["path/to/example_image.jpg", "A sunny landscape", "A stormy night", 0.4, 7.5, 1.5],
        # Add more examples here.
        ["example_images/animals/7.jpg", "A photo of a husky", "A photo of a cat", 0.4, 7.5, 1.0, False],
        ["example_images/tests/diner_square.png", "a photo of people dining in a diner", "a photo of robots dining in a diner", 0.4, 7.5, 1.0, True],
        ["example_images/tests/lion.jpg", "a lion in the field", "a lion made of lego in the field", 0.4, 7.5, 1.0, True],
        ["example_images/garibis_images/5.jpg", "metal elephant statues", "real living elephants", 0.3, 4.0, 1.0, True],
        # ["example_images/tests/monkey.jpg", "a photo of a monkey sitting on a branch in the forest", "a photo of a beaver sitting on a branch in the forest", 0.7, 12, 1.5, False],
        ["example_images/editAR_images/dog_forest.jpg", "a dog running in the forest", "a forest with no one around", 0.4, 7.5, 1.0, False],
        ["example_images/animals/0.jpg", "A photo of a gazelle", "A photo of a gazelle wearing a red hat", 0.4, 7.5, 1.5, False],
        # ["example_images/garibis_images/1.jpg", "white christmas tree", "christmas tree with red ornaments", 0.55, 7.5, 2.0, False],
        ["example_images/people/5.jpg", "a person", "a person with a large thick beard", 0.5, 7.5, 2.0, True],
        ["example_images/magic_brush/262283-input.png", "a lot of vases filled with lots of flowers", "a lot of vases filled with red roses", 0.4, 12, 1.4, True],
    ]
)

if __name__ == "__main__":
    demo.launch()
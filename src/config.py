from dataclasses import dataclass, field
from typing import Any, Dict, List

from src.enums import Model_Type, Scheduler_Type

@dataclass
class LEDITSConfig:
    inversion_skip: float = 0.2
    edit_threshold: float = 0.6
    edit_friendly: bool = False

    @classmethod
    def from_yaml(cls, config):
        return cls(**config)

    def __post_init__(self):
        pass
@dataclass
class RFInversionConfig:
    gamma: float = 0.5
    reconstruction_eta: float = 0.9
    editing_eta: float = 0.9
    reconstruction_start_timestep: float = 0.0
    reconstruction_stop_timestep: float = 0.0
    editing_start_timestep: float = 0.0
    editing_stop_timestep: float = 0.0
    nudge_factor: float = 1.0


    @classmethod
    def from_yaml(cls, config):
        return cls(**config)

    def __post_init__(self):
        pass

@dataclass
class RenoiseConfig:
    max_num_renoise_steps_first_step: int = 5

    num_renoise_steps: int = 9

    renoise_first_step_max_timestep: int = 250

    inversion_max_step: float = 1.0

    # Average Parameters

    average_latent_estimations: bool = True

    average_first_step_range: tuple = (0, 5)

    average_step_range: tuple = (8, 10)

    # Noise Regularization

    noise_regularization_lambda_ac: float = 20.0

    noise_regularization_lambda_kl: float = 0.065
    
    noise_regularization_num_reg_steps: int = 4

    noise_regularization_num_ac_rolls: int = 5

    # Noise Correction

    perform_noise_correction: bool = True

    @classmethod
    def from_yaml(cls, config):
        return cls(**config)

    def __post_init__(self):
        pass

@dataclass
class RunConfig:
    method: str = "ddim_inversion"
    
    use_wandb: bool = False

    model_type : Model_Type = Model_Type.SDXL

    scheduler_type : Scheduler_Type = Scheduler_Type.DDIM

    seed: int = 7865

    num_inference_steps: int = 50

    num_inference_steps_random_image: int = 50

    num_inversion_steps: int = 50

    inversion_max_step: float = 1.0

    inversion_guidance_scale: float = 1.0

    guidance_scale: float = 1.0

    use_cfgpp_inference: bool = False

    use_cfgpp_inversion: bool = False

    reconstruction_guidance_scale: float = 1.0

    random_image_guidance_scale: float = 1.0

    perform_inversion: bool = True

    inversion_use_ipa: bool = False

    inference_use_ipa: bool = False
    
    inference_ipa_scale: float = 0.3

    inversion_ipa_scale: float = 0.3

    saturation_removal_ipa_scale: float = 0.3

    num_gd_steps: int = 0
    
    gd_step_size: float = 0.0

    optimization_start: int = 0

    normalize: bool = False

    random_inference_times: int = 1

    negative_prompt: str = None

    remove_cfg_saturation: bool = False

    renoise: bool = False

    renoise_config: RenoiseConfig = None

    use_empty_inversion_prompt: bool = False
    
    use_description_as_negative_prompt: bool = False

    rf_config: RFInversionConfig = None

    ledits_config: LEDITSConfig = None

    guidance_rescale: float = 0.0

    sharpening_factor: float = 0.0

    use_image_embeds_for_null_prompt: bool = False

    use_float32: bool = False

    override_edit_prompts: List[str] = None

    vae_encode_decode_test: bool = False

    quantize: bool = False

    @classmethod
    def from_yaml(cls, config):
        config['model_type'] = Model_Type[config['model_type']]
        config['scheduler_type'] = Scheduler_Type[config['scheduler_type']]
        if 'renoise_config' in config:
            config['renoise_config'] = RenoiseConfig.from_yaml(config['renoise_config'])
        if 'rf_config' in config:
            config['rf_config'] = RFInversionConfig.from_yaml(config['rf_config'])
        if 'ledits_config' in config:
            config['ledits_config'] = LEDITSConfig.from_yaml(config['ledits_config'])
        return cls(**config)

    def __post_init__(self):
        pass

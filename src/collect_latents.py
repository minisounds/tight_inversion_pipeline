from pathlib import Path

import numpy as np
import pandas as pd

class LatentsCollector:
    def __init__(self, out_path, num_steps=50, inversion: bool = False, save_latents=False):
        self.save_latents = save_latents
        self.out_path = Path(out_path)
        self.out_path_latents = self.out_path / "latents"
        self.out_path_latents.mkdir(exist_ok=True, parents=True)
        self.last_latents = None
        self.results = []
        self.num_steps = num_steps
        self.inversion = inversion
    
    def add_latents(self, latents, step_index, recon_diff=None):
        self.last_latents = latents
        if not self.save_latents:
            return
        latents_np = latents.detach().cpu().float().numpy()
        latents_path = self.out_path_latents / f"latents_{step_index:04d}.npy"
        np.save(latents_path, latents_np)
        if recon_diff is not None:
            diff_np = recon_diff.detach().cpu().float().numpy()
            diff_path = self.out_path_latents / f"diff_{step_index:04d}.npy"
            np.save(diff_path, diff_np)
        self.results.append({"timestep": step_index, "latents_path": latents_path.name, "recon_diff_path": diff_path.name if recon_diff is not None else None})
        
    def __call__(self, pipe, step_index, timestep, callback_kwargs):
        latents = callback_kwargs["latents"]
        recon_diff = callback_kwargs.get("recon_diff", None)
        if self.inversion:
            step_index = step_index + 1
        else:
            step_index = self.num_steps - step_index - 1
            
        self.add_latents(latents, step_index, recon_diff)
        return callback_kwargs

    def save_results(self):
        df = pd.DataFrame(self.results)
        df.to_csv(self.out_path / "results.csv")

    def get_results(self):
        return self.results

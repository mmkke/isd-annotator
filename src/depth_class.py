"""
Depth Anything V2 Model Loader
Michael Massone
Created: 2025/5/15
Updated: 2025/05/22
"""


import torch
from depth_anything_v2.dpt import DepthAnythingV2

class DepthAnythingLoader:
    def __init__(self, encoder='vits', checkpoint_dir=None, device=None):
        """
        Initializes the DepthAnythingV2 model with the specified encoder and loads pretrained weights.

        Args:
            encoder (str): One of {'vits', 'vitb', 'vitl', 'vitg'}.
            checkpoint_dir (str): Path to directory containing the .pth model file.
            device (str or torch.device): Torch device to load the model onto.
        """
        self.encoder = encoder
        self.checkpoint_dir = checkpoint_dir or './checkpoints'
        self.device = device or ('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

        self.model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }

        self.model = self._load_model()

    def _load_model(self):
        config = self.model_configs[self.encoder]
        model = DepthAnythingV2(**config)
        ckpt_path = f"{self.checkpoint_dir}/depth_anything_v2_{self.encoder}.pth"
        state_dict = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(state_dict)
        return model.to(self.device).eval()

    def get_model(self):
        return self.model


# encoder = 'vits'
# checkpoint_dir = 'Depth-Anything-V2/checkpoints'
# model_loader = DepthAnythingLoader(encoder=encoder, checkpoint_dir=checkpoint_dir)
# model = model_loader.get_model()
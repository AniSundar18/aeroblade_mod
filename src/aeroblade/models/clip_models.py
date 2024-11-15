from .clip import clip 
from PIL import Image
import torch.nn as nn


CHANNELS = {
    "RN50" : 1024,
    "ViT-L/14" : 768
}

class CLIPModel(nn.Module):
    def __init__(self, name, num_classes=1, layer=None, spatial=False):
        super(CLIPModel, self).__init__()
        self.layer = layer
        self.spatial = spatial
        self.model, self.preprocess = clip.load(name, device="cpu") # self.preprecess will not be used during training, which is handled in Dataset class 
        self.fc = nn.Linear( CHANNELS[name], num_classes )
 

    def forward(self, x, return_feature=False):
        if self.layer is not None:
            out = self.model.encode_image(x, self.layer, spatial = self.spatial)
            return out
        else:
            features = self.model.encode_image(x) 
        if return_feature:
            return features
        return self.fc(features)


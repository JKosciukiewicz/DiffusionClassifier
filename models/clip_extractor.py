import torch
import torch.nn as nn
import clip
import torchvision.transforms as T

class CLIPExtractor(nn.Module):
    def __init__(self, model_name="ViT-B/32", device=None):
        super().__init__()
        # If device is not provided, we use a sensible default for initialization.
        # Lightning will move the module (and thus this model) to the correct device later.
        if device is None:
            if torch.cuda.is_available():
                init_device = "cuda"
            elif torch.backends.mps.is_available():
                init_device = "mps"
            else:
                init_device = "cpu"
        else:
            init_device = device
            
        self.model, self.preprocess = clip.load(model_name, device=init_device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
            
        # CLIP's preprocess handles normalization and resizing for PIL images.
        # Since we receive tensors, we implement the equivalent transforms here.
        self.resize = T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC)
        self.normalize = T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

    def extract_features(self, x):
        # Ensure model is on the same device as input
        self.model.to(x.device)
        
        # x is (batch_size, channels, H, W)
        # Convert grayscale to RGB if necessary
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        elif x.shape[1] > 3:
            x = x[:, :3, :, :]
            
        # Resize and normalize
        x = self.resize(x)
        # Ensure x is in [0, 1] if it's not already. 
        # Most dataloaders return [0, 1] after ToTensor()
        x = self.normalize(x)
        
        with torch.no_grad():
            features = self.model.encode_image(x)
        return features.float()

    def forward(self, x):
        return self.extract_features(x)

    @property
    def embedding_dim(self):
        return self.model.visual.output_dim

from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from transformers.image_processing_utils import BaseImageProcessor, BatchFeature
from typing import Dict, List, Optional, Union
import torch

class GOTImageProcessor(BaseImageProcessor):
    def __init__(
            self, 
            image_size=384, 
            mean=None, 
            std=None,
            max_patches = 256,
            **kwargs,
            ):
        super().__init__(**kwargs)
        self.image_size=  image_size
        self.mean = (0.48145466, 0.4578275, 0.40821073)
        self.std = (0.26862954, 0.26130258, 0.27577711)
        self.max_patches = max_patches

    def preprocess(
        self,
        images,
        return_tensors: Optional[Union[str]] = None,
        **kwargs,
    ) :
        transform = transforms.Compose(
            [
                transforms.Resize(
                    (self.image_size, self.image_size), interpolation=InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )

        images = torch.stack([transform(image) for image in images])

        return BatchFeature(data={"pixel_values": images}, tensor_type=return_tensors)
from typing import Dict, List, Optional, Union
import numpy as np
from transformers.image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
import PIL
import torch

def build_transform(input_size, mean, std):
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ])
    return transform


class InternViTImageProcessor(BaseImageProcessor):
    model_input_names = ["pixel_values"]
    def __init__(
        self,
        size: Dict[str, int] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.size = size
        self.image_mean = image_mean if image_mean is not None else IMAGENET_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_STD

    def preprocess(
        self,
        images,
        **kwargs,
    ) -> PIL.Image.Image:
        transform = build_transform(self.size, self.image_mean,self.image_std)
        images = [transform(image) for image in images]
        data = {"pixel_values": torch.stack(images)}
        return BatchFeature(data=data)
import os
from typing import Union, Dict, List

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
import pdb

from transformers import Pix2StructVisionConfig
from transformers import CLIPVisionConfig
from chimera.model.kosmos2_5.modeling_kosmos2_5 import Kosmos2_5VisionConfig
from chimera.model.chimera import InternVisionConfig
from chimera.model.got import GotVisionConfig
import copy

logger = logging.get_logger(__name__)


def init_config_from_meta_config(meta, model_type):
    # 避免AutoXXX出现问题，此处显式指定初始化的模型类型
    
    if model_type == 'pix2struct_vision_model':
        return Pix2StructVisionConfig.from_pretrained(meta['model_name_or_path'])
    elif model_type == "kosmos_2_5_vision_model":
        return Kosmos2_5VisionConfig.from_pretrained(meta['model_name_or_path'])
    elif model_type == 'clip_vision_model':
        return CLIPVisionConfig.from_pretrained(meta['model_name_or_path'])
    elif model_type == 'intern_vit_6b':
        return InternVisionConfig.from_pretrained(meta['model_name_or_path'])
    elif model_type == "got_vision_model":
        return GotVisionConfig.from_pretrained(meta['model_name_or_path'])
    else:
        raise NotImplementedError(f'{model_type} is not implemented.')


def init_config_from_dict(dict):
    # 避免AutoXXX出现问题，此处显式指定初始化的模型类型
    if dict['model_type'] == 'pix2struct_vision_model':
        return Pix2StructVisionConfig(**dict)
    elif dict['model_type'] == "kosmos_2_5_vision_model":
        return Kosmos2_5VisionConfig(**dict)
    elif dict['model_type'] == 'clip_vision_model':
        return CLIPVisionConfig(**dict)
    elif dict['model_type'] == 'intern_vit_6b':
        return InternVisionConfig(**dict)
    elif dict['model_type'] == "got_vision_model":
        return GotVisionConfig(**dict)
    else:
        raise NotImplementedError(f'{dict["model_type"]} is not implemented.')




class ExpertEncoderConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ExpertEncoder`]. It is used to
    instantiate a set of vision encoders according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        encoder_config_dicts (`list`):
            A list of dicts of encoders, each dict contains key of "domain", "select_layer" and "config". 
            e.g.,[
            {"domain":"chart", "config": config.to_dict(), "select_layer":-2},
            {"domain":"math", "config": config.to_dict(), "select_layer":-1}
            ]
        llm_hidden_size (`int`, defaults to 4096):
            hidden state dim of the llm.
    """

    model_type = 'expert_encoder'

    def __init__(
            self,
            # encoder_config_dicts: List[Dict] = [],
            config_list: List[Dict] = [],
            llm_hidden_size = 4096,
            **kwargs,
    ):
        super().__init__(**kwargs)
        # 实际拥有的encoder数量，不算index=0对应的
        self.num_encoder = len(config_list)
        if self.num_encoder==0:
            logger.info(
                f"No encoder is added into the ExpertEncoder."
            )
        self.llm_hidden_size = llm_hidden_size
        
        self.config_list = config_list
        
        for i, cur_dict in enumerate(self.config_list):
            assert i == cur_dict['index'], f"Mismatch: get index {cur_dict['index']} from meta file but in the position index of {i}."
            cur_dict['config'] = init_config_from_dict(cur_dict['config'])
        self.separate_load = False

                    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> 'PretrainedConfig':
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        
        if 'expert_encoder_config' in config_dict:  
            config_dict = config_dict['expert_encoder_config']
        

        if 'model_type' in config_dict and hasattr(cls, 'model_type') and config_dict['model_type'] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f'{cls.model_type}. This is not supported for all configurations of models and can yield errors.'
            )

        return cls.from_dict(config_dict, **kwargs)
    
    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output['num_encoder'] = self.num_encoder
        output['llm_hidden_size'] = self.llm_hidden_size
        for i in output['config_list'] :
            i['config'] = i['config'].to_dict()
        

        return output

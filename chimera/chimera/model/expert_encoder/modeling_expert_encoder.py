from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel

from transformers.utils import logging
from transformers import Pix2StructVisionModel
from transformers import CLIPVisionModel
from chimera.model.chimera import InternVisionModel
from chimera.model.kosmos2_5.modeling_kosmos2_5 import Kosmos2_5VisionModel
from chimera.model.got import GotVisionConfig, GoTVisionModel, GOTImageProcessor

import torch
from torch import nn
from transformers.utils import logging

from .configuration_expert_encoder import ExpertEncoderConfig
from typing import Union, Dict, List, Optional
import pdb

try:
    from chimera.model.chimera.flash_attention import FlashAttention
    has_flash_attn = True
except:
    print('FlashAttention is not installed.')
    has_flash_attn = False

logger = logging.get_logger(__name__)

model_map={
    'pix2struct_vision_model': Pix2StructVisionModel,
    'kosmos_2_5_vision_model': Kosmos2_5VisionModel,
    'clip_vision_model': CLIPVisionModel,
    'intern_vit_6b' : InternVisionModel,
    'got_vision_model' : GoTVisionModel
}

def init_model_from_config(config, separate_load = False):
    # 避免AutoXXX出现问题，此处显式指定初始化的模型类型
    assert config.model_type in model_map, f"{config.model_type} is not implemented."
    model_type = model_map[config.model_type]
    name_or_path = config.name_or_path
    
    if name_or_path !="" and separate_load:
        print(f"loading pretrained encoder of {config.model_type} from {name_or_path}.")
        return model_type.from_pretrained(name_or_path, config=config)
    else:
        print(f"initialize encoder of {config.model_type}.")
        return model_type(config)
    
    

class ExpertEncoder(PreTrainedModel):
    config_class = ExpertEncoderConfig

    def __init__(self, config: ExpertEncoderConfig):
        super().__init__(config)
        # 初始化参数
        self.config = config
        self.num_encoder = self.config.num_encoder
        self.llm_hidden_size = config.llm_hidden_size


        # 便于直接用domain取index
        self.domain2index ={e["domain"]:e["index"] for e in config.config_list}
        # 根据索引得到hidden_size, select_layer等信息（简化版，详细信息需要从config取）
        self.index2config = [
            {
                "domain":e['domain'],
                "index":e['index'], 
                "hidden_size":e['config'].hidden_size, 
                "select_layer":e['select_layer']
            } for e in config.config_list
            ]

        # 初始化除了general encoder之外的encoder
        self.encoder = nn.ModuleList([init_model_from_config(e['config'], config.separate_load) for e in config.config_list])
        # 初始化之后不再使用separate_load
        self.config.separate_load = False

        for e in self.encoder:
            e.requires_grad_(False)

        #初始化对应的mlp，由于不采用Pixel shuffle，所以不使用donwnsample_ratio参数
        self.mlp = nn.ModuleList([
                    nn.Sequential(
                    nn.LayerNorm(e['hidden_size']),
                    nn.Linear(e['hidden_size'] , self.llm_hidden_size),
                    nn.GELU(),
                    nn.Linear(self.llm_hidden_size, self.llm_hidden_size)
                ) 
            for e in self.index2config
            ])
        
        
        assert len(self.mlp)==len(self.encoder)==self.num_encoder, f"There is a mismatch between index and domain. Got {len(self.mlp)} mlps, {len(self.encoder)} encoders and {self.num_encoder} encoders in config."
    
    # 根据index取domain
    def get_domain(self, index):
        assert index < self.num_encoder, f"index out of range, have {self.num_encoder} encoders, but get index of {index}."
        mapped_domain = self.index2config[index]['domain']
        assert index == self.domain2index[mapped_domain], "There is a mismatch between index and domain."
        return mapped_domain
    
    # 根据domain取index
    def get_index(self, domain):
        assert domain in self.domain2index, f"unsupported domain, supporting {list(self.domain2index.keys())}, but get {domain}."
        mapped_index = self.domain2index[domain]
        assert domain == self.index2config[mapped_index]['domain'], "There is a mismatch between index and domain."
        return mapped_index
    
    # 根据domina or index得到对应的encoder，select layer和mlp
    def get_uni_encoder(self,domain=None,index=None):
        assert domain is not None or index is not None, f"at least one para is needed"

        if domain is not None:
            mapped_index = self.get_index(domain)
            encoder = self.encoder[mapped_index]
            mlp = self.mlp[mapped_index]
            select_layer = self.index2config[mapped_index]['select_layer']
        if index is not None:
            mapped_domain = self.get_domain(index)
            encoder = self.encoder[index]
            mlp = self.mlp[index]
            select_layer = self.index2config[index]['select_layer']
        
        if domain and index:
            assert mapped_index == index and mapped_domain == domain, "get mismatched index and domain."

        return encoder, mlp, select_layer

    # 调用单个domain的编码器进行编码，用于推理
    def uni_encode(
            self,
            encoder_index = None,
            encoder_domain = None,
            pixel_values: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
    ):

        uni_encoder, mlp, select_layer = self.get_uni_encoder(encoder_domain,encoder_index)
        encoder_type = uni_encoder.config.model_type

        if encoder_type in ('intern_vit_6b','clip_vision_model'):
            assert pixel_values.dim()==4, f"Input shape for clip/internvit model should be (B,C,W,H), but got {pixel_values.shape}."
            if select_layer == -1:
                visual_feature = uni_encoder(
                    pixel_values=pixel_values,
                    output_hidden_states=False,
                    return_dict=True).last_hidden_state
            else:
                visual_feature = uni_encoder(
                    pixel_values=pixel_values,
                    output_hidden_states=True,
                    return_dict=True).hidden_states[select_layer]
            visual_feature = visual_feature[:, 1:, :]
                
        if encoder_type in ('pix2struct_vision_model', "kosmos_2_5_vision_model"):
            assert pixel_values.dim()==3, f"Input shape for pix2struct model should be (B,L,D), but got {pixel_values.shape}."
            if select_layer == -1:
                visual_feature = uni_encoder(
                    flattened_patches = pixel_values,
                    attention_mask = attention_mask,
                    output_hidden_states = False,
                    return_dict = True,
                    ).last_hidden_state
            else:
                visual_feature = uni_encoder(
                    flattened_patches = pixel_values,
                    attention_mask = attention_mask,
                    output_hidden_states = True,
                    return_dict = True,
                    ).hidden_states[select_layer]
                
        if encoder_type in ('got_vision_model'):
            assert pixel_values.dim()==4, f"Input shape for got model should be (B,C,W,H), but got {pixel_values.shape}."
            visual_feature = uni_encoder(pixel_values).last_hidden_state

        visual_feature = mlp(visual_feature)
       
        return visual_feature
    
    # 调用所有的编码器对所有图片进行编码，用于训练，后续过程选择性取出每个图片对应的特征
    def forward(
            self,
            pixel_value_list: List[torch.FloatTensor] = None,
            attention_mask_list: List[torch.FloatTensor] = None,
    ):
        r"""
        Args:
            pixel_values (`list`):
                A list of processed image pixel values, processed by different preporcessors of each domain encoder
                length: self.num_encoder
                element of pixel_values is tensor of size (B,C,W,H) or (B,L,D), the index of each element should match the encoder index of model, 
                i.e., pixel_values[0] is the input of self.encoder[0]
            attention_mask (`list`):
                A list of tensor that specially prepared for pix2struct & Kosmos encoder input.
                length: self.num_encoder
                element of attention_mask is tensor of size or (B,L,D), the index of each element should match the encoder index of model, only the elements corresponding to pix2struct & Kosmos encoder is not None
        """
        
        assert len(pixel_value_list)==self.num_encoder, f"Mismatched images, model has {self.num_encoder} encoder, but got {len(pixel_value_list)} images."
        
        res = []

        for i, cur_pixel_value in enumerate(pixel_value_list):
            cur_attention_mask = attention_mask_list[i]
            cur_visual_feature = self.uni_encode(
                                                encoder_index = i,
                                                pixel_values = cur_pixel_value,
                                                attention_mask = cur_attention_mask,
                                                )
            res.append(cur_visual_feature)
      

        return res



    

    
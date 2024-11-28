import warnings
from typing import Any, List, Optional, Tuple, Union

import torch.distributed as dist
import torch.utils.checkpoint
import transformers
from chimera.conversation import get_conv_template
from chimera.model.internlm2.modeling_internlm2 import InternLM2ForCausalLM
from chimera.model.phi3.modeling_phi3 import Phi3ForCausalLM
from peft import LoraConfig, get_peft_model
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import (AutoModel, GenerationConfig, LlamaForCausalLM,
                          LlamaTokenizer, Qwen2ForCausalLM)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput, logging

from .configuration_chimera import ChimeraChatConfig
from .modeling_intern_vit import InternVisionModel



from chimera.model.expert_encoder.configuration_expert_encoder import ExpertEncoderConfig
from chimera.model.expert_encoder.modeling_expert_encoder import ExpertEncoder
import pdb


logger = logging.get_logger(__name__)


def version_cmp(v1, v2, op='eq'):
    import operator

    from packaging import version
    op_func = getattr(operator, op)
    return op_func(version.parse(v1), version.parse(v2))



class ChimeraChatModel(PreTrainedModel):
    config_class = ChimeraChatConfig
    main_input_name = 'pixel_values'
    _no_split_modules = ['InternVisionModel', 'LlamaDecoderLayer', 'InternLM2DecoderLayer',
                         'Phi3DecoderLayer', 'Qwen2DecoderLayer']
    _supports_flash_attn_2 = True

    def __init__(self, config: ChimeraChatConfig, vision_model=None, language_model=None, expert_encoder = None):
        super().__init__(config)

        assert version_cmp(transformers.__version__, '4.37.0', 'ge')
        image_size = config.force_image_size or config.vision_config.image_size
        patch_size = config.vision_config.patch_size
        self.patch_size = patch_size
        self.select_layer = config.select_layer
        self.template = config.template
        self.num_image_token = int((image_size // patch_size) ** 2 * (config.downsample_ratio ** 2))
        self.downsample_ratio = config.downsample_ratio
        self.ps_version = config.ps_version
        self.llm_arch_name = config.llm_config.architectures[0]

        logger.info(f'num_image_token: {self.num_image_token}')
        logger.info(f'ps_version: {self.ps_version}')

        # LLaVA的做法是config里有才初始化，没有就不初始化
        # InternVL的做法是输入的有就用输入的，没有就按config初始化，默认config里面有visionencoder
        if vision_model is not None:
            self.vision_model = vision_model
        else:
            self.vision_model = InternVisionModel(config.vision_config)

        
        # 0为null，对应不调用ExpertEncoder
        # !! 只有从router中输出的index以及训练使用的label需要-1 shift，其余编号全是0~num_encoder-1 !!
        # pdb.set_trace()
        if expert_encoder is not None:
            if config.expert_encoder_config is None:
                config.expert_encoder_config = expert_encoder.config

            self.num_expert_encoder = config.expert_encoder_config.num_encoder
            self.expert_encoder = expert_encoder
            self.expert_router = nn.Linear(config.vision_config.hidden_size, config.expert_encoder_config.num_encoder + 1, bias=False)
            for i in range(self.num_expert_encoder):
                setattr(self, f'domain_{i}_context_token_id', 0)
        else:
            if config.expert_encoder_config is None:
                self.num_expert_encoder = None
                self.expert_encoder = None
                self.expert_router = None
            else:
                self.num_expert_encoder = config.expert_encoder_config.num_encoder
                self.expert_encoder = ExpertEncoder(config.expert_encoder_config)
                self.expert_router = nn.Linear(config.vision_config.hidden_size, config.expert_encoder_config.num_encoder + 1, bias=False)
                for i in range(self.num_expert_encoder):
                    setattr(self, f'domain_{i}_context_token_id', 0)
        #/pts
        
        # self.table_context_token_id = None
        # self.chart_context_token_id = None
        # self.math_context_token_id = None
        


        if language_model is not None:
            self.language_model = language_model
        else:
            if config.llm_config.architectures[0] == 'LlamaForCausalLM':
                self.language_model = LlamaForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == 'InternLM2ForCausalLM':
                self.language_model = InternLM2ForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == 'Phi3ForCausalLM':
                self.language_model = Phi3ForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == 'Qwen2ForCausalLM':
                self.language_model = Qwen2ForCausalLM(config.llm_config)
            else:
                raise NotImplementedError(f'{config.llm_config.architectures[0]} is not implemented.')

        vit_hidden_size = config.vision_config.hidden_size
        llm_hidden_size = config.llm_config.hidden_size

        self.mlp1 = nn.Sequential(
            nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio) ** 2),
            nn.Linear(vit_hidden_size * int(1 / self.downsample_ratio) ** 2, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size)
        )

        self.img_context_token_id = 0
       


        self.conv_template = get_conv_template(self.template)
        if hasattr(config, 'system_message'):
            self.system_message = config.system_message
        else:
            self.system_message = self.conv_template.system_message
        self.num_samples = 0

        if config.use_backbone_lora:
            self.wrap_backbone_lora(r=config.use_backbone_lora, lora_alpha=2 * config.use_backbone_lora)

        if config.use_llm_lora:
            self.wrap_llm_lora(r=config.use_llm_lora, lora_alpha=2 * config.use_llm_lora)

    def set_domain_context_token_ids(self, token_ids):
        assert len(token_ids) == self.num_expert_encoder, f"Got {len(token_ids)} to set, but supports {self.num_expert_encoder} domain"
        for i in range(self.num_expert_encoder):
            setattr(self, f'domain_{i}_context_token_id', token_ids[i]) 


    def wrap_backbone_lora(self, r=128, lora_alpha=256, lora_dropout=0.05):
        lora_config = LoraConfig(
            r=r,
            target_modules=['attn.qkv', 'attn.proj', 'mlp.fc1', 'mlp.fc2'],
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        self.vision_model = get_peft_model(self.vision_model, lora_config)
        self.vision_model.print_trainable_parameters()

    def wrap_llm_lora(self, r=128, lora_alpha=256, lora_dropout=0.05):
        # Determine the target modules based on the architecture of the language model
        if self.llm_arch_name == 'InternLM2ForCausalLM':
            target_modules = ['attention.wqkv', 'attention.wo', 'feed_forward.w1', 'feed_forward.w2', 'feed_forward.w3']
        elif self.llm_arch_name == 'Phi3ForCausalLM':
            target_modules = ['mlp.down_proj', 'mlp.gate_up_proj', 'self_attn.o_proj', 'self_attn.qkv_proj']
        elif self.llm_arch_name in ['Qwen2ForCausalLM', 'LlamaForCausalLM']:
            target_modules = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
                              'mlp.gate_proj', 'mlp.down_proj', 'mlp.up_proj']
        else:
            raise NotImplemented
        lora_config = LoraConfig(
            r=r,
            target_modules=target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            task_type='CAUSAL_LM'
        )
        self.language_model = get_peft_model(self.language_model, lora_config)
        self.language_model.enable_input_require_grads()
        self.language_model.print_trainable_parameters()
    
    def forward(
            self,
            pixel_values: torch.FloatTensor,
            expert_encoder_pixel_value_list: List[torch.FloatTensor] = [],
            expert_encoder_attention_mask_list: List[torch.FloatTensor] = [],
            expert_domain_ids: Optional[torch.Tensor] = None,
            thumbnail: torch.FloatTensor = None,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            image_flags: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # <-------------------------------- 获取general image feature并替换 -------------------------------->
        image_flags = image_flags.squeeze(-1)
        input_embeds = self.language_model.get_input_embeddings()(input_ids).clone()

        vit_embeds = self.extract_feature(pixel_values)
        vit_embeds = vit_embeds[image_flags == 1]
        vit_batch_size = pixel_values.shape[0]

        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        
        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            print(f'dynamic ViT batch size: {vit_batch_size}, images per sample: {vit_batch_size / B}, dynamic token length: {N}')
            
      

        input_ids = input_ids.reshape(B * N)
        selected = (input_ids == self.img_context_token_id)

        try:
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)
            ignore_flag = False
        except Exception as e:
            vit_embeds = vit_embeds.reshape(-1, C)
            print(f'warning: {e}, input_embeds[selected].shape={input_embeds[selected].shape}, '
                  f'vit_embeds.shape={vit_embeds.shape}')
            n_token = selected.sum()
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds[:n_token]
            ignore_flag = True


        input_embeds = input_embeds.reshape(B, N, C)

        # <-------------------------------- 获取domain image feature并替换 -------------------------------->
        
        assert len(expert_encoder_pixel_value_list)== len(expert_encoder_attention_mask_list) and expert_domain_ids.shape[0] == thumbnail.shape[0], \
        f"for expert encoder branch, number of pixel_value and attention must be consistant, expert_domain_ids and thumbnail must be consistant, but get {len(expert_encoder_pixel_value_list)} pixel value, {len(expert_encoder_attention_mask_list)} attention mask, domain id of shape {expert_domain_ids.shape} and thumbnail of shape {thumbnail.shape}"
        

        route_logits = None
        route_labels = None
        if len(expert_encoder_pixel_value_list)>0 and len(expert_encoder_attention_mask_list)>0 and expert_domain_ids is not None and thumbnail is not None:
            # !! 只有从router中输出的index以及训练使用的label需要-1 shift，其余编号全是0~num_encoder-1 !!
            # B,N_encoder
            route_logits = self.expert_route(thumbnail)
            # B,
            route_labels = expert_domain_ids
            # list, len=num_encoder
            expert_visual_features = self.expert_encoder(
                pixel_value_list = expert_encoder_pixel_value_list,
                attention_mask_list = expert_encoder_attention_mask_list,
                domain_ids = expert_domain_ids-1
            )

            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)
            input_ids = input_ids.reshape(B * N)

            # 按照不同的模态分别替换特征
            # shift -1 ,null类对应-1，其余index和expert encoder一致
            domain_mask = expert_domain_ids-1
            for i in range(self.num_expert_encoder):
                # pdb.set_trace()
                cur_domain_context_token_id = getattr(self, f"domain_{i}_context_token_id")
                cur_domain_mask = domain_mask==i
                # (any, L, D)
                cur_select_domain_feature = expert_visual_features[i][cur_domain_mask]

                cur_selected = (input_ids == cur_domain_context_token_id)
                try:
                    input_embeds[cur_selected] = input_embeds[cur_selected] * 0.0 + cur_select_domain_feature.reshape(-1, C)
                    ignore_flag = False
                except Exception as e:
                    cur_select_domain_feature = cur_select_domain_feature.reshape(-1, C)
                    print(f'warning: {e}, input_embeds[cur_selected].shape={input_embeds[cur_selected].shape}, '
                        f'cur_select_domain_feature.shape={cur_select_domain_feature.shape}')
                    n_token = cur_selected.sum()
                    input_embeds[cur_selected] = input_embeds[cur_selected] * 0.0 + cur_select_domain_feature[:n_token]
                    ignore_flag = True
            
          
            input_embeds = input_embeds.reshape(B, N, C)

        # <-------------------------------- 输入Decoder 进行后续损失计算 -------------------------------->

        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            if ignore_flag:
                loss = loss * 0.0

        
        if route_labels is not None and route_logits is not None:
            # 尺寸分别为(B)和(B,num_encoder),不需要调整
            route_loss = loss_fct(route_logits, route_labels)
            loss = loss + route_loss
        
        
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output


        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                   int(c / (scale_factor * scale_factor)))
        if self.ps_version == 'v1':
            warnings.warn("In ps_version 'v1', the height and width have not been swapped back, "
                          'which results in a transposed image.')
        else:
            x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def extract_feature(self, pixel_values):
        if self.select_layer == -1:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=False,
                return_dict=True).last_hidden_state
        else:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True).hidden_states[self.select_layer]
        vit_embeds = vit_embeds[:, 1:, :]

        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        vit_embeds = self.mlp1(vit_embeds)
        return vit_embeds
    
    def expert_route(self, pixel_values):
        pooled_feature = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=False,
                return_dict=True).pooler_output
        logits = self.expert_router(pooled_feature)
        # pdb.set_trace()
        return logits


    def batch_chat(self, tokenizer, pixel_values, questions, generation_config, num_patches_list=None,
                   history=None, return_history=False, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>',
                   IMG_CONTEXT_TOKEN='<IMG_CONTEXT>', verbose=False, image_counts=None):
        if history is not None or return_history:
            print('Now multi-turn chat is not supported in batch_chat.')
            raise NotImplementedError

        if image_counts is not None:
            num_patches_list = image_counts
            print('Warning: `image_counts` is deprecated. Please use `num_patches_list` instead.')

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        queries = []
        for idx, num_patches in enumerate(num_patches_list):
            question = questions[idx]
            if pixel_values is not None and '<image>' not in question:
                question = '<image>\n' + question
            template = get_conv_template(self.template)
            template.system_message = self.system_message
            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()

            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)
            queries.append(query)

        tokenizer.padding_side = 'left'
        model_inputs = tokenizer(queries, return_tensors='pt', padding=True)
        input_ids = model_inputs['input_ids'].cuda()
        attention_mask = model_inputs['attention_mask'].cuda()
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep)
        generation_config['eos_token_id'] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
        responses = tokenizer.batch_decode(generation_output, skip_special_tokens=True)
        responses = [response.split(template.sep)[0].strip() for response in responses]
        return responses

    def chat(
            self, 
            tokenizer, 
            pixel_values, 
            question, 
            generation_config, 
            expert_encoder_pixel_value_list: List[torch.FloatTensor] = [],
            expert_encoder_attention_mask_list: List[torch.FloatTensor] = [],
            thumbnail: torch.FloatTensor = None,
            num_expert_token_all: List = [],
            history=None, 
            return_history=False,
            num_patches_list=None, 
            IMG_START_TOKEN='<img>', 
            IMG_END_TOKEN='</img>', 
            IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',
            
            DOMAIN_START_TOKEN = '<domain>',
            DOMAIN_END_TOKEN = '</domain>',
            
            verbose=False
             ):
    
        if history is None and pixel_values is not None and '<image>' not in question:
            question = '<image>\n' + question
        
        if num_patches_list is None:
            num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []

        assert question.count("<image>")==len(thumbnail)==len(num_patches_list), f'there are {question.count("<image>")} <image> token in question but get {len(num_patches_list)} input images and {len(thumbnail)} thumbnails'

        
        assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        expert_token_list = []
        for i in range(self.num_expert_encoder):
            expert_token_list.append(f"<DOMAIN_{i}_CONTEXT>")
        expert_context_token_id = tokenizer.convert_tokens_to_ids(expert_token_list)
        self.set_domain_context_token_ids(expert_context_token_id)



        template = get_conv_template(self.template)
        template.system_message = self.system_message
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep)

        history = [] if history is None else history
        for (old_question, old_answer) in history:
            template.append_message(template.roles[0], old_question)
            template.append_message(template.roles[1], old_answer)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        
        if len(expert_encoder_pixel_value_list)>0 and len(expert_encoder_attention_mask_list)>0 and thumbnail is not None:
            num_expert_token_list = []
            domain_context_token_list = []
            route_logits = self.expert_route(thumbnail)
            expert_domain_ids = torch.argmax(route_logits,dim=-1)

            expert_mask = torch.zeros((self.num_expert_encoder,len(thumbnail))).bool().to(device = self.device)
            for i, domain_id in enumerate(expert_domain_ids):
                if domain_id >0:
                    print(f"detect the domain {self.config.expert_encoder_config.config_list[domain_id-1]['domain']} with domain feature length: {num_expert_token_all[domain_id-1]}")
                    domain_context_token_list.append(expert_token_list[domain_id-1])
                    num_expert_token_list.append(num_expert_token_all[domain_id-1])
                    expert_mask[domain_id-1][i]=True
                else:
                    domain_context_token_list.append(None)
                    num_expert_token_list.append(0)
            
            # pdb.set_trace()
            for i in range(self.num_expert_encoder):
                expert_encoder_pixel_value_list[i] = expert_encoder_pixel_value_list[i][expert_mask[i]]
                expert_encoder_attention_mask_list[i] = expert_encoder_attention_mask_list[i][expert_mask[i]] if expert_encoder_attention_mask_list[i] is not None else expert_encoder_attention_mask_list[i]

        
        for i, num_patches in enumerate(num_patches_list):
            if domain_context_token_list and num_expert_token_list and num_expert_token_list[i]>0:
                image_tokens = f'{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * self.num_image_token * num_patches}{IMG_END_TOKEN}{DOMAIN_START_TOKEN}{domain_context_token_list[i] * num_expert_token_list[i]}{DOMAIN_END_TOKEN}'
            else:
                image_tokens = f'{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * self.num_image_token * num_patches}{IMG_END_TOKEN}'
            query = query.replace('<image>', image_tokens, 1)
        


        model_inputs = tokenizer(query, return_tensors='pt')
        input_ids = model_inputs['input_ids'].cuda()
        attention_mask = model_inputs['attention_mask'].cuda()
        generation_config['eos_token_id'] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            expert_encoder_pixel_value_list = expert_encoder_pixel_value_list,
            expert_encoder_attention_mask_list = expert_encoder_attention_mask_list,
            **generation_config
        )
        response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
        response = response.split(template.sep)[0].strip()
        history.append((question, response))
        if return_history:
            return response, history
        else:
            query_to_print = query.replace(IMG_CONTEXT_TOKEN, '')
            query_to_print = query_to_print.replace(f'{IMG_START_TOKEN}{IMG_END_TOKEN}', '<image>')
            if verbose:
                print(query_to_print, response)
            return response
        

    @torch.no_grad()
    def generate(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            input_ids: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            visual_features: Optional[torch.FloatTensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            expert_encoder_pixel_value_list: List[torch.FloatTensor] = [],
            expert_encoder_attention_mask_list: List[torch.FloatTensor] = [],
            **generate_kwargs,
    ) -> torch.LongTensor:

        assert self.img_context_token_id is not None
        if pixel_values is not None:
            if visual_features is not None:
                vit_embeds = visual_features
            else:
                vit_embeds = self.extract_feature(pixel_values)
            input_embeds = self.language_model.get_input_embeddings()(input_ids)
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)


            input_ids = input_ids.reshape(B * N)
            selected = (input_ids == self.img_context_token_id)
            # assert selected.sum() != 0
            if selected.sum() != 0:
                input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)
    
            
            if len(expert_encoder_pixel_value_list)>0 and len(expert_encoder_attention_mask_list)>0 :
                domain_feature = []
                for i, cur_domain_pixel in enumerate(expert_encoder_pixel_value_list):
                    if cur_domain_pixel.size(0)>0:
                        cur_domain_mask = expert_encoder_attention_mask_list[i]
                        domain_feature.append(
                            self.expert_encoder.uni_encode(
                                                encoder_index = i,
                                                pixel_values = cur_domain_pixel,
                                                attention_mask = cur_domain_mask,
                                                )
                        )
                    else:
                        domain_feature.append(None)
                
                for i in range(self.num_expert_encoder):
                    cur_domain_feature = domain_feature[i]
                    if cur_domain_feature is not None:
                        # pdb.set_trace()
                        cur_domain_context_token_id = getattr(self, f"domain_{i}_context_token_id")
                        # (any, L, D)
                        cur_selected = (input_ids == cur_domain_context_token_id)
                        if cur_selected.sum() != 0:
                            input_embeds[cur_selected] =  cur_domain_feature.reshape(-1, C).to(input_embeds.device)
            

            input_embeds = input_embeds.reshape(B, N, C)
        else:
            input_embeds = self.language_model.get_input_embeddings()(input_ids)


        outputs = self.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_cache=True,
            **generate_kwargs,
        )

        return outputs

import warnings
from transformers import ProcessorMixin
from chimera.model.kosmos2_5 import Kosmos2_5ImageProcessor
from chimera.model.got import GOTImageProcessor
from transformers import  CLIPImageProcessor
from transformers import Pix2StructImageProcessor
import copy
import pdb


def init_processor_from_meta_config(meta):
    if meta['image_processor_type'] == 'Pix2StructImageProcessor':
        # return Pix2StructImageProcessor.from_pretrained(meta['model_name_or_path'])
        return Pix2StructImageProcessor(**meta)
    
    elif meta['image_processor_type'] == "Kosmos2_5ImageProcessor":
        # return Kosmos2_5ImageProcessor.from_pretrained(meta['model_name_or_path'])
        return Kosmos2_5ImageProcessor(**meta)
    
    
    elif meta['image_processor_type'] == "GOTImageProcessor":
        # return GOTImageProcessor.from_pretrained(meta['model_name_or_path'])
        return GOTImageProcessor(**meta)
    
    
    elif meta['image_processor_type'] == 'CLIPImageProcessor':
        # processor =  CLIPImageProcessor.from_pretrained(meta['model_name_or_path'])
        processor =  CLIPImageProcessor(**meta)
        # 由于CLIP系列的processor中缺少max_patch这一参数，在此手动补全，根据具体使用的CLIP进行更改
        processor.max_patches = 576
        return processor
    else:
        raise NotImplementedError(f'{meta["image_processor_type"]} is not implemented.')


class ChimeraProcessor(ProcessorMixin):
    attributes = []
    def __init__(
        self,
        expert_processor_list: list,
        **kwargs,
    ) -> None:
        
        self.expert_processor_list = []

        for i in expert_processor_list:
            cur_processor = init_processor_from_meta_config(i)
            self.expert_processor_list.append(cur_processor)

        super().__init__()
    
    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        output['expert_processor_list'] = []
        for i in self.expert_processor_list:
            output['expert_processor_list'].append(i.to_dict())
        
        return output

# if __name__ =="__main__":
    # reasoner_list = [
    #     {
    #         "image_processor_type": "Kosmos2_5ImageProcessor",
    #         "max_patches": 2048,
    #         "processor_class": "Kosmos2_5Processor"
    #     },
    #     {
    #         'do_resize': True, 
    #         'size': {'shortest_edge': 336}, 
    #         'resample': 3, 
    #         'do_center_crop': True, 
    #         'crop_size': {'height': 336, 'width': 336}, 
    #         'do_rescale': True, 
    #         'rescale_factor': 0.00392156862745098, 
    #         'do_normalize': True, 
    #         'image_mean': [0.48145466, 0.4578275, 0.40821073], 
    #         'image_std': [0.26862954, 0.26130258, 0.27577711], 
    #         'do_convert_rgb': True, 
    #         'image_processor_type': 'CLIPImageProcessor'
    #     },
    #     {
    #         "do_convert_rgb": True,
    #         "do_normalize": True,
    #         "image_processor_type": "Pix2StructImageProcessor",
    #         "is_vqa": False,
    #         "max_patches": 2048,
    #         "patch_size": {
    #             "height": 16,
    #             "width": 16
    #         },
    #         "processor_class": "Pix2StructProcessor"
    #     }
    # ]
    # processor = ChimeraProcessor(reasoner_list)
    # processor.save_pretrained('/mnt/workspace/Chimera/checkpoints/test_dir')
    # new = ChimeraProcessor.from_pretrained('/mnt/workspace/Chimera/checkpoints/test_dir')

    # extractor_list = [
    #     {
    #     "image_processor_type": "GOTImageProcessor",
    #     "image_size": 1024,
    #     "mean": [
    #         0.48145466,
    #         0.4578275,
    #         0.40821073
    #     ],
    #     "std": [
    #         0.26862954,
    #         0.26130258,
    #         0.27577711
    #     ],
    #     "max_patches": 256
    #     }
    # ]
    # processor = ChimeraProcessor(extractor_list)
    # processor.save_pretrained('/mnt/workspace/Chimera/checkpoints/extractor_dir')
    # new = ChimeraProcessor.from_pretrained('/mnt/workspace/Chimera/checkpoints/extractor_dir')

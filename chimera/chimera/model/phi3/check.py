from chimera.model.phi3.modeling_phi3 import *
import pdb
from transformers import AutoTokenizer

model = Phi3ForCausalLM.from_pretrained("/cpfs01/shared/ADLab/datasets/science_dataset/checkpoints/OpenGVLab/InternVL2-4B")
tokenizer = AutoTokenizer.from_pretrained("/cpfs01/shared/ADLab/datasets/science_dataset/checkpoints/OpenGVLab/InternVL2-4B",use_fast = False)

test_sentence = "here is the mask, and the test starts"
input = tokenizer(test_sentence,return_tensors="pt")
input['attention_mask'][0,2:4]=0
input['attention_mask'][0,-1]=0

# torch.round(attention_mask[0,0]*100)/100
res = model(**input)
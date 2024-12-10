<div align="center">
<h1>Chimera: Improving Generalist Model with Domain-Specific Experts</h1>


[[ Paper ]](https://arxiv.org/abs/2412.05983) [[ Website ]](https://unimodal4reasoning.github.io/chimera_page/) [[ Dataset🤗 ]]() [[ Models🤗 ]](https://huggingface.co/collections/U4R/chimera-10-6749542e2f0dfa09414232c0) 

</div>

## News :fire:
- [x] Release the inference code and model checkpoints
- [ ] Release the training code and data recipe

## 🛠️ Installation

- Clone this repository:

  ```bash
  git clone https://github.com/UniModal4Reasoning/Chimera.git
  ```

- Create a conda virtual environment and activate it:

  ```bash
  conda create -n chimera python=3.9 -y
  conda activate chimera
  ```

- Install dependencies using `requirements.txt`:

  ```bash
  pip install -r requirements.txt
  ```

- Install other requirements:

  ```bash
  cd chimera/
  pip install --upgrade pip  # enable PEP 660 support
  pip install -e .
  ```

### Additional Instructions

- Install `flash-attn==2.3.4`:

  ```bash
  pip install flash-attn==2.3.4 --no-build-isolation
  ```

  Alternatively you can compile from source:

  ```bash
  git clone https://github.com/Dao-AILab/flash-attention.git
  cd flash-attention
  git checkout v2.3.4
  python setup.py install
  ```


## Quick Start
### Multi-modal reasoning
```python
from chimera.chimera_infer import Chimera4easyuse
import torch
from PIL import Image

# prepare model
# model_path = "U4R/Chimera-Reasoner-2B"
# model_path = "U4R/Chimera-Reasoner-4B"
model_path = "U4R/Chimera-Reasoner-8B"
generation_config = dict(max_new_tokens=256, do_sample=False)
model = Chimera4easyuse(model_path, dtype = torch.bfloat16, generation_config= generation_config)

# prepare input
image_path = "path/to/image"
user_prompt = "<image>\nuser prompt"
input_image = Image.open(image_path).convert('RGB')
response = model.get_response(user_prompt, [input_image])
print(response)
```

### Visual content extraction
```python
from chimera.chimera_infer import Chimera4easyuse
import torch
from PIL import Image

# prepare model
model_path = "U4R/Chimera-Extractor-1B"
generation_config = dict(max_new_tokens=4096, do_sample=False, no_repeat_ngram_size = 20)
model = Chimera4easyuse(model_path, dtype = torch.float16, generation_config= generation_config)

# prepare input
image_path = "path/to/document"
user_prompt = "<image>\nAs a smart PDF to Markdown conversion tool, please convert the content of the provided PDF into Markdown format."
input_image = Image.open(image_path).convert('RGB')
response = model.get_response(user_prompt, [input_image])
print(response)
```


## License
Chimera is released under the [Apache License 2.0](LICENSE)

## Citation
If you find our models / code / papers useful in your research, please consider giving ⭐ and citations 📝, thx :)  
```bibtex
@misc{peng2024chimeraimprovinggeneralistmodel,
      title={Chimera: Improving Generalist Model with Domain-Specific Experts}, 
      author={Tianshuo Peng and Mingsheng Li and Hongbin Zhou and Renqiu Xia and Renrui Zhang and Lei Bai and Song Mao and Bin Wang and Conghui He and Aojun Zhou and Botian Shi and Tao Chen and Bo Zhang and Xiangyu Yue},
      year={2024},
      eprint={2412.05983},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2412.05983}, 
}
```

## Contact Us
If you encounter any issues or have questions, please feel free to contact us via bo.zhangzx@gmail.com or zhangbo@pjlab.org.cn.

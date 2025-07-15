### Getting started
- Installing dependencies
```
pip install -r requirements.txt
```
### Data preparation
```
python ./download_dataset.py
```
### Pre-trained models
To evaluate AASIST [1]:
```
python main.py --eval --config ./config/AASIST.conf
```
To evaluate AASIST-L [1]:
```
python main.py --eval --config ./config/AASIST-L.conf
```
### Conversion to ONNX
```
#!/usr/bin/env python3

import torch
import numpy as np
import json
from torch.ao.quantization import quantize_pt2e

# ────────────────────────────────────────────────────────────────────────────
# 2.  LOAD MODEL
# ────────────────────────────────────────────────────────────────────────────
from AASIST import Model

with open("./config/AASIST.conf", 'r') as f:
  cfg = json.load(f)
  d_args = cfg['model_config']

device = torch.device("cpu")
model  = Model(d_args).to(device)
model.load_state_dict(torch.load("./models/weights/AASIST.pth", map_location=device))
model.eval()

# Create dummy input
dummy_input = torch.randn(1, 16000, dtype=torch.float32)

 # Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "./aasist.onnx",
    export_params=True,
    opset_version=13,
    input_names=["input"],
    output_names=["embedding", "logits"],
    dynamic_axes={"input": {1: "num_samples"}}
)
print(f"Model successfully exported")
```
### Run ONNX inference
```
!python inference_onnx.py \
  --onnx_model_path     aasist.onnx \
  --audio_path     path/to/audio/file
```

### Edge Impulse profiling
```
import edgeimpulse as ei

ei.API_KEY = "ei_656a225s82q1232..." # API key

ei.model.profile(model="./aasist.onnx", device='cortex-m4f-80mhz')
```

### License
```
Copyright (c) 2021-present NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

### Acknowledgements
This repository is built on top of several open source projects. 
- [ASVspoof 2021 baseline repo](https://github.com/asvspoof-challenge/2021/tree/main/LA/Baseline-RawNet2)
- [min t-DCF implementation](https://www.asvspoof.org/resources/tDCF_python_v2.zip)

The repository for baseline RawGAT-ST model will be open
-  https://github.com/eurecom-asp/RawGAT-ST-antispoofing

The dataset we use is ASVspoof 2019 [4]
- https://www.asvspoof.org/index2019.html

### References
[1] AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks
```bibtex
@INPROCEEDINGS{Jung2021AASIST,
  author={Jung, Jee-weon and Heo, Hee-Soo and Tak, Hemlata and Shim, Hye-jin and Chung, Joon Son and Lee, Bong-Jin and Yu, Ha-Jin and Evans, Nicholas},
  booktitle={arXiv preprint arXiv:2110.01200}, 
  title={AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks}, 
  year={2021}
```

[2] End-to-End anti-spoofing with RawNet2
```bibtex
@INPROCEEDINGS{Tak2021End,
  author={Tak, Hemlata and Patino, Jose and Todisco, Massimiliano and Nautsch, Andreas and Evans, Nicholas and Larcher, Anthony},
  booktitle={Proc. ICASSP}, 
  title={End-to-End anti-spoofing with RawNet2}, 
  year={2021},
  pages={6369-6373}
}
```

[3] End-to-end spectro-temporal graph attention networks for speaker verification anti-spoofing and speech deepfake detection
```bibtex
@inproceedings{tak21_asvspoof,
  author={Tak, Hemlata and Jung, Jee-weon and Patino, Jose and Kamble, Madhu and Todisco, Massimiliano and Evans, Nicholas},
  booktitle={Proc. ASVSpoof Challenge},
  title={End-to-end spectro-temporal graph attention networks for speaker verification anti-spoofing and speech deepfake detection},
  year={2021},
  pages={1--8}
```

[4] ASVspoof 2019: A large-scale public database of synthesized, converted and replayed speech
```bibtex
@article{wang2020asvspoof,
  title={ASVspoof 2019: A large-scale public database of synthesized, converted and replayed speech},
  author={Wang, Xin and Yamagishi, Junichi and Todisco, Massimiliano and Delgado, H{\'e}ctor and Nautsch, Andreas and Evans, Nicholas and Sahidullah, Md and Vestman, Ville and Kinnunen, Tomi and Lee, Kong Aik and others},
  journal={Computer Speech \& Language},
  volume={64},
  pages={101114},
  year={2020},
  publisher={Elsevier}
}
```

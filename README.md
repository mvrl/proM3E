# ProM3E: Probabilistic Multi-Modal Masked Embedding Model

<div align="center">
<img src="imgs/prom3e_logo.png" width="250">

[![arXiv](https://img.shields.io/badge/arXiv-2511.02946-red)](https://arxiv.org/pdf/2511.02946)
[![Project Page](https://img.shields.io/badge/Project-Website-green)](https://vishu26.github.io/prom3e/index.html)
[![Hugging Face Models](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Models-yellow
)](https://huggingface.co/MVRL/ProM3E)
[![Hugging Face Datasets](https://img.shields.io/badge/ProM3E-Datasets-blue)](https://huggingface.co/datasets/MVRL/MultiNat)</center>

[Srikumar Sastry*](https://vishu26.github.io/),
[Subash Khanal](https://subash-khanal.github.io/),
[Aayush Dhakal](https://scholar.google.com/citations?user=KawjT_8AAAAJ&hl=en),
[Jiayu Lin](),
[Dan Cher](https://dcher95.github.io/),
[Phoenix Jarosz](),
[Nathan Jacobs](https://jacobsn.github.io/)
(*Corresponding Author)

#### CVPR 2026
</div>

This repository is the official implementation of [ProM3E](https://arxiv.org/pdf/2511.02946).
ProM3E is a probabilistic multimodal model that learns to predict missing modalities in the embedding space given the observed ones. This improves representations of existing encoders and enables robust multimodal learning.

![](imgs/prom3e_logo.jpg)

## ⚙️ Usage
Our pretrained model are made available through `rshf` and `transformers` package for easy inference.

Load and initialize:
```python
from rshf.prom3e import ProM3E

model = ProM3E.from_pretrained("MVRL/ProM3E")
```

Inference:
```python
# Get precomputed embeddings from taxabind for image, sat, loc, env, text, audio
# Replace missing modalities with any vector
# Stack embeddings in the order: image, sat, loc, env, text, audio
# Pass through the model

# Example:
image_embeds = torch.randn(2, 512)
sat_embeds = torch.randn(2, 512)
loc_embeds = torch.randn(2, 512)
env_embeds = torch.randn(2, 512)
text_embeds = torch.randn(2, 512)
audio_embeds = torch.randn(2, 512)

modalities = torch.stack((image_embeds, sat_embeds, loc_embeds, env_embeds, text_embeds, audio_embeds), dim=1)

modalities = torch.nn.functional.normalize(modalities, dim=-1)

unmasked_modalities = [0, 2]

reconstructions, mu, log_var, hidden_repr = model.forward_inference(modalities, unmasked_modalities)
```

## 🎯 Zero-Shot Image Classification
![](imgs/zero_shot.png)
Our framework outperforms the state-of-the-art in both unimodal (BioCLIP, ArborCLIP, TaxaBind) and multimodal setting (ImageBind, TaxaBind).

## 👫 Modality Gap
![](imgs/modality_gap.png)
Our training strategy effectively mitigates the modality gap between the encoders.

## 📈 Data Scalability
![](imgs/scaling.png)
We show that our model can be trained with much less all paired dataset and the performance across various dataset sizes and tasks remain consistent. For instance training with 10% of the dataset (7,913 samples) only results in a performance drop of ∼3% on average.


## 🔥 Large Mulitmodal Ecological Datasets

We release [MultiNat](https://huggingface.co/datasets/MVRL/MultiNat), a truly multimodal dataset containing six paired modalities for evaluating large ecological models.

📑 Citation

```bibtex
@inproceedings{sastry2026prom3e,
    title={ProM3E: Probabilistic Multi-Modal Masked Embedding Model},
    author={Sastry, Srikumar and Khanal, Subash and Dhakal, Aayush and Ahmad, Adeel and Jacobs, Nathan},
    booktitle={Conference on Computer Vision and Pattern Recognition},
    year={2026},
    organization={IEEE/CVF}
}
```


## 🔍 Additional Links
Check out our lab website for other interesting works on geospatial understanding and mapping:
* Multi-Modal Vision Research Lab (MVRL) - [Link](https://mvrl.cse.wustl.edu/)
* Related Works from MVRL - [Link](https://mvrl.cse.wustl.edu/publications/)

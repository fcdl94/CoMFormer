# CoMFormer: Continual Learning in Semantic and Panoptic Segmentation


[Fabio Cermelli](https://fcdl94.github.io/), Matthieu Cord, Arthur Douillard

[ [`arXiv`](https://arxiv.org/abs/2211.13999/) ] [ [`BibTeX`](#Citing) ]

[comment]: <> (<div align="center">)

[comment]: <> (<img src="https://bowenc0221.github.io/images/maskformerv2_teaser.png" width="100%" height="100%"/>)

[comment]: <> (</div><br/>)

## Installation
See [installation instructions](INSTALL.md).

## Getting Started

### Prepare the datasets
See [Preparing Datasets for Mask2Former](datasets/README.md).

### How to configure the methods:
Per-Pixel baseline:
`MODEL.MASK_FORMER.PER_PIXEL True`

Mask-based methods:
`MODEL.MASK_FORMER.SOFTMASK True MODEL.MASK_FORMER.FOCAL True`

CoMFormer:
`CONT.DIST.PSEUDO True CONT.DIST.KD_WEIGHT 10.0 CONT.DIST.UKD True CONT.DIST.KD_REW True`

MiB:
`CONT.DIST.KD_WEIGHT 200.0 CONT.DIST.UKD True CONT.DIST.UCE True`

PLOP:
`CONT.DIST.PSEUDO True CONT.DIST.PSEUDO_TYPE 1  CONT.DIST.POD_WEIGHT 0.001`

### How to run experiments:
ADE Semantic Segmenation:
- Use config file: `cfg_file=configs/ade20k/semantic-segmentation/maskformer2_R101_bs16_90k.yaml`
- 100-50: `CONT.BASE_CLS 100 CONT.INC_CLS 50 CONT.MODE overlap` (see examples in `scripts/ade.sh`) 
- 100-10: `CONT.BASE_CLS 100 CONT.INC_CLS 10 CONT.MODE overlap` (see examples in `scripts/ade10.sh`)
- 100-5: `CONT.BASE_CLS 100 CONT.INC_CLS 5 CONT.MODE overlap` (see examples in `scripts/ade5.sh`)

ADE Panoptic Segmenation:
- Use config file: `cfg_file=configs/ade20k/panoptic-segmentation/maskformer2_R50_bs16_90k.yaml`
- 100-50: `CONT.BASE_CLS 100 CONT.INC_CLS 50 CONT.MODE overlap` (see examples in `scripts/adps.sh`) 
- 100-10: `CONT.BASE_CLS 100 CONT.INC_CLS 10 CONT.MODE overlap` (see examples in `scripts/adps10.sh`)
- 100-5: `CONT.BASE_CLS 100 CONT.INC_CLS 5 CONT.MODE overlap` (see examples in `scripts/adps5.sh`)

## <a name="Citing"></a>Citing CoMFormer
If you use CoMFormer in your research, please use the following BibTeX entry.

```BibTeX
@article{cermelli2023comformer,
  title={CoMFormer: Continual Learning in Semantic and Panoptic Segmentation},
  author={Fabio Cermelli and Matthieu Cord and Arthur Douillard},
  journal={IEEE/CVF Computer Vision and Pattern Recognition Conference},
  year={2023}
}
```

## Acknowledgement
The code is largely based on [Mask2Former](https://github.com/facebookresearch/Mask2Former).

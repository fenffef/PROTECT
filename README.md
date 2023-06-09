# T5-corrector: Few-shot Learning for Robust Chinese Text Correction

---

This repository contains code, model, datasets for T5-corrector



## Pretrain

**dataset**

Pretrain dataset available at [here](https://drive.google.com/file/d/1qvp1G4DkvQALnJDoMPRWt1A4NqoAq7JD/view?usp=share_link)

```
cd scripts
sh pretrain.sh
```

## Finetune

**pretrained model and datasets**

To facilitate replication, we provide pre-trained T5-corrector model and datasets.

pre-trained T5-corrector model available at [here](https://drive.google.com/file/d/16KNXFcbEiC9Wzv638l5OCTZsmmyTlJL8/view?usp=share_link)

datasets available at [here](https://drive.google.com/file/d/1OeEX212_lbleP9a1Wmi3iWrdY1kIbZYi/view?usp=share_link)

```
cd scripts
sh train.sh
```
# Agile Quant

Official repo for paper: [Agile-Quant: Activation-Guided Quantization for Faster Inference of LLMs on the Edge](https://arxiv.org/abs/2312.05693)

This paper is accepted by NeurIPS 2024


## Usage
1. Replace llama model files in `transformers` package with `transformers/models/llama`
2. Download models `sh download.sh`
3. Use GPTQ to quantize weights `sh run-gptq-llama.sh`
4. Quantize activation with `gptq_fq_quant_llama.py`

## Citation
```
@inproceedings{
    shen2024agile,
    title     = {Agile-Quant: Activation-Guided Quantization for Faster Inference of LLMs on the Edge},
    author    = {Shen, Xuan and Dong, Peiyan and Lu, Lei and Kong, Zhenglun and Li, Zhengang and Lin, Ming and Wu, Chao and Wang, Yanzhi},
    booktitle = {AAAI},
    year      = {2024},
}
```

## Acknowledgment
The code is mainly based on the quantization works [GPTQ](https://github.com/IST-DASLab/gptq) and [FQ-ViT](https://github.com/megvii-research/FQ-ViT).


MODEL_DIR="llama-7b"

# Genereate quantized model
 python3 -u gptq_fq_quant_llama.py --model ${MODEL_DIR} --dataset wikitext2 \
         --wbits 4 --true-sequential --act-order --groupsize 128

# Evaluation
python3 -u gptq_fq_quant_llama.py --model ${MODEL_DIR} --dataset wikitext2 \
        --eval \
        --wbits 4 --groupsize 128 \
        --load checkpoints-quantized/llama-7b-fq-quantized.ckpt \
        --save-quant-info checkpoints-quantization-info \
        # --load-quant-info checkpoints-quantization-info \
        # --layers-dist 9:10:10:10:10:10:10:11 \   # this is for llama-1-65b

# Test
python3 -u gptq_fq_quant_llama.py --model ${MODEL_DIR} --dataset wikitext2 \
        --test-generation \
        --wbits 4 --groupsize 128 \
        --load checkpoints-quantized/llama-7b-fq-quantized.ckpt \
        --load-quant-info checkpoints-quantization-info \
        --max-response-length 20 \
        --chat "How about the Northeastern University?"
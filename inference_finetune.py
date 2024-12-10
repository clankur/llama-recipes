# %%
import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
# %% [markdown]
# Run this command to finetune the model
# ```
# python finetuning.py 
# --lr 1e-5  
# --num_epochs 3 
# --batch_size_training 2 
# --model_name meta-llama/Llama-3.2-1B 
# --dist_checkpoint_root_folder ./finetuned_model 
# --dist_checkpoint_folder fine-tuned  
# --use_fast_kernels 
# --dataset "custom_dataset" 
# --custom_dataset.test_split "test" 
# --custom_dataset.file "datasets/cfa_dataset.py"  
# --run_validation True 
# --batching_strategy padding
# ```


# %%
model_name = "meta-llama/Llama-3.2-1B"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
# %%
# safetensors_path = "/tmp/finetuned/"
# model = PeftModel.from_pretrained(model, safetensors_path)
# %%
text = """Read the questions and answers carefully, and choose the one you think is appropriate among the three options A, B and C. Q:The nominal risk-free rate is best described as the sum of the real risk-free rate and a premium for:,CHOICES: A: maturity.,B: liquidity.,C: expected inflation. Answer:"""
inputs = tokenizer(text, return_tensors="pt", max_length=2048)

# Generate predictions
outputs = model.generate(inputs["input_ids"])

# Decode the output
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)

# %%

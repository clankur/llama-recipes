# %%
import os
os.environ["HF_HOME"] = "/media/clankur/Windows Partition/huggingface"

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
# %%
print("HF Home:", os.getenv("HF_HOME"))

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
# model_name = "meta-llama/Llama-3.2-1B"
path = "/media/clankur/Windows Partition/huggingface/hub/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08"
model = AutoModelForCausalLM.from_pretrained(path)
tokenizer = AutoTokenizer.from_pretrained(path)
# %%
# if lora and peft
# safetensors_path = "/tmp/finetuned/"
# model = PeftModel.from_pretrained(model, safetensors_path)
# %%
checkpoint_path = "/tmp/finetuned/model.pt"
state_dict = torch.load(checkpoint_path)
model.load_state_dict(state_dict)
model.eval() 

# %%
text = """Read the questions and answers carefully, and choose the one you think is appropriate among the three options A, B and C. Q:The nominal risk-free rate is best described as the sum of the real risk-free rate and a premium for:,CHOICES: A: maturity.,B: liquidity.,C: expected inflation. Answer:"""
inputs = tokenizer(text, return_tensors="pt", max_length=2048)

# Generate predictions
outputs = model.generate(inputs["input_ids"])

# Decode the output
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)

# %%
from recipes.quickstart.finetuning.datasets.cfa_dataset import get_custom_dataset
dataset_config = {
    "num_examples": 0,
}
dataset_val = get_custom_dataset(
    dataset_config,
    tokenizer,
    "test",
)


# %%
val_iter = iter(dataset_val)
for _ in range(10):
    sample = next(val_iter)
    print(sample)

# %%


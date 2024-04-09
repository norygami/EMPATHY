#@title Imports
import os
import torch
from torch.quantization import get_default_qat_qconfig, prepare_qat, convert
from datasets import load_dataset
from transformers import (
    DataCollatorForLanguageModeling,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
    BitsAndBytesConfig,
)
import bitsandbytes
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
from datasets import concatenate_datasets, load_dataset
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"

!pip install -q accelerate==0.21.0 peft==0.4.0 transformers==4.31.0 trl==0.4.7
!pip install bitsandbytes

#@title Parameters
# Define the base path in Google Drive to store the model
#base_path = '/content/drive/MyDrive/huggingface_models'

# The model that you want to train from the Hugging Face hub
model_id = "georgesung/llama2_7b_chat_uncensored"

# The instruction dataset to use
dataset_name = "norygano/Ganymede"

# Fine-tuned model name
new_model = "llama2_7b_chat_Ganymede"

# Constants
model_name = model_id.split('/')[-1]

################################################################################
# QLoRA parameters
################################################################################

# LoRA attention dimension
lora_r = 12

# Alpha parameter for LoRA scaling
lora_alpha = 32

# Dropout probability for LoRA layers
lora_dropout = 0.5

################################################################################
# bitsandbytes parameters
################################################################################

# Activate 4-bit precision base model loading
use_4bit = True

# Compute dtype for 4-bit base models
bnb_4bit_compute_dtype = "float16"

# Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4"

# Activate nested quantization for 4-bit base models (double quantization)
use_nested_quant = True

################################################################################
# TrainingArguments parameters
################################################################################

# Output directory where the model predictions and checkpoints will be stored
output_dir = "./results"

# Number of training epochs
num_train_epochs = 6

# Enable fp16/bf16 training (set bf16 to True with an A100)
fp16 = False
bf16 = False

# Batch size per GPU for training
per_device_train_batch_size = 4

# Batch size per GPU for evaluation
per_device_eval_batch_size = 4

# Number of update steps to accumulate the gradients for
gradient_accumulation_steps = 1

# Enable gradient checkpointing
gradient_checkpointing = True

# Maximum gradient normal (gradient clipping)
max_grad_norm = 0.3

# Initial learning rate (AdamW optimizer)
learning_rate = 2e-4

# Weight decay to apply to all layers except bias/LayerNorm weights
weight_decay = 0.001

# Optimizer to use
optim = "paged_adamw_32bit"

# Learning rate schedule
lr_scheduler_type = "cosine"

# Number of training steps (overrides num_train_epochs)
max_steps = -1

# Ratio of steps for a linear warmup (from 0 to learning rate)
warmup_ratio = 0.03

# Group sequences into batches with same length
# Saves memory and speeds up training considerably
group_by_length = True

# Save checkpoint every X updates steps
save_steps = 0

# Log every X updates steps
logging_steps = 5

################################################################################
# SFT parameters
################################################################################

# Maximum sequence length to use
max_seq_length = None

# Pack multiple short examples in the same input sequence to increase efficiency
packing = False

# Load the entire model on the GPU 0
device_map = {"": 0}

#@title Train (Single)
import gc


# Load your dataset
torch.autograd.set_detect_anomaly(True)


dataset = load_dataset(dataset_name, split="train")
# Function to duplicate entries in the dataset
def duplicate_entries(dataset, duplication_factor):
    duplicated_datasets = [dataset for _ in range(duplication_factor)]
    concatenated_dataset = concatenate_datasets(duplicated_datasets)
    return concatenated_dataset.shuffle(seed=42)  # Shuffle to mix the entries

# Increase the weight of the dataset by duplicating its entries
dataset = duplicate_entries(dataset, duplication_factor=3)

# Number of training epochs
num_train_epochs = 5

# Dropout probability for LoRA layers
lora_dropout = 0.05

# Assuming `model` is your model variable
if 'model' in locals():
    del model
    torch.cuda.empty_cache()
    gc.collect()

#model = AutoModelForCausalLM.from_pretrained(model_name)

def preprocess_dataset(dataset, tokenizer):
    def process_entry(example):
        text_entry = example["text"]
        #print("Processing entry:", text_entry)

        # Split the entire entry by "### HUMAN:" to process each exchange separately
        # Skip the first split part if it's empty (which it will be if the text starts with "### HUMAN:")
        exchanges = [exchange for exchange in text_entry.split("### HUMAN:") if exchange.strip()]

        formatted_exchanges = []

        for exchange in exchanges:
            # Ensure "### RESPONSE:" is properly included in the split
            if "### RESPONSE:" in exchange:
                # Insert the eos_token right after the response
                parts = exchange.split("### RESPONSE:")
                if len(parts) == 2:
                    human_part, response_part = parts
                    # Add back the "### HUMAN:" and "### RESPONSE:" with eos_token at the end of the response
                    formatted_exchange = f"### HUMAN:{human_part}### RESPONSE:{response_part.strip()}{tokenizer.eos_token}"
                    formatted_exchanges.append(formatted_exchange)
                else:
                    # Log if the structure within the exchange is unexpected
                    print("Unexpected exchange structure:", exchange)
            else:
                print("No '### RESPONSE:' found in exchange:", exchange)

        # Reassemble the formatted exchanges into a single text entry
        formatted_text = "\n\n".join(formatted_exchanges)

        return {"text": formatted_text}

    # Apply the preprocessing function to each entry
    dataset = dataset.map(process_entry, batched=False)

    # Tokenize the preprocessed text entries
    dataset = dataset.map(lambda examples: tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512), batched=True)

    return dataset

# Load tokenizer and model with QLoRA configuration
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)


bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)


# Check GPU compatibility with bfloat16
if compute_dtype == torch.float16 and use_4bit:
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("=" * 80)
        print("Your GPU supports bfloat16: accelerate training with bf16=True")
        print("=" * 80)

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=None,
    device_map=device_map
)
model.config.use_cache = False
model.config.pretraining_tp = 1

# Load LLaMA tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False, legacy=False, truncation=True, add_special_tokens=True)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model.resize_token_embeddings(len(tokenizer))
tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training

# Preprocess the dataset
dataset = preprocess_dataset(dataset, tokenizer)

# Initialize data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, return_tensors="pt", mlm=False)

# Load LoRA configuration
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
)

# Set training parameters
training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    fp16=fp16,
    bf16=bf16,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type,
    report_to="tensorboard"
)

# Set supervised fine-tuning parameters
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    data_collator=data_collator,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=packing,
)

# Train model
trainer.train()

# Save trained model
trainer.model.save_pretrained(new_model)

# Ignore warnings
prompt = "Who are you?"
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)

# Adjust the prompt format
formatted_prompt = f"### HUMAN:\n{prompt}\n\n### RESPONSE: \n"

# Use the formatted prompt
result = pipe(formatted_prompt)

# Print the generated text, assuming the model appends its response after "### RESPONSE: \n"
print(result[0]['generated_text'])


#@title Prompt (Single)
# Ignore warnings
#logging.set_verbosity(logging.CRITICAL)

prompt = "You're cute!"
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)

# Adjust the prompt format
formatted_prompt = f"### HUMAN:\n{prompt}\n\n### RESPONSE: \n"

# Use the formatted prompt
result = pipe(formatted_prompt)

# Print the generated text, assuming the model appends its response after "### RESPONSE: \n"
print(result[0]['generated_text'])

#@title Reload model in FP16 and merge it with LoRA weights
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map=device_map,
)
model = PeftModel.from_pretrained(base_model, new_model)
model = model.merge_and_unload()

# Reload tokenizer to save it
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model.resize_token_embeddings(len(tokenizer))
tokenizer.padding_side = "right"

#@title Push
import locale
locale.getpreferredencoding = lambda: "UTF-8"

model.push_to_hub(new_model, use_temp_dir=False)
tokenizer.push_to_hub(new_model, use_temp_dir=False)

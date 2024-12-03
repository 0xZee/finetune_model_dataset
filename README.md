# finetune_model_dataset
Fine-Tunning Model `meta-llama/Llama-3.2-3B` on local Dataset using : LoRA, QLoRA TLR and PEFT..

Hardware Requirements : 
```
ALMALINUX VM
2 vCPU
16 Go RAM
```

Requirements : 
```
pip install torch transformers datasets pymongo peft accelerate bitsandbytes wandb
pip install trl  # for TLR
```

## Parameter-Efficient Fine-Tuning (PEFT) with 8-bit quantization

```python
# Requirements installation
!pip install transformers datasets torch mongodb bitsandbytes accelerate peft

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import prepare_model_for_int8_training, LoraConfig, get_peft_model
from datasets import Dataset
from pymongo import MongoClient

# Connect to MongoDB and prepare dataset (same as above)
def get_data_from_mongodb():
    client = MongoClient('mongodb://localhost:27017/')
    db = client['your_database']
    collection = db['list_MEP']
    
    data = list(collection.find({}))
    texts = [doc['text'] for doc in data]
    return texts

texts = get_data_from_mongodb()
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")

# Load model in 8-bit
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B",
    load_in_8bit=True,
    device_map="auto",
)

# Prepare model for 8-bit training
model = prepare_model_for_int8_training(model)

# Configure PEFT
peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, peft_config)

# Training arguments
training_args = TrainingArguments(
    output_dir="./peft_llama3-2",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    save_steps=100,
    logging_steps=10,
    max_grad_norm=0.3,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

trainer.train()
```

## Transformers Reinforcement Learning (TRL) - 4-bit Quantization

- TLR Provides an alternative supervised fine-tuning approach
- Scripts use 4-bit quantization to reduce memory footprint
- Batch sizes are kept small (2-4)
- Gradient accumulation helps simulate larger batch sizes

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTrainer, DataCollatorForCompletionOnlyLM
from datasets import Dataset
from pymongo import MongoClient

def load_data_from_mongodb():
    client = MongoClient('mongodb://localhost:27017/')
    db = client['your_database']
    collection = db['list_MEP']
    
    # Fetch data from MongoDB
    data = list(collection.find())
    
    # Prepare data for fine-tuning
    texts = [item['text'] for item in data]
    
    return texts

def prepare_dataset(texts):
    dataset = Dataset.from_dict({'text': texts})
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
    tokenizer.pad_token = tokenizer.eos_token
    
    return dataset, tokenizer

def fine_tune_with_trl():
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-3B", 
        device_map='auto', 
        load_in_4bit=True
    )
    
    # Prepare data
    texts = load_data_from_mongodb()
    dataset, tokenizer = prepare_dataset(texts)
    
    # Data collator for completion
    data_collator = DataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer, 
        mlm=False
    )
    
    # Supervised Fine-Tuning Trainer
    trainer = SFTrainer(
        model=model,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=512,
        tokenizer=tokenizer,
        data_collator=data_collator,
        args=TrainingArguments(
            output_dir='./trl_results',
            num_train_epochs=3,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            logging_dir='./logs'
        )
    )
    
    # Train
    trainer.train()
    
    # Save model
    trainer.save_model('./fine_tuned_trl_llama3-2')

# Execute fine-tuning
fine_tune_with_trl()
```

## LoRA (Low-Rank Adaptation) Training - Most memory efficient

```python
# Requirements installation
!pip install transformers datasets torch mongodb bitsandbytes accelerate loralib

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
from pymongo import MongoClient

# Connect to MongoDB and prepare dataset
def get_data_from_mongodb():
    client = MongoClient('mongodb://localhost:27017/')
    db = client['your_database']
    collection = db['list_MEP']
    
    data = list(collection.find({}))
    texts = [doc['text'] for doc in data]  # Adjust field name as per your schema
    return texts

# Prepare dataset
texts = get_data_from_mongodb()
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")

def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)

dataset = Dataset.from_dict({"text": texts})
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B",
    torch_dtype=torch.float16,
    device_map="auto"
)

# Configure LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)

# Training arguments
training_args = TrainingArguments(
    output_dir="./lora_llama3-2",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    save_steps=100,
    logging_steps=10,
    max_grad_norm=0.3,
)

# Start training
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

trainer.train()
```


## QLoRA (Quantized LoRA) - Best balance of efficiency and performance

```python
# Requirements installation
!pip install transformers datasets torch mongodb bitsandbytes accelerate loralib

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from datasets import Dataset
from pymongo import MongoClient

# Connect to MongoDB and prepare dataset (same as above)
def get_data_from_mongodb():
    client = MongoClient('mongodb://localhost:27017/')
    db = client['your_database']
    collection = db['list_MEP']
    
    data = list(collection.find({}))
    texts = [doc['text'] for doc in data]
    return texts

texts = get_data_from_mongodb()
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3Bb")

# Load model in 4-bit
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B",
    load_in_4bit=True,
    device_map="auto",
    torch_dtype=torch.float16,
)

# Prepare model for QLoRA
model = prepare_model_for_kbit_training(model)

# Configure QLoRA
qlora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, qlora_config)

# Training arguments
training_args = TrainingArguments(
    output_dir="./qlora_llama3-2",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    save_steps=100,
    logging_steps=10,
    max_grad_norm=0.3,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

trainer.train()
```


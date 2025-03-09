from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

model_name = "microsoft/DialoGPT" 
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Cargar dataset
from datasets import load_dataset
dataset = load_dataset("json", data_files="dataset.json")

# Definir argumentos
training_args = TrainingArguments(
    output_dir="./dialoGPT_finetuned",
    per_device_train_batch_size=4,
    num_train_epochs=10,
    save_steps=1000,
    save_total_limit=2,
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
)

trainer.train() 

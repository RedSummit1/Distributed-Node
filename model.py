from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
import pandas as pd
import torch
import logging
import pickle
import os

logging.getLogger("transformers").setLevel(logging.ERROR)


class Model:
    def __init__(self, model):
        # Load model in 8-bit
        self.model = AutoModelForCausalLM.from_pretrained(
            model, load_in_8bit=True, device_map="auto", torch_dtype=torch.float16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

    def prepare_for_training(self):
        # Prepare the model for k-bit training
        self.model = prepare_model_for_kbit_training(self.model)

        # Define LoRA Config
        # Get the layer names first
        target_modules = [
            name
            for name, _ in self.model.named_modules()
            if "q_proj" in name or "v_proj" in name
        ]
        print(f"Found target modules: {target_modules}")

        lora_config = LoraConfig(
            r=16,  # rank
            lora_alpha=32,  # alpha scaling
            target_modules=["q_proj", "v_proj"],  # for TinyLLama attention layers
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        # Get PEFT model
        self.model = get_peft_model(self.model, lora_config)
        print_trainable_parameters(self.model)

    def chat(self, user_input):
        prompt = f"<|user|>{user_input}<|assistant|>"
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=200,
                temperature=0.1,
                do_sample=True,
                top_p=0.2,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.split("<|assistant|>")[-1].strip()

    def prepare_dataset(self, csv_path):
        df = pd.read_csv(csv_path)
        print(f"Loaded CSV with columns: {df.columns.tolist()}")
        print(f"Number of rows: {len(df)}")

        data = []
        for _, row in df.iterrows():
            row_text = " ".join(str(value) for value in row if pd.notna(value))
            try:
                encoded = self.tokenizer(
                    row_text,
                    truncation=True,
                    max_length=256,
                    padding="max_length",
                    return_tensors="pt",
                )

                data.append(
                    {
                        "input_ids": encoded["input_ids"][0],
                        "attention_mask": encoded["attention_mask"][0],
                        "labels": encoded["input_ids"][0].clone(),
                    }
                )

                if len(data) % 1000 == 0:
                    print(f"Processed {len(data)} rows...")

            except Exception as e:
                print(f"Skipping row due to encoding error: {e}")
                continue

        if not data:
            raise ValueError("No valid data was processed from the CSV")

        print(f"Successfully processed {len(data)} rows")
        return Dataset.from_list(data)

    def fine_tune(self, csv_path, output_dir="fine_tuned_model", num_epochs=3):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Prepare model with LoRA
        self.prepare_for_training()

        print("Starting dataset preparation...")
        dataset = self.prepare_dataset(csv_path)
        print("Dataset preparation complete!")

        print("Setting up training configuration...")
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False
        )

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=16,
            save_steps=500,
            logging_steps=50,
            learning_rate=1e-4,
            warmup_steps=100,
            save_total_limit=1,
            remove_unused_columns=False,
            fp16=True,
            optim="paged_adamw_8bit",
            max_grad_norm=0.3,
            gradient_checkpointing=True,
            report_to="none",
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
        )

        print("Starting training...")
        trainer.train()

        print("Saving LoRA adapter...")
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        print("Merging LoRA weights with base model...")
        # Create a directory for the merged model
        merged_dir = os.path.join(output_dir, "merged")
        os.makedirs(merged_dir, exist_ok=True)

        # Merge weights and save
        merged_model = self.model.merge_and_unload()

        print("Saving merged model...")
        merged_model.save_pretrained(
            merged_dir,
            safe_serialization=True,  # Use safetensors format
            max_shard_size="10GB",  # Shard the model if it's too large
        )
        self.tokenizer.save_pretrained(merged_dir)

        # Update the model instance with the merged model
        print("Updating model instance with merged model...")
        del self.model  # Free up memory
        torch.cuda.empty_cache()  # Clear CUDA cache

        # Load the merged model as AutoModelForCausalLM
        self.model = AutoModelForCausalLM.from_pretrained(
            merged_dir, device_map="auto", torch_dtype=torch.float16
        )
        # Update tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(merged_dir)

        print(f"Model saved to {output_dir}")
        print(f"Merged model saved to {merged_dir}")
        print("Model instance successfully updated with merged model")


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


# Example usage
# model = Model("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
# model.fine_tune("your_data.csv", output_dir="my_finetuned_model", num_epochs=3)
#
# while True:
#   user_text = input("You: ")
#   if user_text.lower() == "quit":
#       break
#   response = model.chat(user_text)
#   print("Bot:", response)

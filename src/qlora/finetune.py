from data import get_dataset, get_tokenizer
from model import get_model
from peft import get_peft_config
from transformers import TrainingArguments
from trl import SFTTrainer



class FinetuneLLM():
    def __init__(self, args) -> None:
        self.args = args
        self.train_ds, self.val_ds = get_dataset(args)
        self.train_ds_map = self.train_ds.map(
            lambda x: f"[INST] \n {x['role']}: {x['text']} \n [/INST]",
        )
        self.val_ds_map = self.val_ds.map(
            lambda x: f"[INST] \n {x['role']}: {x['text']} \n [/INST]",
        )
        self.tokenizer = get_tokenizer(args)
        self.model = get_model(args)
        self.peft_config = get_peft_config(args)
        
    def train(self):
        training_args = TrainingArguments(
            output_dir=self.args.output_dir,
            num_train_epochs=self.args.num_train_epochs,
            per_device_train_batch_size=self.args.per_device_train_batch_size,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            optim=self.args.optim,
            save_steps=self.args.save_steps,
            logging_steps=self.args.logging_steps,
            learning_rate=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
            fp16=self.args.fp16,
            bf16=self.args.bf16,
            max_grad_norm=self.args.max_grad_norm,
            max_steps=self.args.max_steps,
            warmup_ratio=self.args.warmup_ratio,
            group_by_length=self.args.group_by_length,
            lr_scheduler_type=self.args.lr_scheduler_type,
            report_to="all",
            evaluation_strategy="steps",
            eval_steps=5  # Evaluate every 20 steps
        )
        trainer = SFTTrainer(
            model=self.model,
            train_dataset=self.train_ds_map,
            eval_dataset=self.val_ds_map,  # Pass validation dataset here
            peft_config=self.peft_config,
            dataset_text_field="text",
            max_seq_length=self.args.max_seq_length,
            tokenizer=self.tokenizer,
            args=training_args,
            packing=packing,
        )



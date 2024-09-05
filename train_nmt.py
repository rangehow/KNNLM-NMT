from functools import partial
import json
import sys
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoConfig,
    Trainer,
    TrainingArguments,
    DataCollator,
)
import datasets
from argparse import ArgumentParser
import torch
from loguru import logger
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset,DataLoader

class MyCollator:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, examples):

        input_ids = self.tokenizer([example["translation"]["de"] for example in examples], 
                                   padding=True, 
                                   return_tensors="pt", 
                                   add_special_tokens=True)
        
        decoder_input_ids = self.tokenizer([tokenizer.bos_token+example["translation"]["en"] for example in examples], 
                                   padding=True, 
                                   return_tensors="pt", 
                                   add_special_tokens=False)
        
        labels = [self.tokenizer.encode(example["translation"]["en"],add_special_tokens=False)+[128001] for example in examples]
        max_len = decoder_input_ids.input_ids.shape[-1]

        labels = torch.tensor(
            [ label+[-100] * (max_len - len(label))  for label in labels]
        )
        
        
        return {
            "input_ids": input_ids.input_ids,
            "attention_mask": input_ids.attention_mask,
            "decoder_input_ids": decoder_input_ids.input_ids,
            "decoder_attention_mask": decoder_input_ids.attention_mask,
            "labels": labels,
        }
    



model = AutoModelForSeq2SeqLM.from_pretrained("/mnt/rangehow/models/llama_nmt")
tokenizer = AutoTokenizer.from_pretrained(
    "/mnt/rangehow/models/Meta-Llama-3.1-8B-Instruct"
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token # instruct的這個是128009，base的是128001


my_collator = MyCollator(tokenizer)



dataset = datasets.load_dataset("wmt/wmt19", "de-en")


class TokenBatchDataset(Dataset):
    def __init__(self, dataset: datasets.Dataset, tokenizer):
        self.en=[]
        self.de=[]
        self.lengths=[]
        for instance in dataset["train"]:
            en_item=tokenizer.encode(instance["translation"]["en"], add_special_tokens=False)+[128001]
            de_item=[128001]+tokenizer.encode(instance["translation"]["de"], add_special_tokens=False)
            self.en.append( en_item)
            self.de.append( de_item)
            self.lengths.append(len(en_item)+len(de_item))
        self.sorted_indices = sorted(
            range(len(self.texts)), key=lambda i: self.lengths[i], reverse=True
        )

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {'translation':{'en':self.en[idx],'de':self.de[idx]}}

train_dataset= TokenBatchDataset(dataset)


logger.debug(f"数据集总量是{len(dataset["train"])}")
from LLaMAEmbeddings import DynamicBatchSampler
class MyTrainer(Trainer):
    
    def get_train_dataloader(self):
        import pdb
        pdb.set_trace()
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        
        sampler=DynamicBatchSampler(train_dataset,1024)
        return self.accelerator.prepare(DataLoader(train_dataset))

trainer = Trainer(
    model=model,
    args=TrainingArguments(
        # optim="adamw_apex_fused",
        output_dir="/mnt/rangehow/models/llama_nmt_trained",
        overwrite_output_dir=True,
        # learning_rate=args.learning_rate,  # 学习率
        per_device_train_batch_size=8,  # 每个设备的训练批量大小
        per_device_eval_batch_size=8,
        # auto_find_batch_size=True,
        num_train_epochs=30,  # 训练的轮次
        # weight_decay=args.weight_decay,
        eval_strategy="epoch",
        gradient_accumulation_steps=32,
        learning_rate=5e-4,
        lr_scheduler_type="cosine",
        dataloader_num_workers=0,
        fp16=True,
        logging_steps=1,
        remove_unused_columns=False,
        save_strategy="epoch",
        warmup_ratio=0.05,
        save_total_limit=3,
        # torch_compile=True,
        # torch_compile_mode="reduce-overhead",
        # torch_compile_backend="inductor",
        # torchdynamo="inductor",
        eval_on_start=True,
        # ddp_find_unused_parameters =False,
        # label_smoothing_factor=args.label_smoothing_factor,
    ),
    train_dataset=train_dataset,
    eval_dataset=dataset["validation"],
    tokenizer=tokenizer,
    data_collator=my_collator,
    
)


trainer.train()


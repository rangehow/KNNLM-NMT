from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    BertModel,
    AutoModelForSeq2SeqLM,
    FSMTTokenizer,
    FSMTForConditionalGeneration,
    LlamaForCausalLM,
)
import datasets
import transformers
import numpy as np
from torch.utils.data import DataLoader, Dataset
from argparse import ArgumentParser
import torch
from tqdm import tqdm
from functools import partial
import logging
from knnlm import KNNWrapper
from LLaMAEmbeddings import LLaMAEmbedding
import config
import time
logger = logging.getLogger(__name__)

class mydataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)



def my_collate(batch, tokenizer,domain):
    inputs = tokenizer(batch, truncation=True, return_tensors="pt", padding=True)
    return inputs

from torch.utils.data import Sampler


class MaxTokenSampler(Sampler):
    def __init__(self, data) -> None:
        self.data = data
        self.max_token = 1
        self.idx = 0
        self.length = 0

    def __len__(self) -> int:
        return self.length

    def __iter__(self):
        batch = []
        curLen = 0
        for i in range(self.idx, len(self.data)):
            curLen += tokenizer(self.data[i], return_length=True)["length"][0]
            # if i<5:
            # print(i,batch,curLen,self.idx)
            if len(batch) == 0 or curLen < self.max_token:
                batch.append(i)
            else:
                self.idx = i
                assert len(batch) != 0, "第i个迭代返回空batch idx了?"
                yield batch
                curLen = tokenizer(self.data[i], return_length=True)["length"][0]
                batch = [i]
        # print('?',self.idx,len(self.data),batch)
        yield list(range(self.idx, len(self.data)))



def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--domain", type=str, help="指定需要处理的json文件",)
    parser.add_argument("--lmbda", type=float, default=0.25,)
    parser.add_argument("--temperature", type=int, default=10,)
    parser.add_argument("--output", type=str, )
    parser.add_argument('--vdb_type',type=str)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    with torch.inference_mode():
        
        tokenizer = AutoTokenizer.from_pretrained(config.nmt_path,use_fast=True,add_eos_token=True, add_bos_token=False,padding_side ="left")
        tokenizer.pad_token_id=0
        llama_tokenizer = AutoTokenizer.from_pretrained(
            config.llama_path[args.vdb_type], use_fast=True, padding_side="left", 
        )
        llama_tokenizer.pad_token_id=0
        # 这里不用和embedding一样的写法是因为generation_config里写的这个，我要用generation

        # tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        nmt_model = AutoModelForSeq2SeqLM.from_pretrained(
            config.nmt_path,
            torch_dtype='auto',
        )
        
        nmt_model.cuda()
        nmt_model.eval()
        model = AutoModelForCausalLM.from_pretrained(
            config.llama_path[args.vdb_type],
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        model.cuda()
        model.eval()

        file = args.domain

        knn_model = KNNWrapper(
            dstore_dir=f"{config.vdb_path}/{file}/{config.vdb[args.vdb_type]}", embeddings=LLaMAEmbedding,
            knn_temp=args.temperature,lmbda=args.lmbda,vdb_type=args.vdb_type
        )
        
        def nmt_knn(self,domain,vdb_type):
            embeddings = self.embeddings(model=self.model,prompt=prompt,vdb_type=vdb_type)
            self.retriever = knnretriver(index_dir=self.dstore_dir, embeddings=embeddings)
            
        
        knn_model.setup_faiss = nmt_knn
        knn_model.break_into(nmt_model,assistant_model=model,domain=args.domain,vdb_type=args.vdb_type)
        with open(f"{config.vdb_path}/{file}/test.de") as ff:
            with open(
                f"{config.vdb_path}/{file}/{args.output}", "w"
            ) as o:
                print(f"{config.vdb_path}/{file}/test.de")
                inputs = ff.readlines()
                sampler = MaxTokenSampler(inputs)
                cnt = 0
                empty = []
                for s in sampler:
                    cnt += 1
                    empty.extend(s)
                # print(empty,len(empty))
                sampler.length = cnt
                sampler.idx = 0
                # print('文件总长度是',len(inputs))
                data = mydataset(inputs)
                dataloader = DataLoader(
                    data,
                    batch_sampler=sampler,
                    collate_fn=partial(my_collate, tokenizer=tokenizer,domain=args.domain),
                    pin_memory=True,
                    num_workers=10,
                )

                # result_list = []
                cnt = 0
                for i,d in enumerate(tqdm(dataloader)):

                    start=time.time()
                    d.to("cuda")
                    inputs_for_llama.to("cuda")

                    while 
                    knn_model.update_encoder_input(inputs_for_llama)

                    preds = nmt_model.generate(
                        labels=None, input_ids=d["input_ids"], use_cache=True
                    )
                    # print(preds)
                    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
                    decoded_preds = [pred.strip() for pred in decoded_preds]

                    print(decoded_preds)
                    end=time.time()

                    for r in decoded_preds:
                        o.write(r.replace('\n','\\n') + "\n")
                    o.flush()

                print(f'{file}推断完成了！')

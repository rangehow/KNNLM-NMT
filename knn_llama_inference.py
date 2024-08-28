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
# log_level = "INFO"
# logger.setLevel(log_level)
# datasets.utils.logging.set_verbosity(log_level)
# transformers.utils.logging.set_verbosity(log_level)
# transformers.utils.logging.enable_default_handler()
# transformers.utils.logging.enable_explicit_format()


# torch.multiprocessing.set_start_method('spawn')
class mydataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def my_collate(batch, tokenizer):
    start=time.time()
    # prompt='translate the above sentence to English, and only return the content translated. no explanation.'
    inputs = [
        f"Translate this from Deutsch to English:\nDeutsch:{b.strip()}\nEnglish:"
        for b in batch
    ]
    # print(inputs)
    inputs = tokenizer(inputs, truncation=True, return_tensors="pt", padding=True)
    end=time.time()
    # print('一个collate的耗时',end-start)
    return inputs


from torch.utils.data import Sampler


tokenizer = AutoTokenizer.from_pretrained(
    config.llama_path, use_fast=True, padding_side="left"
)
# 这里不用和embedding一样的写法是因为generation_config里写的这个，我要用generation
tokenizer.pad_token_id=0


class MaxTokenSampler(Sampler):
    def __init__(self, data) -> None:
        self.data = data
        self.max_token = 128
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


def clean_outputstring(output, key_word="\nEnglish:", logger=logger):
    try:
        out = output.split("\nEnglish:")[1].split("\n")
        if out[0].strip() != "":
            return out[0].strip()
        elif out[1].strip() != "":
            ## If there is an EOL directly after the suffix, ignore it
            logger.info(
                f"Detect empty output, we ignore it and move to next EOL: {out[1].strip()}"
            )
            return out[1].strip()
        else:
            logger.info(
                f"Detect empty output AGAIN, we ignore it and move to next EOL: {out[2].strip()}"
            )
            return out[2].strip()
    except:
        logger.info(
            f"Can not recover the translation by moving to the next EOL.. Trying move to the next suffix"
        )

    try:
        return output.split(key_word)[2].split("\n")[0].strip()
    except:
        logger.info(
            f"Can not solve the edge case, recover the translation to empty string! The output is {output}"
        )
        return ""


def tokenize_function(examples):
    output = tokenizer(examples["text"])
    return output


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--domain", type=str, help="指定需要处理的json文件",)
    parser.add_argument("--lmbda", type=float, default=0.25,)
    parser.add_argument("--temperature", type=int, default=10,)
    parser.add_argument("--output", type=str, help="指定需要处理的json文件",)
    return parser.parse_args()

if __name__ == "__main__":
    
    with torch.inference_mode():
        model = AutoModelForCausalLM.from_pretrained(
            config.llama_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        model.cuda()

        # # print(model)
        # knn_model=KNNWrapper(dstore_dir=f'{config.vdb_path}/koran/knn_vdb',embeddings=LLaMAEmbedding)
        # knn_model.break_into(model)

        # print('最后的输出',(model.generate(**inputs,use_cache=True)))
        # exit()
        # print(model.generation_config)
        # print(model.hf_device_map)


        args = parse_args()
        file = args.domain

        # knn_model=KNNWrapper(dstore_dir=f'{config.vdb_path}/{folder}/knn_vdb')
        knn_model = KNNWrapper(
            dstore_dir=f"{config.vdb_path}/{file}/{config.vdb[args.vdb_type]}", embeddings=LLaMAEmbedding,
            lmbda=args.lmbda,knn_temp=args.temperature
        )
        knn_model.break_into(model)
        with open(f"{config.vdb_path}/{file}/test.de") as ff:
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
                collate_fn=partial(my_collate, tokenizer=tokenizer),
                pin_memory=True,
                num_workers=10,
            )

            result_list = []
            cnt = 0
            for d in tqdm(dataloader):
                start=time.time()
                d.to("cuda")

                preds = model.generate(
                    labels=None, input_ids=d["input_ids"], use_cache=True
                ).to("cpu")

                if int(torch.cuda.current_device()) == 0:
                    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
                    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
                    decoded_preds = [pred.strip() for pred in decoded_preds]

                result = list(map(clean_outputstring, decoded_preds))

                print(result)
                result_list.extend(result)
                end=time.time()
                # print('一个batch的总耗时',end-start)
            # print()
            with open(
                f"{config.vdb_path}/{file}/{args.output}.en", "w"
            ) as o:
                for r in result_list:
                    o.write(r.replace('\n','\\n') + "\n")

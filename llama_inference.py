from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    BertModel,
    AutoModelForSeq2SeqLM,
    FSMTTokenizer,
    FSMTForConditionalGeneration,
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
import config

logger = logging.getLogger(__name__)
log_level = "ERROR"
logger.setLevel(log_level)
datasets.utils.logging.set_verbosity(log_level)
transformers.utils.logging.set_verbosity(log_level)
transformers.utils.logging.enable_default_handler()
transformers.utils.logging.enable_explicit_format()


# torch.multiprocessing.set_start_method('spawn')
class mydataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def my_collate(batch, tokenizer):
    # prompt='translate the above sentence to English, and only return the content translated. no explanation.'
    inputs = [
        f"Translate this from Deutsch to English:\nDeutsch:{b.strip()}\nEnglish:"
        for b in batch
    ]
    # print(inputs)
    inputs = tokenizer(inputs, truncation=True, return_tensors="pt", padding=True)

    return inputs


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
    parser.add_argument("--output", type=str, )
    return parser.parse_args()


with torch.inference_mode():
    # tokenizer = AutoTokenizer.from_pretrained("/home/adu/Llama_dict_NMT_hf", use_fast=True,padding_side='left')
    # 这里不用和embedding一样的写法是因为generation_config里写的这个，我要用generation
    
    tokenizer = AutoTokenizer.from_pretrained("/home/adu/Llama_dict_NMT_hf")
    tokenizer.pad_token_id = 0
    model = AutoModelForSeq2SeqLM.from_pretrained(
        "/home/adu/Llama_dict_NMT_hf",
        torch_dtype='auto',
        # device_map='auto'
    ).to_bettertransformer()
    model.cuda()
    # print(model.generation_config)
    # print(model.hf_device_map)

    # model.resize_token_embeddings(len(tokenizer))
    args=parse_args()

    file=args.domain
    with open(f"{config.vdb_path}/{file}/test.de") as ff:
        print(f"{config.vdb_path}/{file}/test.de")
        inputs = ff.readlines()
        data = mydataset(inputs)
        dataloader = DataLoader(
            data,
            64,
            collate_fn=partial(my_collate, tokenizer=tokenizer),
            pin_memory=True,
            num_workers=10,
        )
        result_list = []
        for d in tqdm(dataloader):
            d.to("cuda")
            # print(d)
            # d.to(model.device)
            preds = model.generate(
                labels=None, input_ids=d["input_ids"], use_cache=True,max_new_tokens=128
            ).to('cpu')
            # print(generated_tokens)
            if int(torch.cuda.current_device()) == 0:
                preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
                # print(preds)
                # print(preds.shape)
                decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

                # Some simple post-processing
                decoded_preds = [pred.strip() for pred in decoded_preds]
            # print(decoded_preds)
            # result = list(
            #     map(
            #         clean_outputstring,
            #         decoded_preds
            #     )
            # )
            
            # print(result)
            result_list.extend(result)

        with open(f"{config.vdb_path}/{file}/pure_nmt.en", "w") as o:
            for r in result_list:
                o.write(r + "\n")


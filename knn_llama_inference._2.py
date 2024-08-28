from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    BertModel,
    AutoModelForSeq2SeqLM,
    FSMTTokenizer,
    FSMTForConditionalGeneration,
    LlamaForCausalLM
)
import datasets
import transformers
import numpy as np
from torch.utils.data import DataLoader, Dataset

import torch
from tqdm import tqdm
from functools import partial
import logging
from knnlm import KNNWrapper
from LLaMAEmbeddings import LLaMAEmbedding


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
    # prompt='translate the above sentence to English, and only return the content translated. no explanation.'
    inputs = [
        f"Translate this from Deutsch to English:\nDeutsch:{b.strip()}\nEnglish:"
        for b in batch
    ]
    # print(inputs)
    inputs = tokenizer(inputs, truncation=True, return_tensors="pt", padding=True).to(
        "cuda"
    )

    return inputs

from torch.utils.data import Sampler



tokenizer = AutoTokenizer.from_pretrained("/data/lxy/abu/llama2-7b", use_fast=True,padding_side='left')
# 这里不用和embedding一样的写法是因为generation_config里写的这个，我要用generation
tokenizer.add_special_tokens({'pad_token': '[PAD]'})




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


with torch.inference_mode():
    # print(tokenizer.batch_decode([[    1,  4103,  9632,   445,   515,  4493,   304,  4223, 29901,    13,
    #       2772,  2434, 29901, 29928,   812,  4454,   300,  5351,   313,  1217,
    #        305, 29897,   604, 19243, 29892,  2949,   762,   951,  1082,   553,
    #      15621,   264, 17412,   267,  3937,   563,  2949, 21131, 28253,  7369,
    #       1752, 29889,    13,    13, 24636, 29901, 11760,   366,   674,  2041,
    #        304,  1073,  1058,   526,   278,  2305,   310,   278,  3233,  2224,
    #        322,  1058,   526,  1492,   368,  1410,  2618, 29889,    13,     2]]))
    # prompt='Translate this from Deutsch to English:\nDeutsch:{de}\nEnglish:'.format_map({'de':'Dann werdet ihr (noch) erfahren, wer die Leute des ebenen Weges sind und wer rechtgeleitet ist.\n'})
    # print(prompt)
    # inputs=tokenizer(prompt,return_tensors='pt',padding=True).to('cuda')
    # print(inputs,inputs.input_ids.shape)
    model = AutoModelForCausalLM.from_pretrained(
        "/data/lxy/abu/llama2-7b",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    )
    model.cuda()

    # # print(model)
    # knn_model=KNNWrapper(dstore_dir=f'/data/lxy/abu/vdbdata/koran/knn_vdb',embeddings=LLaMAEmbedding)
    # knn_model.break_into(model)
    
    # print('最后的输出',(model.generate(**inputs,use_cache=True)))
    # exit()
    # print(model.generation_config)
    # print(model.hf_device_map)

    model.resize_token_embeddings(len(tokenizer))

    folder = ["koran"] # 
    for file in folder:
        # knn_model=KNNWrapper(dstore_dir=f'/data/lxy/abu/vdbdata/{folder}/knn_vdb')
        knn_model=KNNWrapper(dstore_dir=f'/data/lxy/abu/vdbdata/{file}/knn_vdb',embeddings=LLaMAEmbedding)
        knn_model.break_into(model)
        with open(f"/data/lxy/abu/vdbdata/{file}/test.de") as ff:
            print(f"/data/lxy/abu/vdbdata/{file}/test.de")
            inputs = ff.readlines()
            print('文件总长度是',len(inputs))
            data = mydataset(inputs)
            dataloader = DataLoader(
                data,
                3,
                collate_fn=partial(my_collate, tokenizer=tokenizer),
            )

            result_list = []
            for d in tqdm(dataloader):
                d.to("cuda")

                preds = model.generate(
                    labels=None, input_ids=d["input_ids"], use_cache=True
                ).to('cpu')

                if int(torch.cuda.current_device()) == 0:
                    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
                    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
                    decoded_preds = [pred.strip() for pred in decoded_preds]

                result = list(
                    map(
                        clean_outputstring,
                        decoded_preds
                    )
                )
                
                # print(result)
                result_list.extend(result)

            # print()
            with open(f"/data/lxy/abu/vdbdata/{file}/llama_knnlm_{file}_output.en", "w") as o:
                for r in result_list:
                    o.write(r + "\n")


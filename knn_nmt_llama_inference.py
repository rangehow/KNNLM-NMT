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


def llama_collate(batch, tokenizer):
    start=time.time()
    # prompt='translate the above sentence to English, and only return the content translated. no explanation.'
    prefix = "Translate this sentence from German into English and return the translation result only."
    shot = {
        "it": """\nGerman:Zeigt den aktuellen Wert der Feldvariable an.\nEnglish:Displays the current value of the field variable.\nGerman:In diesem Bereich wählen Sie die relativen Größen bezogen auf die Basisgröße.\nEnglish:In this section, you can determine the relative sizes for each type of element with reference to the base size.\nGerman:Geben Sie einen kurzen, beschreibenden Namen für die Schnittstelle ein.\nEnglish:Simply enter a short human-readable description for this device.""",
        "koran": """\nGerman:So führt Gott (im Gleichnis) das Wahre und das Falsche an.\nEnglish:This is how God determines truth and falsehood.\nGerman:Da kamen sie auf ihn zu geeilt.\nEnglish:So the people descended upon him.\nGerman:Wir begehren von euch weder Lohn noch Dank dafür.\nEnglish:We wish for no reward, nor thanks from you.""",
        "law": """\nGerman:Deshalb ist die Regelung von der Ausfuhrleistung abhängig.\nEnglish:In this regard, the scheme is contingent upon export performance.\nGerman:Das Mitglied setzt gleichzeitig den Rat von seinem Beschluß in Kenntnis.\nEnglish:That member shall simultaneously inform the Council of the action it has taken.\nGerman:Dies gilt auch für die vorgeschlagene Sicherheitsleistung.\nEnglish:The same shall apply as regards the security proposed.""",
        "medical": """\nGerman:Das Virus wurde zuerst inaktiviert (abgetötet), damit es keine Erkrankungen verursachen kann.\nEnglish:This may help to protect against the disease caused by the virus.\nGerman:Desirudin ist ein rekombinantes DNS-Produkt, das aus Hefezellen hergestellt wird.\nEnglish:Desirudin is a recombinant DNA product derived from yeast cells.\nGerman:Katzen erhalten eine intramuskuläre Injektion.\nEnglish:In cats, it is given by intramuscular injection.""",
    }
    postfix = "\nGerman:{de}\nEnglish:{en}"
    inputs = [
        f"Translate this sentence from German into English and return the translation result only.:\nDeutsch:{b}\nEnglish:"
        for b in batch
    ]
    # print(inputs)
    inputs = tokenizer(inputs, truncation=True, return_tensors="pt", padding=True)
    end=time.time()
    # print('一个collate的耗时',end-start)
    return inputs

def my_collate(batch, tokenizer,domain):
    start=time.time()
    # prompt='translate the above sentence to English, and only return the content translated. no explanation.'
    # print(inputs)
    inputs = tokenizer(batch, truncation=True, return_tensors="pt", padding=True)
    
    # -----------------------------
    prefix = "Translate this sentence from German into English and return the translation result only."
    shot = {
        "it": """\nGerman:Zeigt den aktuellen Wert der Feldvariable an.\nEnglish:Displays the current value of the field variable.\nGerman:In diesem Bereich wählen Sie die relativen Größen bezogen auf die Basisgröße.\nEnglish:In this section, you can determine the relative sizes for each type of element with reference to the base size.\nGerman:Geben Sie einen kurzen, beschreibenden Namen für die Schnittstelle ein.\nEnglish:Simply enter a short human-readable description for this device.""",
        "koran": """\nGerman:So führt Gott (im Gleichnis) das Wahre und das Falsche an.\nEnglish:This is how God determines truth and falsehood.\nGerman:Da kamen sie auf ihn zu geeilt.\nEnglish:So the people descended upon him.\nGerman:Wir begehren von euch weder Lohn noch Dank dafür.\nEnglish:We wish for no reward, nor thanks from you.""",
        "law": """\nGerman:Deshalb ist die Regelung von der Ausfuhrleistung abhängig.\nEnglish:In this regard, the scheme is contingent upon export performance.\nGerman:Das Mitglied setzt gleichzeitig den Rat von seinem Beschluß in Kenntnis.\nEnglish:That member shall simultaneously inform the Council of the action it has taken.\nGerman:Dies gilt auch für die vorgeschlagene Sicherheitsleistung.\nEnglish:The same shall apply as regards the security proposed.""",
        "medical": """\nGerman:Das Virus wurde zuerst inaktiviert (abgetötet), damit es keine Erkrankungen verursachen kann.\nEnglish:This may help to protect against the disease caused by the virus.\nGerman:Desirudin ist ein rekombinantes DNS-Produkt, das aus Hefezellen hergestellt wird.\nEnglish:Desirudin is a recombinant DNA product derived from yeast cells.\nGerman:Katzen erhalten eine intramuskuläre Injektion.\nEnglish:In cats, it is given by intramuscular injection.""",
    }
    postfix = "\nGerman:{de}\nEnglish:"
    # inputs_for_llama = [
    #     f"Translate this from Deutsch to English:\nDeutsch:{b.strip()}\nEnglish:"
    #     for b in batch
    # ]
    inputs_for_llama = [
        prefix+shot[domain]+postfix.format_map({'de':b.strip()})
        for b in batch
    ]
    inputs_for_llama = tokenizer(inputs_for_llama, truncation=True, return_tensors="pt", padding=True)
    # ---------------------------
    end=time.time()
    # print('一个collate的耗时',end-start)
    return inputs,inputs_for_llama

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

        # knn_model=KNNWrapper(dstore_dir=f'{config.vdb_path}/{folder}/knn_vdb')
        knn_model = KNNWrapper(
            dstore_dir=f"{config.vdb_path}/{file}/{config.vdb[args.vdb_type]}", embeddings=LLaMAEmbedding,
            knn_temp=args.temperature,lmbda=args.lmbda,vdb_type=args.vdb_type
        )
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
                for i,(d,inputs_for_llama) in enumerate(tqdm(dataloader)):

                    start=time.time()
                    d.to("cuda")
                    inputs_for_llama.to("cuda")

                    
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

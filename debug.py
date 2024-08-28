from transformers import AutoTokenizer, FSMTForConditionalGeneration

from torch.utils.data import DataLoader, Dataset
import torch
from tqdm import tqdm
from functools import partial


class mydataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def my_collate(batch, tokenizer):
    inputs = tokenizer(batch, truncation=True, return_tensors="pt", padding=True)
    return inputs


from torch.utils.data import Sampler





if __name__ == "__main__":
    with torch.inference_mode():
        model_name = "/data/lxy/abu/Llama_dict_NMT_hf"
        nmt_model = FSMTForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype="auto",
            low_cpu_mem_usage=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
            add_eos_token=True,
            add_bos_token=False,
            padding_side="left",
        )
        tokenizer.pad_token_id=0
        nmt_model.cuda()
        inputs = [
            "Äußerlich kann Levemir InnoLet durch Abwischen mit einem medizinischen Tupfer gereinigt werden."
        ]
        nmt_model.eval()
        data = mydataset(inputs)
        dataloader = DataLoader(
            data,
            collate_fn=partial(my_collate, tokenizer=tokenizer),
            pin_memory=True,
            num_workers=10,
        )
        cnt = 0
        for d in tqdm(dataloader):
            d.to("cuda")
            preds = nmt_model.generate(
                labels=None, input_ids=d["input_ids"], use_cache=True, num_beams=5
            )
            decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
            decoded_preds = [pred.strip() for pred in decoded_preds]

            print(decoded_preds)

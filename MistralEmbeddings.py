from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
import torch
import json
import logging
import warnings
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)
from torch.utils.data import Dataset, DataLoader
import copy


class documentDataset(Dataset):
    def __init__(self, document) -> None:
        super().__init__()
        self.data = document

    def __getitem__(self, index) -> Any:
        return self.data[index]

    def __len__(self):
        return len(self.data)


from torch.utils.data.sampler import Sampler


class MaxTokenSampler(Sampler):
    def __init__(self, data, tokenizer) -> None:
        self.data = data
        self.max_token = 256
        self.idx = 0
        self.length = 0
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return self.length

    def __iter__(self):
        batch = []
        curLen = 0
        for i in range(self.idx, len(self.data)):
            prompt = (
                "Translate this from Deutsch to English:\nDeutsch:{de}\nEnglish:{en}"
            )
            str = prompt.format_map(
                {"en": self.data[i]["en"], "de": self.data[i]["de"]}
            )
            # print(self.tokenizer.encode(str))
            curLen += self.tokenizer(str, return_length=True)["length"][0]

            # 主要考虑自己一句话就超长了
            if len(batch) == 0 or curLen < self.max_token:
                batch.append(i)
            else:
                self.idx = i
                assert len(batch) != 0, "第i个迭代返回空batch idx了?"
                yield batch
                curLen = self.tokenizer(str, return_length=True)["length"][0]
                batch = [i]

        yield list(range(self.idx, len(self.data)))


class MistralEmbedding:
    def __init__(
        self,
        model_name=None,
        add_eos_token=True,
        model=None,
    ):
        if model is not None:
            self.model = model
        else:
            self.model_name = model_name
            self.model = AutoModel.from_pretrained(
                model_name, torch_dtype=torch.bfloat16
            ).to_bettertransformer()
        # print(self.model.config.num_hidden_layers,'aaaa')
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, model_max_length=4096, add_eos_token=add_eos_token
        )
        self.model.eval()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def my_collate(self, batch):
        prompt = "Translate this from Deutsch to English:\nDeutsch:{de}\nEnglish:{en}"
        # print('batch',batch)
        inputs = [prompt.format_map({"en": b["en"], "de": b["de"]}) for b in batch]
        labels = [b["en"] for b in batch]

        # NOTE 下面这种做法只在label在句尾起效果,因为我是把input拼上label的长度差值作为不相关前缀长度全部屏蔽了
        # print('input',inputs)
        inputs = self.tokenizer.batch_encode_plus(
            inputs, truncation=True, return_length=True
        )

        labels = self.tokenizer.batch_encode_plus(
            labels, truncation=True, return_length=True
        )

        pad_length = [i - j for i, j in zip(inputs.length, labels.length)]

        # 这里要想办法把labels补齐到和inputs一样长
        labels = copy.deepcopy(inputs)
        for i, l in enumerate(pad_length):
            labels.input_ids[i][:l] = [self.tokenizer.pad_token_id for _ in range(l)]

        inputs = self.tokenizer.pad(inputs, return_tensors="pt")
        labels = self.tokenizer.pad(labels, return_tensors="pt").input_ids

        return inputs, labels

    def embed_documents(self, texts: List[dict]) -> List[List[float]]:
        # self.model.cuda()
        """Embed search docs."""
        # print(len(texts),max(len(x) for x in texts))

        dataset = documentDataset(texts)
        sampler = MaxTokenSampler(texts, tokenizer=self.tokenizer)
        cnt = 0
        for s in sampler:
            cnt += 1
            # print(s)
        sampler.length = cnt
        sampler.idx = 0
        dataloader = DataLoader(
            dataset,
            batch_sampler=sampler,
            shuffle=False,
            num_workers=0,
            collate_fn=self.my_collate,
            pin_memory=True,
        )

        from tqdm import tqdm

        data_loader_with_progress = tqdm(
            dataloader, desc="Processing docs into embeddings", ncols=100
        )
        knn_embeddings = torch.empty(0, self.model.config.hidden_size)
        next_token = torch.empty(0)
        cnt = 0
        for inputs, labels in data_loader_with_progress:
            with torch.no_grad():
                if cnt == 0:
                    temp_key = torch.empty(0, self.model.config.hidden_size).to("cuda")

                inputs.to(self.model.device)
                labels.to(self.model.device)

                model_output = self.model(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    labels=labels,
                    return_dict=True,
                    output_hidden_states=True,
                    use_cache=False,
                )

                keys = self.last_token_pool(
                    model_output["last_hidden_state"], inputs.attention_mask
                ).flatten(0, 1)
                values = labels[:, 1:].flatten(0, 1)
                nonpad_mask = values != self.tokenizer.pad_token_id
                values = values[nonpad_mask]
                keys = keys[nonpad_mask].to(temp_key.device)
                temp_key = torch.cat((temp_key, keys), dim=0)

                next_token = torch.cat((next_token, values), dim=0)

                cnt += 1
                if not cnt % 400 and cnt != 0:
                    temp_key = temp_key.to("cpu")
                    knn_embeddings = torch.cat((knn_embeddings, temp_key), dim=0)
                    temp_key = torch.empty(0, self.model.config.hidden_size).to("cuda")
        temp_key = temp_key.to("cpu")
        knn_embeddings = torch.cat((knn_embeddings, temp_key), dim=0)
        maxNorm = torch.max(torch.norm(knn_embeddings, dim=1))
        torch.set_printoptions(edgeitems=32)
        next_token = next_token.tolist()
        return knn_embeddings, next_token, maxNorm

    def embed_query(self, text, check_mode=False) -> List[float]:
        # print(f'真正的query?{text}')
        # print(self.model_name)
        # if self.model_name=='BAAI/bge-large-zh' or self.model_name=='BAAI/bge-large-zh-v1.5':
        #     text="为这个句子生成表示以用于检索相关文章："+text
        """Embed search docs."""
        if check_mode:
            encoded_input = text.to(self.model.device)
        else:
            # NOTE 这个输入text很麻烦,他可能来自两个来源,一个是llama自身的pure knn-lm,一个是和nmt混合的,前者好说,甚至不需要embed,后者难评,甚至需要
            if not isinstance(text, list):
                # print('?')
                text = [text]

            prompt = (
                "Translate this from Deutsch to English:\nDeutsch:{de}\nEnglish:{en}"
            )
            text = [prompt.format_map({"en": b["en"], "de": b["de"]}) for b in text]

            print(text)
            # print(self.tokenizer.decode([29871,13],clean_up_tokenization_spaces=False),'?')

            encoded_input = self.tokenizer(
                text, padding=True, truncation=True, return_tensors="pt"
            ).to(self.model.device)
        # print(encoded_input,encoded_input.input_ids.shape)
        with torch.no_grad():
            if check_mode:
                model_output = self.model(
                    encoded_input,
                    use_cache=False,
                    output_hidden_states=True,
                    return_dict=True,
                )
            else:
                model_output = self.model(
                    **encoded_input,
                    use_cache=False,
                    output_hidden_states=True,
                    return_dict=True
                )

        query_embedding = self.last_token_pool(
            model_output["last_hidden_state"], encoded_input.attention_mask
        )

        return query_embedding.to(torch.float)

    def last_token_pool(last_hidden_states, attention_mask):
        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[
                torch.arange(batch_size, device=last_hidden_states.device),
                sequence_lengths,
            ]

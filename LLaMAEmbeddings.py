import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizerFast
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


# class documentDataset(Dataset):
#     def __init__(self, document) -> None:
#         self.data = document

#     def __getitem__(self, index) -> Any:
#         return self.data[index]

#     def __len__(self):
#         return len(self.data)


class TokenBatchDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer):
        self.texts = [
            tokenizer.encode(text, add_special_tokens=False) for text in texts
        ]

        self.lengths = [len(text) for text in self.texts]
        self.sorted_indices = sorted(
            range(len(self.texts)), key=lambda i: self.lengths[i], reverse=True
        )

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[self.sorted_indices[idx]]


from torch.utils.data.sampler import Sampler


# class MaxTokenSampler(Sampler):
#     def __init__(self, data, tokenizer) -> None:
#         self.data = data
#         self.max_token = 256
#         self.idx = 0
#         self.length = 0
#         self.tokenizer = tokenizer

#     def __len__(self) -> int:
#         return self.length

#     def __iter__(self):

#         batch = []
#         curLen = 0
#         for i in range(self.idx, len(self.data)):
#             prompt = (
#                 "Translate this from Deutsch to English:\nDeutsch:{de}\nEnglish:{en}"
#             )
#             str = prompt.format_map(
#                 {"en": self.data[i]["en"], "de": self.data[i]["de"]}
#             )
#             # print(self.tokenizer.encode(str))
#             curLen += self.tokenizer(str, return_length=True)["length"][0]

#             # 主要考虑自己一句话就超长了
#             if len(batch) == 0 or curLen < self.max_token:
#                 batch.append(i)
#             else:
#                 self.idx = i
#                 assert len(batch) != 0, "第i个迭代返回空batch idx了?"
#                 yield batch
#                 curLen = self.tokenizer(str, return_length=True)["length"][0]
#                 batch = [i]

#         yield list(range(self.idx, len(self.data)))


class DynamicBatchSampler(Sampler):
    def __init__(self, dataset, max_tokens=256, min_len=1, random=False):
        self.dataset = dataset
        self.max_tokens = max_tokens
        self.min_len = min_len
        self.random = random
        # 获取排序后的长度列表
        self.lengths = [self.dataset.lengths[i] for i in self.dataset.sorted_indices]

    def __iter__(self):
        n = len(self.dataset)
        batches = []
        current_batch = []
        current_length = 0

        for i in range(n):
            idx = self.dataset.sorted_indices[i]
            length = self.lengths[i]

            # 如果加入当前样本会超出最大token数，或者当前batch已经足够大
            if current_length + length > self.max_tokens or len(current_batch) >= 512:
                if current_batch:  # 确保不会添加空batch
                    batches.append(current_batch)
                current_batch = [idx]
                current_length = length
            else:
                current_batch.append(idx)
                current_length += length

            # 如果剩余的样本不足以形成一个新的batch，就将它们添加到当前batch
            if i == n - 1 or self.lengths[i + 1] + current_length > self.max_tokens:
                if len(current_batch) >= self.min_len:
                    batches.append(current_batch)
                current_batch = []
                current_length = 0

        # 打乱batch的顺序
        if self.random:
            np.random.shuffle(batches)

        for batch in batches:
            yield batch

    def __len__(self):
        return len(list(self.__iter__()))


import config


class LLaMAEmbedding:
    def __init__(
        self,
        model_name=config.llama_path,
        batch_size=4,
        model=None,
        prompt=None,
        vdb_type=None,
        use_prompt=True,
    ):
        if model is not None:
            self.model = model
        else:
            self.model_name = model_name[vdb_type]
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name[vdb_type], torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
            )
        self.model.eval()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name[vdb_type])
        self.tokenizer.padding_side = "left"
        # TODO 这个行为和旧实现可能有冲突，等下关注一下。
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # self.tokenizer.pad_token_id = 0

        self.batch_size = batch_size
        self.prompt = prompt
        self.use_prompt = use_prompt

    def my_collate(self, batch):

        # prompt='Translate this from Deutsch to English:\nDeutsch:{de}\nEnglish:{en}'
        if self.use_prompt:
            prompt = self.prompt
            # print('batch',batch)
            inputs = [prompt.format_map({"en": b["en"], "de": b["de"]}) for b in batch]
            labels = [b["en"] + self.tokenizer.eos_token for b in batch]
        else:
            # KNN-LM mode, we only need to process the target part(i.e. English)
            inputs = [[self.tokenizer.bos_token_id] + b for b in batch]
            labels = [b + [self.tokenizer.eos_token_id] for b in batch]

        inputs = self.tokenizer.pad({"input_ids": inputs}, return_tensors="pt")

        # 对labels的pad最好单独做，拿-100来，方便。
        max_label_len = max([len(label) for label in labels])

        for i, label in enumerate(labels):
            labels[i] = [-100] * (max_label_len - len(label)) + label
        labels = torch.tensor(labels)
        return inputs, labels

    def embed_documents(self, texts: List[dict]) -> List[List[float]]:
        # self.model.cuda()
        """Embed search docs."""
        # print(len(texts),max(len(x) for x in texts))

        dataset = TokenBatchDataset(texts, self.tokenizer)
        sampler = DynamicBatchSampler(dataset, max_tokens=256, min_len=2)

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
        # temp_key是在GPU上的张量，缓存了在特定时间步内的knn_embeddings,到阈值了就会传回CPU给knn_embeddings
        # 数据流动就是 keys->temp_key->knn_embeddings
        knn_embeddings = torch.empty(0, self.model.config.hidden_size)
        temp_key = torch.empty(0, self.model.config.hidden_size).to("cuda")
        next_token = torch.empty(0)
        cnt = 0  # 用于异步从GPU上卸载下knn datastore的 key，防止爆显存。
        for inputs, labels in data_loader_with_progress:

            with torch.no_grad():
                inputs.to(self.model.device)
                labels.to(self.model.device)

                model_output = self.model(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    return_dict=True,
                    output_hidden_states=True,
                    use_cache=False,
                )

                # hidden_size 是 layers x batch x seqlen x c
                # torch.set_printoptions(edgeitems =5)
                # print('debug input',inputs.input_ids)
                # print('debug',model_output['hidden_states'][-1].shape,model_output['hidden_states'][-1][:,-20:])

                nonpad_mask = labels != -100
                values = labels[nonpad_mask]
                # print(keys,values)
                # print(nonpad_mask)
                # keys = keys[nonpad_mask].to(knn_embeddings.device)
                keys = model_output["hidden_states"][-1][nonpad_mask].to(
                    temp_key.device
                )

                # 这里应该不再需要做偏移了,因为 <s> + ... + tgt 和 mask + tgt + </s> 本身就相当于把values往左移动了
                # keys=model_output['hidden_states'][-1][:, :-1].flatten(0, 1)
                # values=labels[:, 1:].flatten(0, 1)

                # keys=keys.flatten(0, 1)
                # values=values.flatten(0, 1)

                temp_key = torch.cat((temp_key, keys), dim=0)
                # knn_embeddings = torch.cat((knn_embeddings, keys), dim=0)
                # 这里非常非常奇怪，values居然在cpu上
                # TODO 尊嘟假嘟？需要再一次检查这块
                next_token = torch.cat((next_token, values), dim=0)
                # decoder_raw_token=self.tokenizer.batch_decode(decoder_input_ids,skip_special_tokens=True)
                # return_list.append([(embedding,next_token) for next_token,embedding in zip(decoder_raw_token,model_output['decoder_hidden_states'])])

                cnt += 1
                if not cnt % 400:
                    temp_key = temp_key.to("cpu")
                    knn_embeddings = torch.cat((knn_embeddings, temp_key), dim=0)
                    temp_key = torch.empty(0, self.model.config.hidden_size).to("cuda")

        temp_key = temp_key.to("cpu")
        knn_embeddings = torch.cat((knn_embeddings, temp_key), dim=0)
        # print(knn_embeddings,knn_embeddings.shape)
        maxNorm = torch.max(torch.norm(knn_embeddings, dim=1))
        torch.set_printoptions(edgeitems=32)
        # print(f'knn_embeddings的形状是{(knn_embeddings.shape)}，knn_embeddings中最大的范数是{maxNorm}')
        # print(f'next_token的形状是{(next_token.shape)}')
        next_token = next_token.tolist()
        # print('knn_embeddings',knn_embeddings,knn_embeddings.shape)
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

            # prompt='Translate this from Deutsch to English:\nDeutsch:{de}\nEnglish:{en}'
            if self.use_prompt:
                prompt = self.prompt
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

        # print(model_output['hidden_states'][-1],model_output['hidden_states'][-1].shape)
        query_embedding = model_output["hidden_states"][-1][:, -1]
        # print(model_output['hidden_states'][-1][:, -1])
        # torch.set_printoptions(edgeitems =7)
        # print(query_embedding,query_embedding.shape)

        return query_embedding.to(torch.float)

    # def mean_pooling(self,model_output, attention_mask):
    #     """
    #     本来参数是model_output，但是我在外面抽出了最后一层状态，这样有很大的问题，因为这里依赖于attention矩阵！好在这个正则化相当于自身的归一。
    #     之所以需要这一步，是因为pad位置的输出还不一样，而且也不是0，为了消除这个影响，只能手动对他们置于0
    #     """
    #     token_embeddings = model_output.last_hidden_state  # First element of model_output contains all token embeddings
    #     # 这个操作使得mask和embedding是一个维度了，本来一个是bsz x seqlen x hdz，mask是bsz x seqlen的，unsqueeze之后就是bsz x seqlen x1
    #     # 然后在最后一维复制hdz次，转成float是因为下面要运算
    #     input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    #     # 需要被mask掉的位置就会失去他的光辉，沿着句长维度求和。clamp是把数压缩在min，max之间，也是沿着句长维度求和，
    #     # 之所以min取了一个数字，是因为全0的问题？或者下溢出？
    #     return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

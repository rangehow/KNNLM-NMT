import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizerFast
import torch
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
from loguru import logger
import config
from load import balanced_load
from torch.utils.data.sampler import Sampler
import numpy as np
from torch.utils.data import Sampler


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
        # print(idx)
        return self.texts[idx]


class DynamicBatchSampler(Sampler):
    """要求与其配合的数据集存在sorted_indices以及lengths两个成员

    Args:
        Sampler (_type_): _description_
    """
    def __init__(self, dataset, max_tokens=256, min_len=1, random=False):
        self.dataset = dataset
        self.max_tokens = int(max_tokens * 0.9)
        self.min_len = min_len
        self.random = random
        # 获取排序后的长度列表
        self.lengths = [self.dataset.lengths[i] for i in self.dataset.sorted_indices]

    def __iter__(self):
        n = len(self.dataset)
        batches = []
        current_batch = []
        max_len_in_batch = 0
        current_tokens = 0
        i = 0

        while i < n:
            idx = self.dataset.sorted_indices[i]
            length = self.lengths[i]

            # 计算如果加入当前样本，整个batch的token数
            new_max_len = max(max_len_in_batch, length)
            new_batch_size = len(current_batch) + 1
            new_total_tokens = (new_max_len + 1) * new_batch_size

            # 如果加入当前样本会超出最大token数，或者当前batch已经足够大
            if new_total_tokens > self.max_tokens or new_batch_size > 512:
                if current_batch and len(current_batch) >= self.min_len:
                    # fake_length=max(
                    #     self.lengths[self.dataset.sorted_indices.index(i)]
                    #     for i in current_batch
                    # )
                    # fake_total_tokens = fake_length* len(current_batch)
                    # print(fake_length,fake_total_tokens)
                    # import pdb

                    # pdb.set_trace()
                    batches.append(current_batch)

                    current_batch = []
                    max_len_in_batch = 0
                    current_tokens = 0
                    # 不增加 i，再次尝试将当前实例添加到新的 batch
                else:
                    # 如果当前 batch 太小，强制添加当前实例
                    current_batch.append(idx)
                    max_len_in_batch = new_max_len
                    current_tokens = new_total_tokens
                    i += 1
            else:
                current_batch.append(idx)
                max_len_in_batch = new_max_len
                current_tokens = new_total_tokens
                i += 1

        if current_batch:
            batches.append(current_batch)

        # 打乱batch的顺序
        if self.random:
            np.random.shuffle(batches)

        for batch in batches:
            # total_tokens = (
            #     max(self.lengths[self.dataset.sorted_indices.index(i)] for i in batch)
            #     + 1
            # ) * len(batch)

            # logger.debug(f"sampler认为的长度是{total_tokens}")
            # logger.debug(f"sampler吐出的下标是{batch}")
            yield batch

    def __len__(self):
        return len(list(self.__iter__()))


import torch


def find_max_tokens(model, initial_tokens=81, max_possible_tokens=8192 * 8):
    lower_bound = initial_tokens
    upper_bound = initial_tokens
    max_tokens = initial_tokens

    # 快速逼近上限阶段
    while upper_bound <= max_possible_tokens:
        try:
            input_ids = torch.randint(100, 1000, (1, upper_bound)).to(model.device)
            outputs = model(input_ids)
            print(f"Successfully ran with {upper_bound} tokens")
            max_tokens = upper_bound
            lower_bound = upper_bound
            upper_bound *= 2
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print(f"CUDA out of memory at {upper_bound} tokens")
                break
            else:
                raise e

    # 精确搜索阶段（二分搜索）
    while lower_bound <= upper_bound:
        mid = (lower_bound + upper_bound) // 2
        try:
            input_ids = torch.randint(100, 1000, (1, mid)).to(model.device)
            outputs = model(input_ids)
            print(f"Successfully ran with {mid} tokens")
            max_tokens = mid
            lower_bound = mid + 1
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print(f"CUDA out of memory at {mid} tokens")
                upper_bound = mid - 1
            else:
                raise e

    return max_tokens


class LLaMAEmbedding:
    def __init__(
        self,
        model_name=config.llama_path,
        batch_size=4,
        prompt=None,
        vdb_type=None,
        use_prompt=True,
    ):

        self.model = balanced_load(model_name[vdb_type])
        # self.model.to_bettertransformer()
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
        # logger.debug(f"batch的数量是")
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
        optimum_max_tokens = find_max_tokens(
            self.model, initial_tokens=8192, max_possible_tokens=8192 * 2
        )
        logger.info(f"寻找到的batch最大token数量是{optimum_max_tokens}")
        sampler = DynamicBatchSampler(dataset, max_tokens=optimum_max_tokens, min_len=1)

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
                # logger.debug(f"这个batch的token数量是：{inputs.input_ids.numel()}")
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
                if not cnt % 2:
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
                    return_dict=True,
                )

        # print(model_output['hidden_states'][-1],model_output['hidden_states'][-1].shape)
        query_embedding = model_output["hidden_states"][-1][:, -1]
        # print(model_output['hidden_states'][-1][:, -1])
        # torch.set_printoptions(edgeitems =7)
        # print(query_embedding,query_embedding.shape)

        return query_embedding.to(torch.float)

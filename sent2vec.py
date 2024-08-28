from transformers import BertTokenizer, BertModel, AutoModel, AutoTokenizer
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



def cos_sim(a, b):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))





class documentDataset(Dataset):
    def __init__(self, document) -> None:
        super().__init__()
        self.data = document

    def __getitem__(self, index) -> Any:
        return self.data[index]

    def __len__(self):
        return len(self.data)


class Sent2VecEmbeddings:
    def __init__(
        self, model_name=None, batch_size=256, retriever_mode=False
    ):
        self.model_name = model_name
        self.model = AutoModel.from_pretrained(
            model_name,
        ).to_bettertransformer()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, model_max_length=self.model.config.max_length)
        self.model.eval()

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        # 这里如果解开了注释,在多线程的时候就会多占显存,但是不解开的话 embed query那里逻辑又很怪
        # self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # self.model.to(self.device)
        self.batch_size = batch_size

    def my_collate(self, batch,):
        
        return self.tokenizer(batch, padding=True, truncation=True, return_tensors='pt'),batch
        # return self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # self.model.cuda()
        """Embed search docs."""
        # print(len(texts),max(len(x) for x in texts))
        dataset = documentDataset(texts)

        # 数据量如果不够bsz会有逆天报错……
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=32,
            collate_fn=self.my_collate,
            pin_memory=True,
        )  # GPU张量没法pin的，都不在CPU上哪来内存

        
       
        print(
            f"model_name={self.model_name},hidden_size={self.model.config.hidden_size}"
        )
        with torch.no_grad():
            doc_embeddings = torch.empty(0, self.model.config.hidden_size).to(self.model.device)
            # doc_embeddings = torch.empty(0, self.model.module.config.hidden_size)
            from tqdm import tqdm
            data_loader_with_progress = tqdm(
                dataloader, desc="Processing docs into embeddings", ncols=100
            )
            doc_text=[]
            
            for encoded_input,batch_text in data_loader_with_progress:
            # for encoded_input in data_loader_with_progress:
                
                encoded_input.to(self.model.device)
                model_output = self.model(**encoded_input)
                # model_output = accelerator.gather(model_output)
                mean_output = self.mean_pooling(
                    model_output, encoded_input["attention_mask"]
                )

                doc_embeddings = torch.cat((doc_embeddings, mean_output), dim=0)
            doc_embeddings=doc_embeddings.to('cpu')

            
            maxNorm = torch.max(torch.norm(doc_embeddings, dim=1))

            print(f"doc_embeddings的shape是{doc_embeddings.shape}，doc_embeddings中最大的范数是{maxNorm}")

        
        # 下面这个行为是错误的！因为不是保序变换，正确的做法可能得是算出max norm，然后直接除？
        # doc_embeddings=torch.nn.functional.normalize(doc_embeddings, p=2, dim=1)
        # doc_embeddings=doc_embeddings/maxNorm
        # print(f'doc_embeddings的shape是{doc_embeddings.shape}，norm之后，doc_embeddings中最大的范数是{torch.max(torch.norm(doc_embeddings,dim=1))}')
        return doc_embeddings, maxNorm

    def embed_query(self, text: str) -> List[float]:
        # print(f'真正的query?{text}')
        # print(self.model_name)
        # if self.model_name=='BAAI/bge-large-zh' or self.model_name=='BAAI/bge-large-zh-v1.5':
        #     text="为这个句子生成表示以用于检索相关文章："+text
        """Embed search docs."""

        encoded_input = self.tokenizer(
            text, padding=True, truncation=True, return_tensors="pt"
        ).to(self.model.device)
        with torch.inference_mode():
            model_output = self.model(**encoded_input)
        query_embedding = self.mean_pooling(
            model_output, encoded_input["attention_mask"]
        )

        return query_embedding

    def mean_pooling(self, model_output, attention_mask):
        """
        本来参数是model_output，但是我在外面抽出了最后一层状态，这样有很大的问题，因为这里依赖于attention矩阵！好在这个正则化相当于自身的归一。
        之所以需要这一步，是因为pad位置的输出还不一样，而且也不是0，为了消除这个影响，只能手动对他们置于0
        """
        token_embeddings = (
            model_output.last_hidden_state
        )  # First element of model_output contains all token embeddings
        # 这个操作使得mask和embedding是一个维度了，本来一个是bsz x seqlen x hdz，mask是bsz x seqlen的，unsqueeze之后就是bsz x seqlen x1
        # 然后在最后一维复制hdz次，转成float是因为下面要运算
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        # 需要被mask掉的位置就会失去他的光辉，沿着句长维度求和。clamp是把数压缩在min，max之间，也是沿着句长维度求和，
        # 之所以min取了一个数字，是因为全0的问题？或者下溢出？
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

from transformers import BertTokenizer, BertModel,AutoModel,AutoTokenizer
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
from torch.utils.data import Dataset,DataLoader

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

model_name='shibing624/text2vec-base-chinese-paraphrase'
model_max_length={'GanymedeNil/text2vec-large-chinese':512,'shibing624/text2vec-base-chinese-paraphrase':2048,'BAAI/bge-large-zh':512
                  ,'BAAI/bge-large-zh-noinstruct':512,'BAAI/bge-large-zh-v1.5':512}


class documentDataset(Dataset):
    def __init__(self,document) -> None:
        super().__init__()
        self.data=document
    def __getitem__(self, index) -> Any:
        return self.data[index]
    def __len__(self):
        return len(self.data)




class NMTEmbeddings():
    def __init__(self,model_name='/hy-tmp/wmt21',batch_size=16):
        
        self.model_name = model_name
        
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name,torch_dtype='auto').to_bettertransformer()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name,model_max_length=self.model.config.max_length)
        # self.model=torch.compile(self.model)
        self.model.eval()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # self.device = "cpu"
        self.model.to(self.device)
        self.batch_size=batch_size


    def my_collate(self,batch):
        encoder=[b['a'] for b in batch]
        decoder=[b['b'] for b in batch]

        encoder=self.tokenizer(encoder, padding=True, truncation=True, return_tensors='pt')
        decoder=self.tokenizer(decoder, padding=True, truncation=True, return_tensors='pt')
        return encoder,decoder


    def embed_documents(self, texts: List[dict]) -> List[List[float]]:
        # self.model.cuda()
        """Embed search docs."""
        # print(len(texts),max(len(x) for x in texts))
        dataset=documentDataset(texts)
        
        dataloader=DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=80,
                             collate_fn=self.my_collate, pin_memory=True) #GPU张量没法pin的，都不在CPU上哪来内存
        
        print(f'model_name={self.model_name},hidden_size={self.model.config.hidden_size}')
        return_list=[]
        from tqdm import tqdm 
        data_loader_with_progress = tqdm(dataloader, desc="Processing docs into embeddings", ncols=100)
        for input in data_loader_with_progress:
            with torch.inference_mode():
                encoder,decoder=input
                encoder.to('cuda')
                decoder.to('cuda')
                input_ids=encoder.input_ids
                attention_mask=encoder.attention_mask
                decoder_input_ids=decoder.input_ids
                decoder_attention_mask=decoder.attention_mask

                model_output = self.model(input_ids=input_ids,attention_mask=attention_mask,
                                          decoder_input_ids=decoder_input_ids,decoder_attention_mask=decoder_attention_mask,
                                          return_dict=True,output_hidden_states=True)
                print(decoder_input_ids.shape,model_output['decoder_hidden_states'].shape)
                decoder_raw_token=self.tokenizer.batch_decode(decoder_input_ids,skip_special_tokens=True)
                return_list.append([(embedding,next_token) for next_token,embedding in zip(decoder_raw_token,model_output['decoder_hidden_states'])])
            
        # 这里加0.1是因为，反正是保序变换了，整大一点，完全没关系，除了会影响相似度的绝对值，不会影响序列关系，之前碰到阴间case，可能是精度问题，导致最大精度稍微溢出1.0
        maxNorm=torch.max(torch.norm([x[0] for x in return_list],dim=1))
        print(f'doc_embeddings的数量是{len(return_list)},一个向量的形状是{(return_list[0][0].shape)}，doc_embeddings中最大的范数是{maxNorm}')
        
        return return_list,maxNorm

    def embed_query(self, text: str) -> List[float]:
        # print(f'真正的query?{text}')
        # print(self.model_name)
        # if self.model_name=='BAAI/bge-large-zh' or self.model_name=='BAAI/bge-large-zh-v1.5':
        #     text="为这个句子生成表示以用于检索相关文章："+text
        """Embed search docs."""
        
        encoded_input = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt').to(self.device)
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        query_embedding = self.mean_pooling(model_output, encoded_input['attention_mask'])[0]
        

        return query_embedding

    def mean_pooling(self,model_output, attention_mask):
        """
        本来参数是model_output，但是我在外面抽出了最后一层状态，这样有很大的问题，因为这里依赖于attention矩阵！好在这个正则化相当于自身的归一。
        之所以需要这一步，是因为pad位置的输出还不一样，而且也不是0，为了消除这个影响，只能手动对他们置于0
        """
        token_embeddings = model_output.last_hidden_state  # First element of model_output contains all token embeddings
        # 这个操作使得mask和embedding是一个维度了，本来一个是bsz x seqlen x hdz，mask是bsz x seqlen的，unsqueeze之后就是bsz x seqlen x1
        # 然后在最后一维复制hdz次，转成float是因为下面要运算
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        # 需要被mask掉的位置就会失去他的光辉，沿着句长维度求和。clamp是把数压缩在min，max之间，也是沿着句长维度求和，
        # 之所以min取了一个数字，是因为全0的问题？或者下溢出？
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


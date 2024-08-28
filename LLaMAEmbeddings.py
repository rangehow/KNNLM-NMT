from transformers import AutoModelForCausalLM,AutoTokenizer,LlamaTokenizerFast
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
import copy


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
    def __init__(self,document) -> None:
        super().__init__()
        self.data=document
    def __getitem__(self, index) -> Any:
        return self.data[index]
    def __len__(self):
        return len(self.data)
    

from torch.utils.data.sampler import Sampler
class MaxTokenSampler(Sampler):
    def __init__(self, data,tokenizer) -> None:
        self.data = data
        self.max_token=256
        self.idx=0
        self.length=0
        self.tokenizer=tokenizer
    def __len__(self) -> int:
        return self.length
    def __iter__(self):
        
        batch=[]
        curLen=0
        for i in range(self.idx,len(self.data)):
            prompt='Translate this from Deutsch to English:\nDeutsch:{de}\nEnglish:{en}'
            str=prompt.format_map({'en':self.data[i]['en'],'de':self.data[i]['de']})
            # print(self.tokenizer.encode(str))                
            curLen+=self.tokenizer(str,return_length=True)['length'][0]
            
            #主要考虑自己一句话就超长了
            if len(batch)==0 or curLen<self.max_token:
                batch.append(i)
            else :   
                self.idx = i
                assert len(batch) != 0, "第i个迭代返回空batch idx了?"
                yield batch
                curLen = self.tokenizer(str, return_length=True)["length"][0]
                batch = [i]
            
        yield list(range(self.idx,len(self.data)))

import config
class LLaMAEmbedding():
    def __init__(self,model_name=config.llama_path,batch_size=4,add_eos_token=True,model=None,prompt=None,vdb_type=None,use_prompt=True):
        if model is not None:
            self.model=model
        else:
            self.model_name = model_name[vdb_type]
            self.model = AutoModelForCausalLM.from_pretrained(model_name[vdb_type],torch_dtype=torch.bfloat16,low_cpu_mem_usage=True)
        # print(self.model.config.num_hidden_layers,'aaaa')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name[vdb_type],model_max_length=4096)
        # LLAMA的tokenizer没有pad，但是不要用unk，也不要用eos，因为他们都会在后面构建knn数据库时被屏蔽，这个行为是不正确的
        # unk可能会在查询阶段出现，没想明白
        # 但是eos一定要留
        # self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        # self.model.resize_token_embeddings(len(self.tokenizer)) # 不更新embedding层会匹配不上维度，很烦
        self.tokenizer.pad_token_id=0
        self.model.config.pad_token_id=self.tokenizer.pad_token_id # 我疑心这个有没有用
        # self.tokenizer.pad_token = self.tokenizer.unk_token
        # knnsaver= KNNSaver()
        # knnsaver.break_into(self.model)
        
        
        self.model.eval()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        
        self.batch_size=batch_size
        self.prompt=prompt
        self.use_prompt=use_prompt

    def my_collate(self,batch):
        # prompt='Translate this from Deutsch to English:\nDeutsch:{de}\nEnglish:{en}'
        if self.use_prompt:
            prompt=self.prompt
            # print('batch',batch)
            inputs=[prompt.format_map({'en':b['en'],'de':b['de']}) for b in batch]
            labels=[b['en']+self.tokenizer.eos_token for b in batch]
        else:
            inputs=[b['en'] for b in batch]
            labels=[b['en']+self.tokenizer.eos_token for b in batch]
        # NOTE 下面这种做法只在label在句尾起效果,因为我是把input拼上label的长度差值作为不相关前缀长度全部屏蔽了
        # print('input',inputs)
        inputs=self.tokenizer.batch_encode_plus(inputs,truncation=True,return_length=True)

        self.tokenizer.add_bos_token=False
        labels=self.tokenizer.batch_encode_plus(labels,truncation=True,return_length=True)
        self.tokenizer.add_bos_token=True
        
        pad_length=[i-j for i,j in zip(inputs.length,labels.length)]

        # 这里要想办法把labels补齐到和inputs一样长
        # labels = copy.deepcopy(inputs)
        for i,l in enumerate(pad_length):
            labels.input_ids[i]=[self.tokenizer.pad_token_id for _ in range(l)] +labels.input_ids[i]

        try:
            inputs=self.tokenizer.pad(inputs,return_tensors='pt')
            del labels['attention_mask']
            labels=self.tokenizer.pad(labels,return_tensors='pt').input_ids
        except:
            import pdb
            pdb.set_trace()
        

        return inputs,labels


    def embed_documents(self, texts: List[dict]) -> List[List[float]]:
        # self.model.cuda()
        """Embed search docs."""
        # print(len(texts),max(len(x) for x in texts))
        
        dataset=documentDataset(texts)
        sampler=MaxTokenSampler(texts,tokenizer=self.tokenizer)
        cnt=0
        for s in sampler:
            cnt+=1
        sampler.length=cnt
        sampler.idx=0
        dataloader=DataLoader(dataset, batch_sampler=sampler, shuffle=False, num_workers=8,
                             collate_fn=self.my_collate, pin_memory=True) #GPU张量没法pin的，都不在CPU上哪来内存
        
        from tqdm import tqdm 
        data_loader_with_progress = tqdm(dataloader, desc="Processing docs into embeddings", ncols=100)
        knn_embeddings = torch.empty(0, self.model.config.hidden_size)
        next_token= torch.empty(0)
        cnt=0
        for inputs,labels in data_loader_with_progress:

            with torch.no_grad():
                if cnt==0:
                    temp_key=torch.empty(0, self.model.config.hidden_size).to('cuda')
                
                inputs.to(self.model.device)
                labels.to(self.model.device)
 
                model_output = self.model(input_ids=inputs.input_ids,
                                          attention_mask=inputs.attention_mask,
                                          return_dict=True,output_hidden_states=True,
                                          use_cache=False)

                # hidden_size 是 layers x batch x seqlen x c 
                # torch.set_printoptions(edgeitems =5)
                # print('debug input',inputs.input_ids)
                # print('debug',model_output['hidden_states'][-1].shape,model_output['hidden_states'][-1][:,-20:])
                
                nonpad_mask = labels != self.tokenizer.pad_token_id
                values = labels[nonpad_mask]
                # print(keys,values)
                # print(nonpad_mask)
                # keys = keys[nonpad_mask].to(knn_embeddings.device)
                keys = model_output['hidden_states'][-1][nonpad_mask].to(temp_key.device)
                
                # 这里应该不再需要做偏移了,因为 <s> + ... + tgt 和 mask + tgt + </s> 本身就相当于把values往左移动了 
                # keys=model_output['hidden_states'][-1][:, :-1].flatten(0, 1)
                # values=labels[:, 1:].flatten(0, 1)
                
                # keys=keys.flatten(0, 1)
                # values=values.flatten(0, 1)
                
                temp_key = torch.cat((temp_key, keys), dim=0)
                # knn_embeddings = torch.cat((knn_embeddings, keys), dim=0)
                # 这里非常非常奇怪，values居然在cpu上，但是keys不在？
                next_token= torch.cat((next_token, values), dim=0)
                # decoder_raw_token=self.tokenizer.batch_decode(decoder_input_ids,skip_special_tokens=True)
                # return_list.append([(embedding,next_token) for next_token,embedding in zip(decoder_raw_token,model_output['decoder_hidden_states'])])
                
                cnt+=1
                if not cnt%400 and cnt!=0:
                    temp_key=temp_key.to('cpu')
                    # print(temp_key)
                    knn_embeddings=torch.cat((knn_embeddings, temp_key), dim=0)
                    temp_key=torch.empty(0, self.model.config.hidden_size).to('cuda')

        temp_key=temp_key.to('cpu')
        knn_embeddings=torch.cat((knn_embeddings, temp_key), dim=0)
        # print(knn_embeddings,knn_embeddings.shape)
        maxNorm = torch.max(torch.norm(knn_embeddings, dim=1))
        torch.set_printoptions(edgeitems =32)    
        # print(f'knn_embeddings的形状是{(knn_embeddings.shape)}，knn_embeddings中最大的范数是{maxNorm}')
        # print(f'next_token的形状是{(next_token.shape)}')
        next_token=next_token.tolist()
        # print('knn_embeddings',knn_embeddings,knn_embeddings.shape)
        return knn_embeddings,next_token,maxNorm

    def embed_query(self, text,check_mode=False) -> List[float]:
        # print(f'真正的query?{text}')
        # print(self.model_name)
        # if self.model_name=='BAAI/bge-large-zh' or self.model_name=='BAAI/bge-large-zh-v1.5':
        #     text="为这个句子生成表示以用于检索相关文章："+text
        """Embed search docs."""
        if check_mode:
            encoded_input=text.to(self.model.device)
        else:
            # NOTE 这个输入text很麻烦,他可能来自两个来源,一个是llama自身的pure knn-lm,一个是和nmt混合的,前者好说,甚至不需要embed,后者难评,甚至需要
            if not isinstance(text ,list):
                # print('?')
                text=[text]
            
            # prompt='Translate this from Deutsch to English:\nDeutsch:{de}\nEnglish:{en}'
            if self.use_prompt:
                prompt=self.prompt
                text=[prompt.format_map({'en':b['en'],'de':b['de']}) for b in text]
                
            print(text)
            # print(self.tokenizer.decode([29871,13],clean_up_tokenization_spaces=False),'?')

            encoded_input = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt').to(self.model.device)\
        
        # print(encoded_input,encoded_input.input_ids.shape)
        with torch.no_grad():
            if check_mode:
                model_output = self.model(encoded_input,use_cache=False,output_hidden_states=True,return_dict =True)
            else:
                model_output = self.model(**encoded_input,use_cache=False,output_hidden_states=True,return_dict =True)
  

        # print(model_output['hidden_states'][-1],model_output['hidden_states'][-1].shape)
        query_embedding = model_output['hidden_states'][-1][:, -1]
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


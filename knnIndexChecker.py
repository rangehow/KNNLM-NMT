
from argparse import ArgumentParser
import json
from langchain.document_loaders import TextLoader, JSONLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from faissManager import FAISS
from LLaMAEmbeddings import LLaMAEmbedding
from langchain.document_transformers import (
    LongContextReorder,
)
from dataclasses import dataclass
from transformers import AutoTokenizer
# @dataclass
# class template:

# 故意打错类名的，:) 防止后面出现好笑的和文件重名的事故
class knnretriver:
    def __init__(self, index_dir, embeddings) -> None:
        
        self.db = FAISS.load_local(index_dir, embeddings)
        print(
            f"成功加载了向量数据库{index_dir}-------------------------------------------------------------------------------------------------------"
        )

    def retrieve(
        self,
        query,
        k=10,
        score_threshold=None,
        debug=False,
    ):
        # begin_idx=query.rfind('Human:')+7
        # end_idx=query.rfind('\nAssistant:')
        # real_question=query[begin_idx:end_idx]
        # print(roles)

        if debug:
            print(f"获取到的问题：{query}\n")
        # docs = self.db.similarity_search(real_question,k,)
        docs = self.db.similarity_search(
            query,  k, score_threshold=score_threshold
        )

        if debug:
            print(
                f"过滤掉了{2*k-len(docs)}个距离超出{score_threshold}检索到的文档\n"
            )
        return docs
        
    def check(
            self,
            query,
            k=10,
            score_threshold=None,
            debug=False,
        ):
            # begin_idx=query.rfind('Human:')+7
            # end_idx=query.rfind('\nAssistant:')
            # real_question=query[begin_idx:end_idx]
            # print(roles)

            if debug:
                print(f"获取到的问题：{query}\n")
            # docs = self.db.similarity_search(real_question,k,)
            docs = self.db.similarity_search(
                query,  k, score_threshold=score_threshold,check_mode=True
            )

            if debug:
                print(
                    f"过滤掉了{2*k-len(docs)}个距离超出{score_threshold}检索到的文档\n"
                )
            return docs

def parse_args():
  parser=ArgumentParser()
  parser.add_argument('--domain',type=str,help='指定需要处理的json文件',default='koran')

  return parser.parse_args()

from tqdm import tqdm
if __name__=='__main__':
    args=parse_args()
    
    index_dir=f'/data/lxy/abu/vdbdata/{args.domain}/knn_vdb'
    # db = FAISS.load_local(index_dir, None)

    embeddings=LLaMAEmbedding(add_eos_token=True)
    r=knnretriver(index_dir,embeddings)
    data=list(map(json.loads,open(f'/data/lxy/abu/vdbdata/{args.domain}/pair.jsonl').readlines()[:32]))
    cnt,acc=0,0
    for q in tqdm(data):
        # koran
        tokenizer=embeddings.tokenizer
        # tokenizer = AutoTokenizer.from_pretrained('/data/lxy/abu/llama2-7b',model_max_length=4096,add_eos_token=True)
        
        en_length=len(tokenizer.encode(q['en']))-1
        prompt='Translate this from Deutsch to English:\nDeutsch:{de}\nEnglish:{en}'
        text=[prompt.format_map({'en':q['en'],'de':q['de']})]
        input=tokenizer(text,return_tensors='pt').input_ids
        
        # print(input,en_length)
        # print(tokenizer.batch_decode(input[:,:-en_length]))
        for i in range(input.shape[1]-en_length,input.shape[1]-1):
            query=input[:,:i]
            target=input[:,i]
            cnt+=1
            result=r.check(query,debug=False,k=1)
            # print(result)
            if result[0][0]['page_content']==target:
                acc+=1
            else:
                pass
                print(input,query,result[0][0]['page_content'],target,tokenizer.batch_decode([result[0][0]['page_content'],target]))
                
    print(acc/cnt,'acc:',acc,'cnt:',cnt)
        # law
        # query=[{"de": "Auf Antrag werden auch Arbeitnehmervertreter einbezogen.\n", "en": "Whereas, moreover, employees' representatives may decide not to seek the setting-up of a European Works Council or the parties concerned may decide on other procedures for the transnational "}]
        
        # medical
        # query=[{"de": "Xiliarx 50 mg Tabletten\n", "en": "Jalra 50 mg "}] # Jalra 50 mg tablets vildagliptin\n"}

        # print(r.retrieve(q,debug=True,k=5))
        # exit()
       

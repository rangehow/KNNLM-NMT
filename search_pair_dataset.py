from retriever import retriver
from langchain.document_loaders import JSONLoader
import json
from faissManager import FAISS
from NMTEmbeddings import NMTEmbeddings
# from LLaMAEmbeddings import LLaMAEmbeddings
from sent2vec import Sent2VecEmbeddings
from argparse import ArgumentParser
import os
from torch.utils.data import DataLoader
import json
from tqdm import tqdm

if __name__ == "__main__":
    embedding = Sent2VecEmbeddings(
            model_name='/data/lxy/abu/LaBSE', batch_size=4
        )
    
    for domain in ['it','koran','law','medical']:
        r=retriver(index_dir=f'/data/lxy/abu/vdbdata/{domain}/vdb',embeddings=embedding)
        print(domain)
        raw_dataset_dir=f'/data/lxy/abu/vdbdata/{domain}/test.de'
        save_dir=f'/data/lxy/abu/vdbdata/{domain}/vdb'
        
        if os.path.isfile(raw_dataset_dir):
            raw_data=open(raw_dataset_dir).readlines()
            
        with open(f'/data/lxy/abu/vdbdata/{domain}/pair.jsonl','w') as o:
            bsz=4
            dataloader=DataLoader(raw_data,batch_size=bsz)
            # print(len(dataloader))
            for d in tqdm(dataloader):
                # len(d)是实际的bsz
                
                # 这里返回的是一个batch的result,shape是 bsz x k
                result= r.retrieve(d,k=32)
                # print(result,len(result))
                # 一个batch的数据?
                for i in range(len(d)):
                    for j in range(len(result[i])):
                        tempdata=json.dumps({'de':d[i],'en':result[i][j].page_content},ensure_ascii=False)

                        o.write(tempdata+'\n')
                        # print(tempdata)
                # exit()



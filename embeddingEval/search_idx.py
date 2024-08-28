from retriever import retriver
import json
from argparse import ArgumentParser
import os
from torch.utils.data import DataLoader
import json
from tqdm import tqdm
import sys
sys.path.append('/data/lxy/abu/realvdb')
from MistralEmbeddings import MistralEmbeddings



if __name__ == "__main__":
    embedding = MistralEmbeddings(
            model_name='/data/lxy/abu/e5-mistral7b'
        )
    
    for domain in ['it','koran','law','medical']:
        r=retriver(index_dir=f'/data/lxy/abu/vdbdata/{domain}/vdb',embeddings=embedding)
        print(f'正在为{domain}创建索引')
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
                result= r.retrieve(d,k=10)
                # print(result,len(result))
                # 一个batch的数据?
                for i in range(len(d)):
                    for j in range(len(result[i])):
                        tempdata=json.dumps({'de':d[i],'en':result[i][j].page_content},ensure_ascii=False)

                        o.write(tempdata+'\n')
                        # print(tempdata)
                # exit()

        

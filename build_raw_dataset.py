from langchain.document_loaders import JSONLoader
import json
from faissManager import FAISS

from sent2vec import Sent2VecEmbeddings
from argparse import ArgumentParser
import os


def parse_args():
  parser=ArgumentParser()
  parser.add_argument('--domain',type=str,help='指定需要处理的json文件',default='/home/lxy/wxbdata/doc_js.json')

  return parser.parse_args()

if __name__ == '__main__':
    args=parse_args()
    embedding = Sent2VecEmbeddings(
            model_name='/data/lxy/abu/LaBSE', batch_size=4
        )
    domain=args.domain
    # for domain in ['it','koran','law','medical']:
    print(domain)
    raw_dataset_dir=f'/data/lxy/abu/vdbdata/{domain}/train.en'
    save_dir=f'/data/lxy/abu/vdbdata/{domain}/vdb'
    
    if os.path.isfile(raw_dataset_dir):
        raw_data=open(raw_dataset_dir).readlines()
        # if accelerator.is_main_process:
        #     print(raw_data)


    # 第一次索引找小集合
    raw_db = FAISS.from_idxType_and_documents(
        documents=raw_data,
        embedding=embedding,
        tensorSaveDir=None,
        reuse=False,
        index_type="Flat",
    )

    # print('raw_data',raw_data)  
    raw_db.save_local(save_dir)

        

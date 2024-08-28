import sys
sys.path.append('/data/lxy/abu/realvdb')
from MistralEmbeddings import MistralEmbeddings
import json
from faissManager import FAISS
from NMTEmbeddings import NMTEmbeddings
from LLaMAEmbeddings import LLaMAEmbeddings
from sent2vec import Sent2VecEmbeddings
from argparse import ArgumentParser
import os


def parse_args():
  parser=ArgumentParser()
  parser.add_argument('--domain',type=str,help='指定需要处理的json文件',default='/home/lxy/wxbdata/doc_js.json')

  return parser.parse_args()

if __name__ == '__main__':
    args=parse_args()
    embedding = MistralEmbeddings(
            model_name='/data/lxy/abu/e5-mistral7b'
        )
    domain=args.domain
    print(domain)
    raw_dataset_dir=f'/data/lxy/abu/vdbdata/{domain}/test.en'
    save_dir=f'/data/lxy/abu/realvdb/embeddingEval/{domain}/vdb'
    
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
    )

    # print('raw_data',raw_data)  
    raw_db.save_local(save_dir)

        

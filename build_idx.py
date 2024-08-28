import json
from faissManager import FAISS
# from NMTEmbeddings import NMTEmbeddings
from LLaMAEmbeddings import LLaMAEmbedding
# from sent2vec import Sent2VecEmbeddings
from argparse import ArgumentParser
import os
import config

def parse_args():
  parser=ArgumentParser()
  parser.add_argument('--domain',type=str,help='指定需要处理的json文件',default='koran')
  parser.add_argument('--vdb_type',type=str,default='base')
  parser.add_argument('--prompt_type',type=str)
  return parser.parse_args()

if __name__ == '__main__':
    args=parse_args()
    domain=args.domain
    prefix = "Translate this sentence from German into English and return the translation result only."
    shot = {
        "it": """\nGerman:Zeigt den aktuellen Wert der Feldvariable an.\nEnglish:Displays the current value of the field variable.\nGerman:In diesem Bereich wählen Sie die relativen Größen bezogen auf die Basisgröße.\nEnglish:In this section, you can determine the relative sizes for each type of element with reference to the base size.\nGerman:Geben Sie einen kurzen, beschreibenden Namen für die Schnittstelle ein.\nEnglish:Simply enter a short human-readable description for this device.""",
        "koran": """\nGerman:So führt Gott (im Gleichnis) das Wahre und das Falsche an.\nEnglish:This is how God determines truth and falsehood.\nGerman:Da kamen sie auf ihn zu geeilt.\nEnglish:So the people descended upon him.\nGerman:Wir begehren von euch weder Lohn noch Dank dafür.\nEnglish:We wish for no reward, nor thanks from you.""",
        "law": """\nGerman:Deshalb ist die Regelung von der Ausfuhrleistung abhängig.\nEnglish:In this regard, the scheme is contingent upon export performance.\nGerman:Das Mitglied setzt gleichzeitig den Rat von seinem Beschluß in Kenntnis.\nEnglish:That member shall simultaneously inform the Council of the action it has taken.\nGerman:Dies gilt auch für die vorgeschlagene Sicherheitsleistung.\nEnglish:The same shall apply as regards the security proposed.""",
        "medical": """\nGerman:Das Virus wurde zuerst inaktiviert (abgetötet), damit es keine Erkrankungen verursachen kann.\nEnglish:This may help to protect against the disease caused by the virus.\nGerman:Desirudin ist ein rekombinantes DNS-Produkt, das aus Hefezellen hergestellt wird.\nEnglish:Desirudin is a recombinant DNA product derived from yeast cells.\nGerman:Katzen erhalten eine intramuskuläre Injektion.\nEnglish:In cats, it is given by intramuscular injection.""",
    }
    postfix = "\nGerman:{de}\nEnglish:{en}"
    if args.prompt_type=='few':
        prompt = prefix + shot[domain] + postfix
    else:
        prompt = prefix + postfix
    embedding = LLaMAEmbedding(prompt=prompt,vdb_type=args.vdb_type)
    
    print(domain)
    raw_dataset_dir=f'{config.vdb_path}/{domain}/pair.jsonl'
    
    save_dir=f'{config.vdb_path}/{domain}/{config.vdb[args.vdb_type]}'

    if os.path.isfile(raw_dataset_dir):
        raw_data=list(map(json.loads,open(raw_dataset_dir).readlines()))
    


    # 第一次索引找小集合
    raw_db = FAISS.build_knn_datastore(
        documents=raw_data,
        embedding=embedding,
        tensorSaveDir=None,
        vdb_type=args.vdb_type
    )

    # print('raw_data',raw_data)  
    raw_db.save_local(save_dir)
    print('knn数据库索引成功保存到了',save_dir)

        

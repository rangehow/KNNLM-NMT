from comet import download_model, load_from_checkpoint
from sacrebleu.metrics import BLEU, CHRF, TER
import config
import logging


logger = logging.getLogger(__name__)
log_level = "ERROR"
logger.setLevel(log_level)

folder=['it','koran','law','medical',]
for file in folder:
    print(file)
    src=f'{config.vdb_path}/{file}/test.de'
    # mt=f'{config.vdb_path}/{file}/llama_nmt_knnlm_{file}_output.en'
    mt=f'{config.vdb_path}/{file}/pure_nmt.en'
    ref=f'{config.vdb_path}/{file}/test.en'

    src_data=open(src).readlines()
    mt_data=open(mt).readlines()
    ref_data=open(ref).readlines()



    bleu = BLEU()
    print(bleu.corpus_score(mt_data, [ref_data]))

    # chrf = CHRF()
    # print(chrf.corpus_score(mt_data, [ref_data]))
# comet22

# data = [
#     {
#         "src": "Dem Feuer konnte Einhalt geboten werden",
#         "mt": "The fire could be stopped",
#         "ref": "They were able to control the fire."
#     },
#     {
#         "src": "Schulen und Kindergärten wurden eröffnet.",
#         "mt": "Schools and kindergartens were open",
#         "ref": "Schools and kindergartens opened"
#     }
# ]
    model = load_from_checkpoint(f'{config.comet_path}')
    data=[{"src": s,"mt": m,"ref": r}  for s,m,r in  zip(src_data,mt_data,ref_data)]
    model_output = model.predict(data, batch_size=8, gpus=8)
    print (f'{file} comet22\n',model_output[-1])

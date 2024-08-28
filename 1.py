import pickle

with open("/data/lxy/abu/vdbdata/it/knn_vdb/index.pkl", "rb") as f:
    docstore, index_to_docstore_id = pickle.load(f)

    print(index_to_docstore_id[0])

from faissManager import FAISS
from LLaMAEmbeddings import LLaMAEmbedding



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
        docs = self.db.similarity_search(query, k, score_threshold=score_threshold)

        if debug:
            print(f"过滤掉了{2*k-len(docs)}个距离超出{score_threshold}检索到的文档\n")
        return docs

    def retrieve_with_vector(
        self,
        query,
        k=10,
        score_threshold=None,
        debug=False,
    ):
        return docs


if __name__ == "__main__":
    index_dir = "/data/lxy/abu/vdbdata/koran/knn_vdb"
    # db = FAISS.load_local(index_dir, None)

    embeddings = LLaMAEmbedding(add_eos_token=False)
    r = knnretriver(index_dir, embeddings)
    while True:

        # koran
        query = [
            {
                "de": "Äußerlich kann Levemir InnoLet durch Abwischen mit einem medizinischen Tupfer gereinigt werden.\n",
                "en": "",
            }
        ]  # defrauded them.\n

        # law
        # query=[{"de": "Auf Antrag werden auch Arbeitnehmervertreter einbezogen.\n", "en": "Whereas, moreover, employees' representatives may decide not to seek the setting-up of a European Works Council or the parties concerned may decide on other procedures for the transnational "}]

        # medical
        # query=[{"de": "Xiliarx 50 mg Tabletten\n", "en": "Jalra 50 mg "}] # Jalra 50 mg tablets vildagliptin\n"}

        print(r.retrieve(query, debug=True, k=10))
        exit()

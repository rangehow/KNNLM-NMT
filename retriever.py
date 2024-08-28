from langchain.document_loaders import TextLoader, JSONLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from faissManager import FAISS
from sent2vec import Sent2VecEmbeddings
from langchain.document_transformers import (
    LongContextReorder,
)
from dataclasses import dataclass

# @dataclass
# class template:


class retriver:
    def __init__(self, index_dir, embeddings) -> None:

        self.db = FAISS.load_local(index_dir, embeddings)
        print(
            f"成功加载了向量数据库{index_dir}-------------------------------------------------------------------------------------------------------"
        )

    def retrieve(
        self,
        query,
        k=10,
        score_threshold=1,
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
            query, k, score_threshold=score_threshold
        )

        if debug:
            print(
                f"过滤掉了{2*k-len(docs)}个距离超出{score_threshold}检索到的文档\n"
            )
        return docs
        

if __name__=='__main__':
    index_dir='/data/lxy/abu/vdbdata/medica/vdb'
    model_name='/data/lxy/abu/LaBSE'
    embeddings=Sent2VecEmbeddings()
    r=retriver(index_dir,embeddings)
    while True:
        # query=str(input())
        query=['They pointed out that there are important price differences between the predominantly lower quality silicon exported from Russia, and the higher quality silicon from other third countries.\n', 'Where the Commission considers, after this consultation, that the measure is justified, it shall immediately so inform the Member State which took the initiative and the other Member States.\n', 'Whereas the adaptation of criteria and techniques used for the assessment of the ambient air quality to scientific and technical progress and the arrangements needed to exchange the information to be provided pursuant to this Directive may be desirable; whereas, in order to facilitate implementation of the work necessary to this end, a procedure should be set up to establish close cooperation between the Member States and the Commission within a committee;\n', 'The State aid which the Netherlands is planning to implement for Bodewes Scheepswerven BV, amounting to EUR […], for Scheepswerf Visser, amounting to EUR […], for Bodewes Scheepswerf Volharding Foxhol, amounting to EUR […] and for Scheepswerf De Merwede, amounting to EUR […] is incompatible with the common market.\n', '(31) The current interim review should be terminated without the imposition of measures because only a small part of imports of the product concerned originating in India was dumped and this negligible volume of dumped imports, which is not likely to change significantly in future, cannot cause injury.\n', 'but including that which has not kept its natural round surface, originating in Canada, China, Japan, Korea, Taiwan and the United States of America, may not be introduced into the Community, unless it has undergone an appropriate heat treatment to achieve a minimum wood core temperature of 56 °C for 30 minutes and if accompanied by the certificates prescribed in Articles 7 or 8 of the said Directive;\n', '(a) each seafarer assigned to any of its ships holds an appropriate certificate in accordance which the provisions of this Directive and as established by the Member State;\n', 'For this procedure, besides sugar and isoglucose, other products with a significant sugar equivalent content should be also considered as they could also be possible targets of speculation.\n', 'On this issue Mediaset observed that, in so far as the terms of funding require the public broadcaster to participate in co-productions with independent producers, it is necessary to ensure that such participation does not confer an indirect benefit on the public operator in its relationship with such film producers.\n', 'VISAS\n']
        for q in query:
            print((r.retrieve(q,debug=False,k=1,score_threshold=0.05)[0].page_content[:15],q[:15]))
        exit()
       

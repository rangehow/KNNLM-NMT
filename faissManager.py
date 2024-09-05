"""Wrapper around FAISS vector database."""

from __future__ import annotations
from abc import ABC, abstractmethod
from collections import defaultdict
from enum import Enum
import torch
import operator
import os
import pickle
import uuid
import warnings
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sized,
    Tuple,
    Union,
)
import copy
import numpy as np
import time

# from langchain.docstore.base import AddableMixin, Docstore
# from langchain.docstore.document import Document
# from langchain.docstore.in_memory import InMemoryDocstore
# from langchain.embeddings.base import Embeddings
from langchain.vectorstores.base import VectorStore

# from langchain.vectorstores.utils import DistanceStrategy, maximal_marginal_relevance


def dependable_faiss_import(no_avx2: Optional[bool] = None) -> Any:
    """
    Import faiss if available, otherwise raise error.
    If FAISS_NO_AVX2 environment variable is set, it will be considered
    to load FAISS with no AVX2 optimization.

    Args:
        no_avx2: Load FAISS strictly with no AVX2 optimization
            so that the vectorstore is portable and compatible with other devices.
    """
    if no_avx2 is None and "FAISS_NO_AVX2" in os.environ:
        no_avx2 = bool(os.getenv("FAISS_NO_AVX2"))

    try:
        if no_avx2:
            from faiss import swigfaiss as faiss
        else:
            import faiss
    except ImportError:
        raise ImportError(
            "Could not import faiss python package. "
            "Please install it with `pip install faiss-gpu` (for CUDA supported GPU) "
            "or `pip install faiss-cpu` (depending on Python version)."
        )
    return faiss


def _len_check_if_sized(x: Any, y: Any, x_name: str, y_name: str) -> None:
    """
    相当于一个assert断言加提示了，要求x和y是等长的。
    """
    if isinstance(x, Sized) and isinstance(y, Sized) and len(x) != len(y):
        raise ValueError(
            f"{x_name} and {y_name} expected to be equal length but "
            f"len({x_name})={len(x)} and len({y_name})={len(y)}"
        )
    return


def cosine_similarity(X, Y) -> np.ndarray:
    """Row-wise cosine similarity between two equal-width matrices."""
    if len(X) == 0 or len(Y) == 0:
        return np.array([])
    X = np.array(X)
    Y = np.array(Y)
    if X.shape[1] != Y.shape[1]:
        raise ValueError(
            f"Number of columns in X and Y must be the same. X has shape {X.shape} "
            f"and Y has shape {Y.shape}."
        )

    X_norm = np.linalg.norm(X, axis=1)
    Y_norm = np.linalg.norm(Y, axis=1)
    # Ignore divide by zero errors run time warnings as those are handled below.
    with np.errstate(divide="ignore", invalid="ignore"):
        similarity = np.dot(X, Y.T) / np.outer(X_norm, Y_norm)
    similarity[np.isnan(similarity) | np.isinf(similarity)] = 0.0
    return similarity


def maximal_marginal_relevance(
    query_embedding: np.ndarray,
    embedding_list: list,
    lambda_mult: float = 0.5,
    k: int = 4,
) -> List[int]:
    """Calculate maximal marginal relevance."""
    if min(k, len(embedding_list)) <= 0:
        return []
    if query_embedding.ndim == 1:
        query_embedding = np.expand_dims(query_embedding, axis=0)
    similarity_to_query = cosine_similarity(query_embedding, embedding_list)[0]
    most_similar = int(np.argmax(similarity_to_query))
    idxs = [most_similar]
    selected = np.array([embedding_list[most_similar]])
    while len(idxs) < min(k, len(embedding_list)):
        best_score = -np.inf
        idx_to_add = -1
        similarity_to_selected = cosine_similarity(embedding_list, selected)
        for i, query_score in enumerate(similarity_to_query):
            if i in idxs:
                continue
            redundant_score = max(similarity_to_selected[i])
            equation_score = (
                lambda_mult * query_score - (1 - lambda_mult) * redundant_score
            )
            if equation_score > best_score:
                best_score = equation_score
                idx_to_add = i
        idxs.append(idx_to_add)
        selected = np.append(selected, [embedding_list[idx_to_add]], axis=0)
    return idxs


class DistanceStrategy(str, Enum):
    """Enumerator of the Distance strategies for calculating distances
    between vectors."""

    EUCLIDEAN_DISTANCE = "EUCLIDEAN_DISTANCE"
    MAX_INNER_PRODUCT = "MAX_INNER_PRODUCT"
    DOT_PRODUCT = "DOT_PRODUCT"
    JACCARD = "JACCARD"
    COSINE = "COSINE"


class Document:
    """Class for storing a piece of text and associated metadata."""

    page_content: str
    """String text."""
    metadata: dict
    """Arbitrary metadata about the page content (e.g., source, relationships to other
        documents, etc.).
    """


class InMemoryDocstore:
    """Simple in memory docstore in the form of a dict."""

    def __init__(self, _dict: Optional[Dict[str, Document]] = None):
        """Initialize with dict."""
        self._dict = _dict if _dict is not None else {}

    def add(self, texts: Dict[str, Document]) -> None:
        """Add texts to in memory dictionary.

        Args:
            texts: dictionary of id -> document.

        Returns:
            None
        """
        overlapping = set(texts).intersection(self._dict)
        if overlapping:
            raise ValueError(f"Tried to add ids that already exist: {overlapping}")
        self._dict = {**self._dict, **texts}

    def delete(self, ids: List) -> None:
        """Deleting IDs from in memory dictionary."""
        overlapping = set(ids).intersection(self._dict)
        if not overlapping:
            raise ValueError(f"Tried to delete ids that does not  exist: {ids}")
        for _id in ids:
            self._dict.pop(_id)

    def search(self, search: str) -> Union[str, Document]:
        """Search via direct lookup.

        Args:
            search: id of a document to search for.

        Returns:
            Document if found, else error message.
        """
        if search not in self._dict:
            return f"ID {search} not found."
        else:
            return self._dict[search]


class Embeddings(ABC):
    """Interface for embedding models."""

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Asynchronous Embed search docs."""
        raise NotImplementedError

    async def aembed_query(self, text: str) -> List[float]:
        """Asynchronous Embed query text."""
        raise NotImplementedError


class FAISS(VectorStore):
    """Wrapper around FAISS vector database.

    To use, you should have the ``faiss`` python package installed.

    Example:
        .. code-block:: python

            from langchain import FAISS
            faiss = FAISS(embedding_function, index, docstore, index_to_docstore_id)

    """

    def __init__(
        self,
        embedding_function: Callable,  # 这个方法通常是只用来处理query的，因为所有的classmethod都会预支doc或者doc的embedding
        index: Any,
        docstore: InMemoryDocstore,
        index_to_docstore_id: Dict[int, str],
        relevance_score_fn: Optional[Callable[[float], float]] = None,
        normalize_L2: bool = False,  # 这里本来是false的，这个主要是search的时候用来操纵query的归一化行为的
        distance_strategy: DistanceStrategy = DistanceStrategy.EUCLIDEAN_DISTANCE,
        maxNorm=None,
        # distance_strategy: DistanceStrategy = DistanceStrategy.COSINE,
    ):
        """Initialize with necessary components."""

        self.embedding_function = embedding_function
        self.index = index
        self.docstore = docstore
        self.index_to_docstore_id = index_to_docstore_id
        self.distance_strategy = distance_strategy
        self.override_relevance_score_fn = relevance_score_fn
        self._normalize_L2 = normalize_L2
        self.maxNorm = maxNorm
        if (
            self.distance_strategy != DistanceStrategy.EUCLIDEAN_DISTANCE
            and self._normalize_L2
        ):
            warnings.warn(
                "Normalizing L2 is not applicable for metric type: {strategy}".format(
                    strategy=self.distance_strategy
                )
            )

    @property
    def embeddings(self) -> Optional[Embeddings]:
        # TODO: Accept embeddings object directly
        return None

    def __add(
        self,
        texts: Iterable[str],
        embeddings: Iterable[List[float]],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        # if not isinstance(self.docstore, AddableMixin):
        #     raise ValueError(
        #         "If trying to add texts, the underlying docstore should support "
        #         f"adding items, which {self.docstore} does not"
        #     )
        # 旧特性
        # documents = []
        # for i, text in enumerate(texts):
        #     metadata = metadatas[i] if metadatas else {}
        #     documents.append(Document(page_content=text, metadata=metadata))
        # if ids is None:
        #     ids = [str(uuid.uuid4()) for _ in texts]
        # # Add to the index, the index_to_id mapping, and the docstore.
        # starting_len = len(self.index_to_docstore_id)
        # faiss = dependable_faiss_import()

        # 新特性
        faiss = dependable_faiss_import()
        _len_check_if_sized(texts, metadatas, "texts", "metadatas")
        _metadatas = metadatas or ({} for _ in texts)
        documents = [
            Document(page_content=t, metadata=m) for t, m in zip(texts, _metadatas)
        ]

        _len_check_if_sized(documents, embeddings, "documents", "embeddings")
        _len_check_if_sized(documents, ids, "documents", "ids")

        # Add to the index.

        vector = np.array(embeddings, dtype=np.float32)
        if self._normalize_L2:
            faiss.normalize_L2(vector)
        self.index.add(vector)

        # 旧特性
        # # Get list of index, id, and docs.
        # full_info = [(starting_len + i, ids[i], doc) for i, doc in enumerate(documents)]
        # Add information to docstore and index.
        # self.docstore.add({_id: doc for _, _id, doc in full_info})
        # index_to_id = {index: _id for index, _id, _ in full_info}

        ids = ids or [str(uuid.uuid4()) for _ in texts]
        self.docstore.add({id_: doc for id_, doc in zip(ids, documents)})
        starting_len = len(self.index_to_docstore_id)
        index_to_id = {starting_len + j: id_ for j, id_ in enumerate(ids)}

        self.index_to_docstore_id.update(index_to_id)
        # return [_id for _, _id, _ in full_info]
        return ids

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            ids: Optional list of unique IDs.

        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        # if not isinstance(self.docstore, AddableMixin):
        #     raise ValueError(
        #         "If trying to add texts, the underlying docstore should support "
        #         f"adding items, which {self.docstore} does not"
        #     )
        # Embed and create the documents.
        # Warning！ 不要用这个方法，自己处理好了用add_embeddings
        # 这个设计好蠢……，因为默认embedding_function是query的处理器，所以得串行化送，
        # 这意味着通过这个大幅度增加索引会很慢很慢

        embeddings = [self.embedding_function(text) for text in texts]
        return self.__add(texts, embeddings, metadatas=metadatas, ids=ids, **kwargs)

    def add_embeddings(
        self,
        text_embeddings: Iterable[Tuple[str, List[float]]],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            text_embeddings: Iterable pairs of string and embedding to
                add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            ids: Optional list of unique IDs.

        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        # if not isinstance(self.docstore, AddableMixin):
        #     raise ValueError(
        #         "If trying to add texts, the underlying docstore should support "
        #         f"adding items, which {self.docstore} does not"
        #     )
        # Embed and create the documents.
        texts, embeddings = zip(*text_embeddings)

        return self.__add(texts, embeddings, metadatas=metadatas, ids=ids, **kwargs)

    def similarity_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        fetch_k: int = 20,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query.

        Args:
            embedding: Embedding vector to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter (Optional[Dict[str, Any]]): Filter by metadata. Defaults to None.
            fetch_k: (Optional[int]) Number of Documents to fetch before filtering.
                      Defaults to 20.
            **kwargs: kwargs to be passed to similarity search. Can include:
                score_threshold: Optional, a floating point value between 0 to 1 to
                    filter the resulting set of retrieved docs

        Returns:
            List of documents most similar to the query text and L2 distance
            in float for each. Lower score represents more similarity.
        """
        # 也是危险操作
        # embeddingNorm = torch.norm(embedding, dim=-1, p=2)

        # 感觉应该无论什么时候都直接除以才是正确的？因为是保序变换？但是这样过滤机制可能会比较麻烦，因为文档向量已经被我除过了
        # embedding /= self.maxNorm
        # # 这样写可以支持批量查询
        # if all(torch.ge(embeddingNorm, 1)):
        #     # print('?')
        # embedding /= self.maxNorm
        #     # print(f'查询向量经过{self.maxNorm}归一化前范数是{embeddingNorm}，现在是{torch.norm(embedding,dim=-1,p=2)}')

        faiss = dependable_faiss_import()

        if len(embedding.shape) < 2:
            vector = [embedding]
        else:
            vector = embedding

        # print(vector)
        # print(vector.shape)
        # print(vector)
        # if self._normalize_L2:
        if getattr(kwargs, "normalize_L2", False):
            print("危险操作，faissManager里将要对问题做normalize_L2")
            faiss.normalize_L2(vector)


        scores, indices = self.index.search(vector, k if filter is None else fetch_k)
        # print(scores,indices)

        # print('faiss search time',end-start)
        # print('scores:',scores)
        batch_docs = []
        # 2139  2142 15150  2159 34576
        # print(indices)
        # torch.cuda.synchronize()

        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()
            # print(indices)

        for b in range(len(indices)):
            docs = []
            # for j, i in enumerate(indices[0]):
            for j, i in enumerate(indices[b]):
                if i == -1:
                    # This happens when not enough docs are returned.
                    continue
                _id = self.index_to_docstore_id[i]
                doc = self.docstore.search(_id)
                # if not isinstance(doc, Document):
                #     raise ValueError(f"Could not find document for id {_id}, got {doc}")
                if filter is not None:
                    filter = {
                        key: [value] if not isinstance(value, list) else value
                        for key, value in filter.items()
                    }
                    if all(
                        doc.metadata.get(key) in value for key, value in filter.items()
                    ):
                        docs.append((doc, scores[b][j]))
                else:
                    docs.append((doc, scores[b][j]))
            batch_docs.append(docs)

        # print(batch_docs,len(batch_docs))
        score_threshold = kwargs.get("score_threshold")
        # print(f"faissmanager.py:每个文档的相似性分数分别是：{[similarity for  doc, similarity in docs]}\n")
        if score_threshold is not None:
            # 这个地方相似性阈值，如果是最大内积或者jaccard就是大于？其他就要求小于？
            cmp = (
                operator.ge
                if self.distance_strategy
                # in (DistanceStrategy.MAX_INNER_PRODUCT, DistanceStrategy.JACCARD)
                in (DistanceStrategy.MAX_INNER_PRODUCT, DistanceStrategy.JACCARD)
                else operator.le
            )
            for batch_idx in range(len(batch_docs)):

                batch_docs[batch_idx] = [
                    (doc, similarity)
                    for doc, similarity in batch_docs[batch_idx]
                    if cmp(similarity, score_threshold)
                ]

        # return docs[:k]
        # NOTE 不太清楚这里为什么还要保留前k个,但是会影响batch的召回
        return batch_docs

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        fetch_k: int = 20,
        multiR=False,
        history_embedding=None,
        check_mode=False,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.
            fetch_k: (Optional[int]) Number of Documents to fetch before filtering.
                      Defaults to 20.

        Returns:
            List of documents most similar to the query text with
            L2 distance in float. Lower score represents more similarity.
        """

        sentence_embedding = self.embedding_function(query).to("cpu")

        batch_docs = self.similarity_search_with_score_by_vector(
            sentence_embedding,
            k,
            filter=filter,
            fetch_k=fetch_k,
            **kwargs,
        )

        return batch_docs

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        fetch_k: int = 20,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs most similar to embedding vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.
            fetch_k: (Optional[int]) Number of Documents to fetch before filtering.
                      Defaults to 20.

        Returns:
            List of Documents most similar to the embedding.
        """
        docs_and_scores = self.similarity_search_with_score_by_vector(
            embedding,
            k,
            filter=filter,
            fetch_k=fetch_k,
            **kwargs,
        )
        return [doc for doc, _ in docs_and_scores]

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        fetch_k: int = 20,
        multiR=False,
        history_embedding=None,
        check_mode=False,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.
            fetch_k: (Optional[int]) Number of Documents to fetch before filtering.
                      Defaults to 20.

        Returns:
            List of Documents most similar to the query.
        """

        batch_docs_and_scores = self.similarity_search_with_score(
            query,
            k,
            filter=filter,
            fetch_k=fetch_k,
            multiR=multiR,
            history_embedding=history_embedding,
            check_mode=check_mode,
            **kwargs,
        )

        # 现在这里返回的很抽象,是一个bsz with docs and scores的东西
        return [
            [doc for doc, _ in docs_and_scores]
            for docs_and_scores in batch_docs_and_scores
        ]

    def max_marginal_relevance_search_with_score_by_vector(
        self,
        embedding: List[float],
        *,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
        """Return docs and their similarity scores selected using the maximal marginal
            relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch before filtering to
                     pass to MMR algorithm.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5.
        Returns:
            List of Documents and similarity scores selected by maximal marginal
                relevance and score for each.
        """
        scores, indices = self.index.search(
            np.array([embedding], dtype=np.float32),
            fetch_k if filter is None else fetch_k * 2,
        )
        if filter is not None:
            filtered_indices = []
            for i in indices[0]:
                if i == -1:
                    # This happens when not enough docs are returned.
                    continue
                _id = self.index_to_docstore_id[i]
                doc = self.docstore.search(_id)
                if not isinstance(doc, Document):
                    raise ValueError(f"Could not find document for id {_id}, got {doc}")
                if all(
                    (
                        doc.metadata.get(key) in value
                        if isinstance(value, list)
                        else doc.metadata.get(key) == value
                    )
                    for key, value in filter.items()
                ):
                    filtered_indices.append(i)
            indices = np.array([filtered_indices])
        # -1 happens when not enough docs are returned.
        embeddings = [self.index.reconstruct(int(i)) for i in indices[0] if i != -1]
        mmr_selected = maximal_marginal_relevance(
            np.array([embedding], dtype=np.float32),
            embeddings,
            k=k,
            lambda_mult=lambda_mult,
        )
        selected_indices = [indices[0][i] for i in mmr_selected]
        selected_scores = [scores[0][i] for i in mmr_selected]
        docs_and_scores = []
        for i, score in zip(selected_indices, selected_scores):
            if i == -1:
                # This happens when not enough docs are returned.
                continue
            _id = self.index_to_docstore_id[i]
            doc = self.docstore.search(_id)
            if not isinstance(doc, Document):
                raise ValueError(f"Could not find document for id {_id}, got {doc}")
            docs_and_scores.append((doc, score))
        return docs_and_scores

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch before filtering to
                     pass to MMR algorithm.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5.
        Returns:
            List of Documents selected by maximal marginal relevance.
        """
        docs_and_scores = self.max_marginal_relevance_search_with_score_by_vector(
            embedding, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult, filter=filter
        )
        return [doc for doc, _ in docs_and_scores]

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch before filtering (if needed) to
                     pass to MMR algorithm.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5.
        Returns:
            List of Documents selected by maximal marginal relevance.
        """
        embedding = self.embedding_function(query)
        docs = self.max_marginal_relevance_search_by_vector(
            embedding,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filter=filter,
            **kwargs,
        )
        return docs

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        """Delete by ID. These are the IDs in the vectorstore.

        Args:
            ids: List of ids to delete.

        Returns:
            Optional[bool]: True if deletion is successful,
            False otherwise, None if not implemented.
        """
        if ids is None:
            raise ValueError("No ids provided to delete.")
        missing_ids = set(ids).difference(self.index_to_docstore_id.values())
        if missing_ids:
            raise ValueError(
                f"Some specified ids do not exist in the current store. Ids not found: "
                f"{missing_ids}"
            )

        reversed_index = {id_: idx for idx, id_ in self.index_to_docstore_id.items()}
        index_to_delete = [reversed_index[id_] for id_ in ids]

        self.index.remove_ids(np.array(index_to_delete, dtype=np.int64))
        self.docstore.delete(ids)

        remaining_ids = [
            id_
            for i, id_ in sorted(self.index_to_docstore_id.items())
            if i not in index_to_delete
        ]
        self.index_to_docstore_id = {i: id_ for i, id_ in enumerate(remaining_ids)}

        return True

    def merge_from(self, target: FAISS) -> None:
        """Merge another FAISS object with the current one.

        Add the target FAISS to the current one.

        Args:
            target: FAISS object you wish to merge into the current one

        Returns:
            None.
        """
        # if not isinstance(self.docstore, AddableMixin):
        #     raise ValueError("Cannot merge with this type of docstore")
        # Numerical index for target docs are incremental on existing ones
        starting_len = len(self.index_to_docstore_id)

        # Merge two IndexFlatL2
        self.index.merge_from(target.index)

        # Get id and docs from target FAISS object
        full_info = []
        for i, target_id in target.index_to_docstore_id.items():
            doc = target.docstore.search(target_id)
            if not isinstance(doc, Document):
                raise ValueError("Document should be returned")
            full_info.append((starting_len + i, target_id, doc))

        # Add information to docstore and index_to_docstore_id.
        self.docstore.add({_id: doc for _, _id, doc in full_info})
        index_to_id = {index: _id for index, _id, _ in full_info}
        self.index_to_docstore_id.update(index_to_id)

    @classmethod
    def __from(
        cls,
        texts: List[str],
        embeddings: List[List[float]],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        normalize_L2: bool = False,
        distance_strategy: DistanceStrategy = DistanceStrategy.EUCLIDEAN_DISTANCE,
        **kwargs: Any,
    ) -> FAISS:
        faiss = dependable_faiss_import()

        # 这个写法好cool！但是是旧特性了:(
        # distance_strategy = kwargs.get(
        #     "distance_strategy", DistanceStrategy.EUCLIDEAN_DISTANCE
        # )

        if distance_strategy == DistanceStrategy.MAX_INNER_PRODUCT:
            index = faiss.IndexFlatIP(len(embeddings[0]))
        else:
            # Default to L2, currently other metric types not initialized.
            index = faiss.IndexFlatL2(len(embeddings[0]))

        # 旧特性，和新的区别在于是自行完成docstore的，现在等于说交付给_add完成了
        # vector = np.array(embeddings, dtype=np.float32)
        # if normalize_L2 and distance_strategy == DistanceStrategy.EUCLIDEAN_DISTANCE:
        #     faiss.normalize_L2(vector)
        # # Flat都不用训练，所以直接就可以库库往里头塞数据，如果是自己的魔改索引该咋写
        # index.add(vector)
        # documents = []
        # # 创建文档数量的uuid，那么下面应该要维护一个uuid2text的字典了，不然没法直接从text里取，为啥不直接顺序range啊
        # if ids is None:
        #     ids = [str(uuid.uuid4()) for _ in texts]
        # for i, text in enumerate(texts):
        #     metadata = metadatas[i] if metadatas else {}
        #     documents.append(Document(page_content=text, metadata=metadata))
        # # 看来我还是太年轻了，确实要维护一个字典，不过他很聪明用了顺序id2uuid，这样字典会小很多，不过为什么不直接用顺序的id？
        # index_to_id = dict(enumerate(ids))

        # if len(index_to_id) != len(documents):
        #     raise Exception(
        #         f"{len(index_to_id)} ids provided for {len(documents)} documents."
        #         " Each document should have an id."
        #     )

        # docstore = InMemoryDocstore(dict(zip(index_to_id.values(), documents)))
        # return cls(
        #     embedding.embed_query,
        #     index,
        #     docstore,
        #     index_to_id,
        #     normalize_L2=normalize_L2,
        #     **kwargs,
        # )
        vecstore = cls(
            embedding.embed_query,
            index,
            InMemoryDocstore(),
            {},
            normalize_L2=normalize_L2,
            distance_strategy=distance_strategy,
            **kwargs,
        )
        vecstore.__add(texts, embeddings, metadatas=metadatas, ids=ids)
        return vecstore

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> FAISS:
        """Construct FAISS wrapper from raw documents.

        This is a user friendly interface that:
            1. Embeds documents.
            2. Creates an in memory docstore
            3. Initializes the FAISS database

        This is intended to be a quick way to get started.

        Example:
            .. code-block:: python

                from langchain import FAISS
                from langchain.embeddings import OpenAIEmbeddings
                embeddings = OpenAIEmbeddings()
                faiss = FAISS.from_texts(texts, embeddings)
        """
        # 接入自己的embedding模型: )
        embeddings = embedding.embed_documents(texts)
        return cls.__from(
            texts,
            embeddings,
            embedding,
            metadatas=metadatas,
            ids=ids,
            **kwargs,
        )

    @classmethod
    def from_embeddings(
        cls,
        # text_embeddings: List[Tuple[str, List[float]]],
        text_embeddings: Iterable[Tuple[str, List[float]]],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> FAISS:
        """Construct FAISS wrapper from raw documents.

        和from_document/texts不同之处在于，这个方法必须直接传入encode之后的embedding 新特性里不需要是list了，直接zip进来就行
        text_embeddings=[(embedding,text)],只是给了人们更多格式上的选择，如果采用这个，还是需要根据langchain的embeddings类去写
        因为FAISS还需要保留有query的embedding方法，所以还是要把编码器传上去

        This is a user friendly interface that:
            1. Embeds documents.
            2. Creates an in memory docstore
            3. Initializes the FAISS database

        This is intended to be a quick way to get started.

        Example:
            .. code-block:: python

                from langchain import FAISS
                from langchain.embeddings import OpenAIEmbeddings
                embeddings = OpenAIEmbeddings()
                text_embeddings = embeddings.embed_documents(texts)
                text_embedding_pairs = zip(texts, text_embeddings)
                faiss = FAISS.from_embeddings(text_embedding_pairs, embeddings)
        """
        texts = [t[0] for t in text_embeddings]
        embeddings = [t[1] for t in text_embeddings]
        return cls.__from(
            texts,
            embeddings,
            embedding,
            metadatas=metadatas,
            ids=ids,
            **kwargs,
        )

    @classmethod
    def from_idxType_and_documents(
        cls,
        documents: dict,
        embedding: Embeddings,
        tensorSaveDir: str,
        qa: bool = False,
        query_document: List[str] = None,
        reuse: bool = False,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        index_type: str = "IVF10_HNSW32,FlatDedup",
        efSearch=128,
        do_max_norm=True,
        normalize_L2: bool = False,
        # d=768, 不要这个参数了，直接用模型的
        nprobe=64,
        **kwargs: Any,
    ) -> FAISS:
        """Construct FAISS wrapper from raw documents.

        This is a user friendly interface that:
            1. Embeds documents.
            2. Creates an in memory docstore
            3. Initializes the FAISS database

        This is intended to be a quick way to get started.

        Example:
            .. code-block:: python

                from langchain import FAISS
                from langchain.embeddings import OpenAIEmbeddings
                embeddings = OpenAIEmbeddings()
                faiss = FAISS.from_texts(texts, embeddings)
        """

        assert not (
            normalize_L2 and do_max_norm
        ), "不能同时做l2和最大norm，没啥意义啊感觉"
        # 获取embedding区域+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        texts = documents
        metadatas = None
        # 如果qa开启了，则必须有查询的文本
        assert (qa and query_document is not None) or not (
            qa and query_document is not None
        ), f"{qa} & {query_document}"
        if qa:
            query_texts = [d.page_content for d in query_document]

        # 过滤长文本++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 这里疑似有问题
        # pop_cnt=0
        # if not qa:
        #     # 过滤掉一下超长的文本
        #     for i in range(len(texts)-1,-1,-1):
        #         if len(texts[i])>2048:
        #             texts.pop(i)
        #             pop_cnt+=1
        # else :
        #     # qa模式下一般q都很短，所以根据a一起弹出就行
        #     for i in range(len(texts)-1,-1,-1):
        #         if len(texts[i])>2048:
        #             texts.pop(i)
        #             query_texts.pop(i)
        #             pop_cnt+=1
        # print(f'根据长度弹出了{pop_cnt}个数据')
        # ------------------------------------------------------------------------------------------------

        print(len(texts), max(len(x) for x in texts))

        # 编码文档++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if not reuse:
            if not qa:
                # # 接入自己的embedding模型: ) ,我还返回了norm
                embeddings, maxNorm = embedding.embed_documents(texts)

            else:
                print("编码query中")
                embeddings, maxNorm = embedding.embed_documents(query_texts)

        # print(embeddings,maxNorm)
        # print(texts)
        # --------------------------------------------------------------------------------------------------

        # 按最大范数做归一化+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if torch.max(torch.norm(embeddings, dim=1)) > 1 and do_max_norm:
            embeddings /= maxNorm
            print(
                f"自己编码的向量最大范数原本是{maxNorm},现在变成了{torch.max(torch.norm(embeddings,dim=1))}"
            )
        # ------------------------------------------------------------------------------------------------------

        # 保存/从磁盘加载++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # if not reuse:
        #     # 这里最好先保存一下，因为很多时候吧下面可能会有问题，然后embedding时间开销一般挺大，不在特定的embedding里做而在这里做是因为这样比较普适
        #     # 先关了这个行为吧，每次都保存还挺逆天的，毕竟现在稳定了 23/12/18
        #     # if os.path.exists(tensorSaveDir+'.pt'):
        #     #     print('这个存embedding的文件名是不是重复了……？改成了'+tensorSaveDir+'_dup.pt')
        #     #     torch.save(embeddings,tensorSaveDir+'_dup.pt')
        #     # else:
        #     #     torch.save(embeddings,tensorSaveDir+'.pt')
        #     # torch.save(maxNorm,tensorSaveDir+'_maxNorm.pt')
        #     # print(f'  成功保存到 { os.path.join(os.path.curdir,tensorSaveDir) } ')
        # else:
        #     assert os.path.exists(tensorSaveDir+'.pt')
        #     maxNorm=torch.load(tensorSaveDir+'_maxNorm.pt')
        #     embeddings=torch.load(tensorSaveDir+'.pt')
        #     print('load',embeddings.shape,f'maxNorm={maxNorm}')
        #     if  torch.max(torch.norm(embeddings,dim=1)) >1 and do_max_norm:
        #         embeddings/=maxNorm
        #         print(f'原本复用的张量最大范数是{maxNorm},现在变成了{torch.max(torch.norm(embeddings,dim=1))}')
        # ------------------------------------------------------------------------------------------------------------------------------

        import faiss

        if index_type != "Flat":
            index = faiss.index_factory(embeddings.shape[-1], index_type)
            index.efSearch = efSearch
            index.nprobe = nprobe
            vector = np.array(embeddings, dtype=np.float32)
            if normalize_L2:
                faiss.normalize_L2(vector)
            # res = faiss.StandardGpuResources()
            # # 中间的是gpuid，我默认加载到0卡上√
            # index = faiss.index_cpu_to_gpu(res, 1,index)
            index.train(vector)
            print("faissManager.py: INDEX 训练完成了")
            index.add(vector)
            print("faissManager.py: 数据已添加到索引当中，索引现有元素", index.ntotal)
        else:
            index = faiss.IndexFlatL2(embeddings.shape[-1])
            vector = np.array(embeddings, dtype=np.float32)
            if normalize_L2:
                faiss.normalize_L2(vector)
            index.add(vector)
            print(
                "faissManager.py: 该索引类型不需要训练，数据已添加到索引当中，索引现有元素",
                index.ntotal,
            )

        # 和langchain交互++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        documents = []
        # 创建文档数量的uuid，那么下面应该要维护一个uuid2text的字典了，不然没法直接从text里取，为啥不直接顺序range啊
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
        for i, text in enumerate(texts):
            metadata = metadatas[i] if metadatas else {}
            documents.append(Document(page_content=text, metadata=metadata))
        # 看来我还是太年轻了，确实要维护一个字典，不过他很聪明用了顺序id2uuid，这样字典会小很多，不过为什么不直接用顺序的id？
        index_to_id = dict(enumerate(ids))

        if len(index_to_id) != len(documents):
            raise Exception(
                f"{len(index_to_id)} ids provided for {len(documents)} documents."
                " Each document should have an id."
            )

        docstore = InMemoryDocstore(dict(zip(index_to_id.values(), documents)))
        return cls(
            embedding.embed_query,
            index,
            docstore,
            index_to_id,
            maxNorm=maxNorm,
            **kwargs,
        )

    @classmethod
    def build_knn_datastore(
        cls,
        documents: list[dict],
        embedding: Embeddings,
        tensorSaveDir: str,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        # index_type:str="IVF10_HNSW32,Flat",
        # index_type:str="Flat",
        index_type: str = "IVF256_HNSW32,SQ8",
        efSearch=128,
        do_max_norm=True,
        normalize_L2: bool = False,
        nprobe=128,
        vdb_type=None,
        **kwargs: Any,
    ) -> FAISS:
        """Construct FAISS wrapper from raw documents.

        This is a user friendly interface that:
            1. Embeds documents.
            2. Creates an in memory docstore
            3. Initializes the FAISS database

        This is intended to be a quick way to get started.

        Example:
            .. code-block:: python

                from langchain import FAISS
                from langchain.embeddings import OpenAIEmbeddings
                embeddings = OpenAIEmbeddings()
                faiss = FAISS.from_texts(texts, embeddings)
        """

        embeddings, texts, maxNorm = embedding.embed_documents(documents)
        # print(embeddings)

        # print(embeddings,maxNorm)
        # print(texts)
        # --------------------------------------------------------------------------------------------------

        # 按最大范数做归一化+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if torch.max(torch.norm(embeddings, dim=1)) > 1 and do_max_norm:
            embeddings /= maxNorm
            print(
                f"自己编码的向量最大范数原本是{maxNorm},现在变成了{torch.max(torch.norm(embeddings,dim=1))}"
            )
        # ------------------------------------------------------------------------------------------------------
        # print(embeddings,'??')
        # 保存/从磁盘加载++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # if not reuse:
        #     # 这里最好先保存一下，因为很多时候吧下面可能会有问题，然后embedding时间开销一般挺大，不在特定的embedding里做而在这里做是因为这样比较普适
        #     # 先关了这个行为吧，每次都保存还挺逆天的，毕竟现在稳定了 23/12/18
        #     # if os.path.exists(tensorSaveDir+'.pt'):
        #     #     print('这个存embedding的文件名是不是重复了……？改成了'+tensorSaveDir+'_dup.pt')
        #     #     torch.save(embeddings,tensorSaveDir+'_dup.pt')
        #     # else:
        #     #     torch.save(embeddings,tensorSaveDir+'.pt')
        #     # torch.save(maxNorm,tensorSaveDir+'_maxNorm.pt')
        #     # print(f'  成功保存到 { os.path.join(os.path.curdir,tensorSaveDir) } ')
        # else:
        #     assert os.path.exists(tensorSaveDir+'.pt')
        #     maxNorm=torch.load(tensorSaveDir+'_maxNorm.pt')
        #     embeddings=torch.load(tensorSaveDir+'.pt')
        #     print('load',embeddings.shape,f'maxNorm={maxNorm}')
        #     if  torch.max(torch.norm(embeddings,dim=1)) >1 and do_max_norm:
        #         embeddings/=maxNorm
        #         print(f'原本复用的张量最大范数是{maxNorm},现在变成了{torch.max(torch.norm(embeddings,dim=1))}')
        # ------------------------------------------------------------------------------------------------------------------------------
        # torch.save(embeddings,'embeddings.pt')
        # torch.save(texts,'text.pt')
        # embeddings=torch.load('/data/lxy/abu/realvdb/embeddings.pt')
        # texts=torch.load('/data/lxy/abu/realvdb/text.pt')
        # maxNorm=torch.load('/data/lxy/abu/vdbdata/it/knn_vdb/maxNorm.pt')
        # print(f'原本复用的张量最大范数是{torch.max(torch.norm(embeddings,dim=1))}')
        import faiss

        if index_type != "Flat":
            index = faiss.index_factory(embeddings.shape[-1], index_type)
            index.efSearch = efSearch
            index.nprobe = nprobe
            vector = np.array(embeddings, dtype=np.float32)
            if normalize_L2:
                faiss.normalize_L2(vector)
            start = time.time()

            # res = faiss.StandardGpuResources()
            # # 中间的是gpuid，我默认加载到0卡上√
            # index = faiss.index_cpu_to_gpu(res, 1,index)

            index.train(vector)
            print("faissManager.py: INDEX 训练完成了")
            end = time.time()
            try:
                print("cpu训练时间", end - start)
            except Exception as e:
                print(e)
            index.add(vector)
            print("faissManager.py: 数据已添加到索引当中，索引现有元素", index.ntotal)
        else:
            index = faiss.IndexFlatL2(embeddings.shape[-1])
            vector = np.array(embeddings, dtype=np.float32)
            if normalize_L2:
                faiss.normalize_L2(vector)
            index.add(vector)
            print(
                "faissManager.py: 该索引类型不需要训练，数据已添加到索引当中，索引现有元素",
                index.ntotal,
            )

        # 和langchain交互++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        documents = []
        # 创建文档数量的uuid，那么下面应该要维护一个uuid2text的字典了，不然没法直接从text里取，为啥不直接顺序range啊
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]

        from transformers import AutoTokenizer
        import config

        tokenizer = AutoTokenizer.from_pretrained(
            config.llama_path[vdb_type], model_max_length=4096, add_eos_token=True
        )
        # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        for i, text in enumerate(texts):
            tempdict = {}
            tempdict["page_content"] = int(text)
            # metadata = metadatas[i] if metadatas else {}
            tempdict["metadata"] = tokenizer.decode(int(text))
            documents.append(tempdict)
        # 看来我还是太年轻了，确实要维护一个字典，不过他很聪明用了顺序id2uuid，这样字典会小很多，不过为什么不直接用顺序的id？
        index_to_id = dict(enumerate(ids))
        # print(documents)
        if len(index_to_id) != len(documents):
            raise Exception(
                f"{len(index_to_id)} ids provided for {len(documents)} documents."
                " Each document should have an id."
            )

        docstore = InMemoryDocstore(dict(zip(index_to_id.values(), documents)))
        return cls(
            embedding.embed_query,
            index,
            docstore,
            index_to_id,
            maxNorm=maxNorm,
            **kwargs,
        )

    def save_local(
        self, folder_path: str, index_name: str = "index", model_name=""
    ) -> None:
        """Save FAISS index, docstore, and index_to_docstore_id to disk.

        Args:
            folder_path: folder path to save index, docstore,
                and index_to_docstore_id to.
            index_name: for saving with a specific index file name
        """
        # 保存之前如果不把索引从GPU整到CPU会产生 don't know how to serialize this type of index错误
        faiss = dependable_faiss_import()
        print("faissmanager 这里没问题")
        self.index = faiss.index_gpu_to_cpu(self.index)
        path = Path(folder_path)
        path.mkdir(exist_ok=True, parents=True)

        # save index separately since it is not picklable
        # faiss = dependable_faiss_import()
        faiss.write_index(
            self.index, str(path / "{index_name}.faiss".format(index_name=index_name))
        )

        # save docstore and index_to_docstore_id
        with open(path / "{index_name}.pkl".format(index_name=index_name), "wb") as f:
            pickle.dump((self.docstore, self.index_to_docstore_id), f)

        torch.save(self.maxNorm, os.path.join(folder_path, "maxNorm.pt"))

        if model_name != "":
            torch.save(model_name, os.path.join(folder_path, "modelname.pt"))

    @classmethod
    def load_local(
        cls,
        folder_path: str,
        embeddings: Embeddings = None,
        index_name: str = "index",
        **kwargs: Any,
    ) -> FAISS:
        """Load FAISS index, docstore, and index_to_docstore_id from disk.

        Args:
            folder_path: folder path to load index, docstore,
                and index_to_docstore_id from.
            embeddings: Embeddings to use when generating queries
            index_name: for saving with a specific index file name
        """
        path = Path(folder_path)
        # load index separately since it is not picklable
        faiss = dependable_faiss_import()
        index = faiss.read_index(
            str(path / "{index_name}.faiss".format(index_name=index_name))
        )

        print("这个index里有这么多数据", index.ntotal)
        index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, index)

        maxNorm = torch.load(os.path.join(folder_path, "maxNorm.pt"))

        with open(path / "{index_name}.pkl".format(index_name=index_name), "rb") as f:
            docstore, index_to_docstore_id = pickle.load(f)

        return cls(
            embeddings.embed_query if embeddings else None,
            index,
            docstore,
            index_to_docstore_id,
            maxNorm=maxNorm,
            **kwargs,
        )

    def serialize_to_bytes(self) -> bytes:
        """Serialize FAISS index, docstore, and index_to_docstore_id to bytes."""
        return pickle.dumps((self.index, self.docstore, self.index_to_docstore_id))

    @classmethod
    def deserialize_from_bytes(
        cls,
        serialized: bytes,
        embeddings: Embeddings,
        **kwargs: Any,
    ) -> FAISS:
        """
        Deserialize FAISS index, docstore, and index_to_docstore_id from bytes.
        这个方法看起来能从本地读出pickle dumps的东西，但我还是持有一点

        """
        index, docstore, index_to_docstore_id = pickle.loads(serialized)
        return cls(
            embeddings.embed_query, index, docstore, index_to_docstore_id, **kwargs
        )

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        """
        The 'correct' relevance function
        may differ depending on a few things, including:
        - the distance / similarity metric used by the VectorStore
        - the scale of your embeddings (OpenAI's are unit normed. Many others are not!)
        - embedding dimensionality
        - etc.
        """
        if self.override_relevance_score_fn is not None:
            return self.override_relevance_score_fn

        # Default strategy is to rely on distance strategy provided in
        # vectorstore constructor
        if self.distance_strategy == DistanceStrategy.MAX_INNER_PRODUCT:
            return self._max_inner_product_relevance_score_fn
        elif self.distance_strategy == DistanceStrategy.EUCLIDEAN_DISTANCE:
            # Default behavior is to use euclidean distance relevancy
            return self._euclidean_relevance_score_fn
        else:
            raise ValueError(
                "Unknown distance strategy, must be cosine, max_inner_product,"
                " or euclidean"
            )

    def _similarity_search_with_relevance_scores(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        fetch_k: int = 20,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs and their similarity scores on a scale from 0 to 1."""
        # Pop score threshold so that only relevancy scores, not raw scores, are
        # filtered.
        relevance_score_fn = self._select_relevance_score_fn()
        if relevance_score_fn is None:
            raise ValueError(
                "normalize_score_fn must be provided to"
                " FAISS constructor to normalize scores"
            )
        docs_and_scores = self.similarity_search_with_score(
            query,
            k=k,
            filter=filter,
            fetch_k=fetch_k,
            **kwargs,
        )
        docs_and_rel_scores = [
            (doc, relevance_score_fn(score)) for doc, score in docs_and_scores
        ]
        return docs_and_rel_scores

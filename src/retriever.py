from typing import List, Dict, Tuple
import os
import uuid
import logging
import numpy as np
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO)


def get_random_doc_id():
    return f'_{uuid.uuid4()}'


class LocalEmbeddingRetriever:
    """Local retrieval using FAISS and sentence-transformers embeddings."""

    def __init__(
        self,
        embedding_model: str = 'all-MiniLM-L6-v2',
        collection_name: str = 'default',
    ):
        from sentence_transformers import SentenceTransformer
        import faiss

        self.model = SentenceTransformer(embedding_model)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product (cosine after normalization)
        self.documents: List[str] = []
        self.doc_ids: List[str] = []
        self.collection_name = collection_name

    def add_documents(self, doc_ids: List[str], documents: List[str]):
        """Add documents to the FAISS index."""
        if not documents:
            return
        embeddings = self.model.encode(documents, normalize_embeddings=True)
        self.index.add(np.array(embeddings, dtype=np.float32))
        self.documents.extend(documents)
        self.doc_ids.extend(doc_ids)

    def search(self, queries: List[str], topk: int = 5) -> List[List[Tuple[str, float, str]]]:
        """Search for similar documents."""
        if self.index.ntotal == 0:
            return [[] for _ in queries]
        query_embeddings = self.model.encode(queries, normalize_embeddings=True)
        scores, indices = self.index.search(
            np.array(query_embeddings, dtype=np.float32),
            min(topk, self.index.ntotal)
        )
        results = []
        for i in range(len(queries)):
            query_results = []
            for j in range(len(indices[i])):
                idx = indices[i][j]
                if idx == -1:
                    continue
                query_results.append((self.doc_ids[idx], scores[i][j], self.documents[idx]))
            results.append(query_results)
        return results

    def retrieve(
        self,
        corpus=None,
        queries: Dict[int, str] = None,
        **kwargs,
    ):
        qs = list(queries.values())
        query_texts = [q[0] if isinstance(q, tuple) else q for q in qs]
        topk = kwargs.get('top_k', 10)
        search_results = self.search(query_texts, topk=topk)

        qid2results: Dict[int, Dict[str, Tuple[float, str]]] = {}
        for (qid, query), results in zip(queries.items(), search_results):
            qid2results[qid] = {
                doc_id + get_random_doc_id(): (score, text)
                for doc_id, score, text in results
            }
        return qid2results


class ChromaRetriever:
    """Local retrieval using ChromaDB with sentence-transformers embeddings."""

    def __init__(
        self,
        embedding_model: str = 'all-MiniLM-L6-v2',
        collection_name: str = 'default',
        persist_directory: str = None,
    ):
        import chromadb
        from chromadb.config import Settings

        if persist_directory:
            self.client = chromadb.PersistentClient(path=persist_directory)
        else:
            self.client = chromadb.Client()

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self.embedding_model_name = embedding_model

    def add_documents(self, doc_ids: List[str], documents: List[str]):
        """Add documents to the ChromaDB collection."""
        if not documents:
            return
        self.collection.add(
            ids=doc_ids,
            documents=documents,
        )

    def search(self, queries: List[str], topk: int = 5) -> List[List[Tuple[str, float, str]]]:
        """Search for similar documents."""
        if self.collection.count() == 0:
            return [[] for _ in queries]
        results = self.collection.query(
            query_texts=queries,
            n_results=min(topk, self.collection.count()),
        )
        output = []
        for i in range(len(queries)):
            query_results = []
            if results['ids'] and results['ids'][i]:
                for j in range(len(results['ids'][i])):
                    doc_id = results['ids'][i][j]
                    distance = results['distances'][i][j] if results.get('distances') else 0.0
                    score = 1.0 - distance  # Convert distance to similarity
                    doc_text = results['documents'][i][j] if results.get('documents') else ''
                    query_results.append((doc_id, score, doc_text))
            output.append(query_results)
        return output

    def retrieve(
        self,
        corpus=None,
        queries: Dict[int, str] = None,
        **kwargs,
    ):
        qs = list(queries.values())
        query_texts = [q[0] if isinstance(q, tuple) else q for q in qs]
        topk = kwargs.get('top_k', 10)
        search_results = self.search(query_texts, topk=topk)

        qid2results: Dict[int, Dict[str, Tuple[float, str]]] = {}
        for (qid, query), results in zip(queries.items(), search_results):
            qid2results[qid] = {
                doc_id + get_random_doc_id(): (score, text)
                for doc_id, score, text in results
            }
        return qid2results


class BM25:
    def __init__(
        self,
        tokenizer: AutoTokenizer = None,
        index_name: str = None,
        engine: str = 'faiss',
        embedding_model: str = 'all-MiniLM-L6-v2',
        persist_directory: str = None,
        **kwargs,
    ):
        self.tokenizer = tokenizer
        assert engine in {'faiss', 'chromadb'}
        if engine == 'faiss':
            self.max_ret_topk = 1000
            self.retriever = LocalEmbeddingRetriever(
                embedding_model=embedding_model,
                collection_name=index_name or 'default',
            )
        else:  # chromadb
            self.max_ret_topk = 1000
            self.retriever = ChromaRetriever(
                embedding_model=embedding_model,
                collection_name=index_name or 'default',
                persist_directory=persist_directory,
            )

    def add_documents(self, doc_ids: List[str], documents: List[str]):
        """Add documents to the retriever index."""
        self.retriever.add_documents(doc_ids, documents)

    def retrieve(
        self,
        queries: List[str],  # (bs,)
        filter_ids: List[str] = None,  # (bs,)
        topk: int = 1,
        max_query_length: int = None,
    ):
        assert topk <= self.max_ret_topk
        bs = len(queries)

        # truncate queries
        if max_query_length and self.tokenizer:
            ori_ps = self.tokenizer.padding_side
            ori_ts = self.tokenizer.truncation_side
            self.tokenizer.padding_side = 'left'
            self.tokenizer.truncation_side = 'left'
            tokenized = self.tokenizer(
                queries,
                truncation=True,
                padding=True,
                max_length=max_query_length,
                add_special_tokens=False,
                return_tensors='pt')['input_ids']
            self.tokenizer.padding_side = ori_ps
            self.tokenizer.truncation_side = ori_ts
            queries = self.tokenizer.batch_decode(tokenized, skip_special_tokens=True)

        # retrieve
        filter_ids = filter_ids or ([None] * len(queries))
        results: Dict[str, Dict[str, Tuple[float, str]]] = self.retriever.retrieve(
            None, dict(zip(range(len(queries)), list(zip(queries, filter_ids)))),
            top_k=topk)

        # prepare outputs
        docids: List[str] = []
        docs: List[str] = []
        for qid, query in enumerate(queries):
            _docids: List[str] = []
            _docs: List[str] = []
            if qid in results:
                for did, (score, text) in results[qid].items():
                    _docids.append(did)
                    _docs.append(text)
                    if len(_docids) >= topk:
                        break
            if len(_docids) < topk:  # add dummy docs
                _docids += [get_random_doc_id() for _ in range(topk - len(_docids))]
                _docs += [''] * (topk - len(_docs))
            docids.extend(_docids)
            docs.extend(_docs)

        docids = np.array(docids).reshape(bs, topk)  # (bs, topk)
        docs = np.array(docs).reshape(bs, topk)  # (bs, topk)
        return docids, docs

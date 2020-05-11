#!/usr/bin/env python
# -*- coding:utf-8 -*-

import json
import logging
from multiprocessing import Pool

from arqmath_eval import get_topics, get_judged_documents, get_ndcg, get_random_normalized_ndcg
from gensim.corpora import Dictionary
from gensim.models import FastText, TfidfModel, WordEmbeddingSimilarityIndex
from gensim.similarities import SparseTermSimilarityMatrix
from gensim.similarities.index import AnnoyIndexer
import numpy as np
from tqdm import tqdm

from .common import ArXMLivParagraphIterator, read_json_file
from .configuration import FASTTEXT_PARAMETERS, TERMSIM_MATRIX_PARAMETERS, TERMSIM_INDEX_PARAMETERS, ARXMLIV_NOPROBLEM_FILENAMES, ARXMLIV_NOPROBLEM_HTML5_NUM_PARAGRAPHS, ARQMATH_COLLECTION_POSTS_NUM_DOCUMENTS, POOL_NUM_WORKERS, POOL_CHUNKSIZE


TASK='task1-votes'
TOPN=1000


def get_random_results(task=TASK, subset='train'):
    from random import random
    results = {}
    topic_ids = sorted(get_topics(task=task, subset=subset))
    if subset == 'train':  # subsample the train subset
        topic_ids = topic_ids[:len(topic_ids) // 10]
    for topic_id in topic_ids:
        results[topic_id] = {}
        document_ids = get_judged_documents(task=task, subset=subset, topic=topic_id)
        for document_id in document_ids:
            results[topic_id][document_id] = random()
    return results


def get_results(corpus, task=TASK, subset='train'):
    results = {}
    description = 'Getting results for task {} subset {} topics'.format(task, subset)
    topic_ids = sorted(get_topics(task=task, subset=subset))
    if subset == 'train':  # subsample the train subset
        topic_ids = topic_ids[:len(topic_ids) // 10]
    topics = (corpus[topic_id] for topic_id in topic_ids)
    document_ids_list = [
        list(get_judged_documents(task=task, subset=subset, topic=topic_id))
        for topic_id in topic_ids
    ]
    documents_list = (
        [
            corpus[document_id]
            for document_id in document_ids
        ]
        for document_ids in document_ids_list
    )
    topics_and_documents = zip(topic_ids, topics, document_ids_list, documents_list)
    topics_and_documents = tqdm(topics_and_documents, desc=description, total=len(topic_ids))
    with np.errstate(divide='ignore', invalid='ignore'):
        with Pool(POOL_NUM_WORKERS) as pool:
            ranked_topics_and_documents = pool.imap(get_results_worker, topics_and_documents, POOL_CHUNKSIZE)
            for topic_id, document_ids, similarities in ranked_topics_and_documents:
                results[topic_id] = {
                    document_id: float(similarity)
                    for document_id, similarity
                    in zip(document_ids, similarities)
                }
    return results


def get_results_worker(args):
    topic_id, topic, document_ids, documents = args
    similarities = np.ravel(similarity_matrix.inner_product(topic, documents, normalized=True))
    return topic_id, document_ids, similarities


if __name__ == '__main__':
    logging.basicConfig(level=logging.WARNING)
    for json_filename, fasttext_filename, scm_filename, dictionary_filename, tfidf_filename, corpus_filename, train_result_filename, validation_result_filename, discard_math in tqdm(ARXMLIV_NOPROBLEM_FILENAMES, desc='Training FastText models'):
        try:
            # with open(train_result_filename, 'rt') as f:
            #     pass
            with open(validation_result_filename, 'rt') as f:
                pass
        except IOError:
            paragraphs = ArXMLivParagraphIterator([json_filename], [ARXMLIV_NOPROBLEM_HTML5_NUM_PARAGRAPHS], discard_math)
            try:
                dictionary = Dictionary.load(dictionary_filename)
            except IOError:
                dictionary = Dictionary(paragraphs)
                dictionary.save(dictionary_filename)
            try:
                tfidf = TfidfModel.load(tfidf_filename)
            except IOError:
                tfidf = TfidfModel(dictionary=dictionary, smartirs='dtn')
                tfidf.save(tfidf_filename)
            try:
                similarity_matrix = SparseTermSimilarityMatrix.load(scm_filename)
            except IOError:
                try:
                    model = FastText.load(fasttext_filename)
                except IOError:
                    model = FastText(paragraphs, **FASTTEXT_PARAMETERS)
                    model.save(fasttext_filename)
                annoy_indexer = AnnoyIndexer(model, num_trees=1)
                termsim_index = WordEmbeddingSimilarityIndex(model.wv, kwargs={'indexer': annoy_indexer}, **TERMSIM_INDEX_PARAMETERS)
                similarity_matrix = SparseTermSimilarityMatrix(termsim_index, dictionary, tfidf=tfidf, **TERMSIM_MATRIX_PARAMETERS)
                similarity_matrix.save(scm_filename)
                del model, termsim_index
            relevant_ids = (
                get_topics(task=TASK, subset='train') | get_judged_documents(task=TASK, subset='train') |
                get_topics(task=TASK, subset='validation') | get_judged_documents(task=TASK, subset='validation')
            )
            corpus = {
                document_id: tfidf[dictionary.doc2bow(document)]
                for document_id, document in read_json_file(corpus_filename, ARQMATH_COLLECTION_POSTS_NUM_DOCUMENTS, discard_math)
                if document_id in relevant_ids
            }
            del relevant_ids
            # with open(train_result_filename, 'wt') as f:
            #     train_results = get_results(corpus, subset='train')
            #     ndcg = get_ndcg(train_results, task=TASK, subset='train', topn=TOPN)
            #     line = json.dumps({'ndcg': ndcg})
            #     print(line, file=f)
            #     del train_results
            with open(validation_result_filename, 'wt') as f:
                validation_results = get_results(corpus, subset='validation')
                for topic_id, documents in validation_results.items():
                    top_documents = sorted(documents.items(), key=lambda x: x[1], reverse=True)[:TOPN]
                    for rank, (document_id, similarity) in enumerate(top_documents):
                        line = '{}\txxx\t{}\t{}\t{}\txxx'.format(topic_id, document_id, rank + 1, similarity)
                        print(line, file=f)
                del validation_results, documents, top_documents            
            del dictionary, tfidf, similarity_matrix, corpus

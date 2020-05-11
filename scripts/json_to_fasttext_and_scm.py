#!/usr/bin/env python
# -*- coding:utf-8 -*-

import json
import logging
from multiprocessing import Pool

from arqmath_eval import get_topics, get_judged_documents, get_ndcg
from gensim.corpora import Dictionary
from gensim.models import FastText, TfidfModel, WordEmbeddingSimilarityIndex
from gensim.models.phrases import Phrases, Phraser
from gensim.similarities import SparseTermSimilarityMatrix
from gensim.similarities.index import AnnoyIndexer
import numpy as np
from tqdm import tqdm

from .common import ArXMLivParagraphIterator, read_json_file
from .configuration import POOL_NUM_WORKERS, POOL_CHUNKSIZE, get_configurations


TASK='task1-votes'
TOPN=1000


def get_results(corpus, task=TASK, subset='train'):
    results = {}
    description = 'Getting results for task {} subset {} topics'.format(task, subset)
    topic_ids = sorted(get_topics(task=task, subset=subset))
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
    for configuration in get_configurations():

        json_filename = configuration['json_filename']
        json_num_paragraphs = configuration['json_num_paragraphs']
        dataset_parameters = configuration['dataset_parameters']
        fasttext_parameters = configuration['fasttext_parameters']
        fasttext_filename = configuration['fasttext_filename']
        termsim_matrix_parameters = configuration['termsim_matrix_parameters']
        termsim_index_parameters = configuration['termsim_index_parameters']
        scm_filename = configuration['scm_filename']
        dictionary_filename = configuration['dictionary_filename']
        tfidf_filename = configuration['tfidf_filename']
        phraser_filename = configuration['phraser_filename']
        corpus_filename = configuration['corpus_filename']
        corpus_num_documents = configuration['corpus_num_documents']
        validation_result_filename = configuration['validation_result_filename']
        discard_math = configuration['discard_math']

        try:
            with open(validation_result_filename, 'rt') as f:
                pass
        except IOError:
            paragraphs = ArXMLivParagraphIterator([json_filename], [json_num_paragraphs], discard_math)
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
                    model = FastText(paragraphs, **fasttext_parameters)
                    model.save(fasttext_filename)
                annoy_indexer = AnnoyIndexer(model, num_trees=1)
                termsim_index = WordEmbeddingSimilarityIndex(model.wv, kwargs={'indexer': annoy_indexer}, **termsim_index_parameters)
                similarity_matrix = SparseTermSimilarityMatrix(termsim_index, dictionary, tfidf=tfidf, **termsim_matrix_parameters)
                similarity_matrix.save(scm_filename)
                del model, termsim_index
            relevant_ids = (
                get_topics(task=TASK, subset='train') | get_judged_documents(task=TASK, subset='train') |
                get_topics(task=TASK, subset='validation') | get_judged_documents(task=TASK, subset='validation')
            )
            corpus = {
                document_id: tfidf[dictionary.doc2bow(document)]
                for document_id, document in read_json_file(corpus_filename, corpus_num_documents, discard_math)
                if document_id in relevant_ids
            }
            del relevant_ids
            with open(validation_result_filename, 'wt') as f:
                validation_results = get_results(corpus, subset='validation')
                for topic_id, documents in validation_results.items():
                    top_documents = sorted(documents.items(), key=lambda x: x[1], reverse=True)[:TOPN]
                    for rank, (document_id, similarity) in enumerate(top_documents):
                        line = '{}\txxx\t{}\t{}\t{}\txxx'.format(topic_id, document_id, rank + 1, similarity)
                        print(line, file=f)
                del validation_results, documents, top_documents            
            del dictionary, tfidf, similarity_matrix, corpus

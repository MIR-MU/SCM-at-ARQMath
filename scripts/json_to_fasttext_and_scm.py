#!/usr/bin/env python
# -*- coding:utf-8 -*-

import csv
import logging

from gensim.corpora import Dictionary
from gensim.models import FastText, TfidfModel, WordEmbeddingSimilarityIndex
from gensim.models.phrases import Phrases, Phraser
from gensim.similarities import SparseTermSimilarityMatrix
from gensim.similarities.index import AnnoyIndexer
import numpy as np
from tqdm import tqdm

from .common import ArXMLivParagraphIterator, read_json_file, get_judged_results
from .configuration import POOL_NUM_WORKERS, POOL_CHUNKSIZE, CSV_PARAMETERS, get_fasttext_configurations


def get_judged_results_worker(args):
    topic_id, topic, document_ids, documents = args
    with np.errstate(divide='ignore', invalid='ignore'):
        similarities = np.ravel(similarity_matrix.inner_product(topic, documents, normalized=True))
        return topic_id, document_ids, similarities


if __name__ == '__main__':
    logging.basicConfig(level=logging.WARNING)
    for configuration in get_fasttext_configurations():

        json_filenames = configuration['json_filenames']
        json_nums_paragraphs = configuration['json_nums_paragraphs']
        judged_results = configuration['judged_results']
        topic_judgements = configuration['topic_judgements']
        dataset_parameters = configuration['dataset_parameters']
        fasttext_parameters = configuration['fasttext_parameters']
        fasttext_filename = configuration['fasttext_filename']
        termsim_matrix_parameters = configuration['termsim_matrix_parameters']
        termsim_index_parameters = configuration['termsim_index_parameters']
        scm_filename = configuration['scm_filename']
        dictionary_filename = configuration['dictionary_filename']
        tfidf_filename = configuration['tfidf_filename']
        phraser_filename = configuration['phraser_filename']
        topic_ids = configuration['topic_ids']
        topic_corpus_filename = configuration['topic_corpus_filename']
        topic_corpus_num_documents = configuration['topic_corpus_num_documents']
        document_ids = configuration['document_ids']
        document_corpus_filename = configuration['document_corpus_filename']
        document_corpus_num_documents = configuration['document_corpus_num_documents']
        validation_result_filename = configuration['validation_result_filename']

        min_count = fasttext_parameters['min_count']
        phrases_parameters = {
            'min_count': min_count,
            'delimiter': b' ',
            'threshold': 100.0,  # This is the value used by word2vec's word2phrase.c tool
        }

        discard_math = configuration['discard_math']
        concat_math = False
        reader_args = [json_filenames, json_nums_paragraphs]
        reader_kwargs = {
            'discard_math': discard_math,
            'concat_math': concat_math,
        }

        tfidf_parameters = {
            'smartirs': 'dtn',
        }

        try:
            with open(validation_result_filename, 'rt') as f:
                pass
        except IOError:
            paragraphs = ArXMLivParagraphIterator(*reader_args, **reader_kwargs)

            phraser = None
            phraser_num_iterations = dataset_parameters['phrases']
            if phraser_num_iterations > 0:
                try:
                    phraser = Phraser.load(phraser_filename.format(phraser_num_iterations))
                except IOError:
                    transformed_paragraphs = paragraphs
                    for phraser_iteration in tqdm(range(phraser_num_iterations), 'Modeling phrases'):
                        try:
                            phraser = Phraser.load(phraser_filename.format(phraser_iteration + 1))
                        except IOError:
                            phraser = Phraser(Phrases(transformed_paragraphs, **phrases_parameters))
                            phraser.save(phraser_filename.format(phraser_iteration + 1))
                        reader_kwargs['phraser'] = phraser
                        transformed_paragraphs = ArXMLivParagraphIterator(*reader_args, **reader_kwargs)
                    del transformed_paragraphs
                reader_kwargs['phraser'] = phraser
                paragraphs = ArXMLivParagraphIterator(*reader_args, **reader_kwargs)

            try:
                dictionary = Dictionary.load(dictionary_filename)
            except IOError:
                dictionary = Dictionary(paragraphs)
                dictionary.save(dictionary_filename)

            try:
                tfidf = TfidfModel.load(tfidf_filename)
            except IOError:
                tfidf = TfidfModel(dictionary=dictionary, **tfidf_parameters)
                tfidf.save(tfidf_filename)
            termsim_matrix_parameters['tfidf'] = tfidf

            try:
                similarity_matrix = SparseTermSimilarityMatrix.load(scm_filename)
            except IOError:
                try:
                    model = FastText.load(fasttext_filename, mmap='r')
                except IOError:
                    model = FastText(paragraphs, **fasttext_parameters)
                    model.save(fasttext_filename)
                annoy_indexer = AnnoyIndexer(model, num_trees=1)
                termsim_index_parameters = {**termsim_index_parameters, **{'kwargs': {'indexer': annoy_indexer}}}
                termsim_index = WordEmbeddingSimilarityIndex(model.wv, **termsim_index_parameters)
                similarity_matrix = SparseTermSimilarityMatrix(termsim_index, dictionary, **termsim_matrix_parameters)
                similarity_matrix.save(scm_filename)
                del model, termsim_index

            topic_corpus = dict()
            document_corpus = dict()
            for topic_id, topic in read_json_file(topic_corpus_filename, topic_corpus_num_documents, **reader_kwargs):
                if topic_id in topic_ids:
                    topic_corpus[topic_id] = tfidf[dictionary.doc2bow(topic)]
                if topic_corpus_filename == document_corpus_filename:
                    document_id = topic_id
                    document = topic
                    if document_id in document_ids:
                        document_corpus[document_id] = tfidf[dictionary.doc2bow(document)]
            if topic_corpus_filename != document_corpus_filename:
                for document_id, document in read_json_file(document_corpus_filename, document_corpus_num_documents, **reader_kwargs):
                    if document_id in document_ids:
                        document_corpus[document_id] = tfidf[dictionary.doc2bow(document)]

            with open(validation_result_filename, 'wt') as f:
                csv_writer = csv.writer(f, **CSV_PARAMETERS)
                if judged_results:
                    results = get_judged_results(topic_corpus, document_corpus, topic_judgements, get_judged_results_worker)
                else:
                    assert False  # FIXME
                for topic_id, top_documents in results:
                    for rank, (document_id, similarity) in enumerate(top_documents):
                        if judged_results:
                            row = (topic_id, 'xxx', document_id, rank + 1, similarity, 'xxx')
                        else:
                            row = (topic_id, document_id, rank + 1, similarity, 'Run_SCM_0')
                        csv_writer.writerow(row)

            del dictionary, tfidf, similarity_matrix, corpus

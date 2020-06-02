#!/usr/bin/env python
# -*- coding:utf-8 -*-

import csv
import logging

from gensim.models import Doc2Vec
from gensim.models.phrases import Phrases, Phraser
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .common import ArXMLivParagraphIterator, read_json_file, get_judged_results, TASK, SUBSET
from .configuration import POOL_NUM_WORKERS, POOL_CHUNKSIZE, CSV_PARAMETERS, get_doc2vec_configurations


def get_judged_results_worker(args):
    global model
    topic_id, topic, document_ids, documents = args
    topic = np.array([
        model.infer_vector(topic)
    ])
    documents = np.array([
        model.infer_vector(document)
        for document in documents
    ])
    similarities = np.ravel(cosine_similarity(topic, documents))
    return topic_id, document_ids, similarities


if __name__ == '__main__':
    logging.basicConfig(level=logging.WARNING)
    for configuration in get_doc2vec_configurations():

        json_filenames = configuration['json_filenames']
        json_nums_paragraphs = configuration['json_nums_paragraphs']
        judged_results = configuration['judged_results']
        topic_judgements = configuration['topic_judgements']
        dataset_parameters = configuration['dataset_parameters']
        doc2vec_parameters = configuration['doc2vec_parameters']
        doc2vec_filename = configuration['doc2vec_filename']
        phraser_filename = configuration['phraser_filename']
        topic_ids = configuration['topic_ids']
        topic_corpus_filename = configuration['topic_corpus_filename']
        topic_corpus_num_documents = configuration['topic_corpus_num_documents']
        document_ids = configuration['document_ids']
        document_corpus_filename = configuration['document_corpus_filename']
        document_corpus_num_documents = configuration['document_corpus_num_documents']
        validation_result_filename = configuration['validation_result_filename']

        min_count = doc2vec_parameters['min_count']
        phrases_parameters = {
            'min_count': min_count,
            'delimiter': b' ',
            'threshold': 100.0,  # This is the value used by word2vec's word2phrase.c tool
        }

        discard_math = configuration['discard_math']
        concat_math = False
        reader_args = [json_filenames, [json_nums_paragraphs]]
        reader_kwargs = {
            'discard_math': discard_math,
            'concat_math': concat_math,
            'tagged_documents': True,
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
                model = Doc2Vec.load(doc2vec_filename, mmap='r')
            except IOError:
                model = Doc2Vec(paragraphs, **doc2vec_parameters)
                model.save(doc2vec_filename)
            model.delete_temporary_training_data(keep_doctags_vectors=False, keep_inference=True)

            topic_corpus = dict()
            document_corpus = dict()
            for topic_id, topic in read_json_file(topic_corpus_filename, topic_corpus_num_documents, **reader_kwargs):
                if topic_id in topic_ids:
                    topic_corpus[topic_id] = topic
                if topic_corpus_filename == document_corpus_filename:
                    document_id = topic_id
                    document = topic
                    if document_id in document_ids:
                        document_corpus[document_id] = document
            if topic_corpus_filename != document_corpus_filename:
                for document_id, document in read_json_file(document_corpus_filename, document_corpus_num_documents, **reader_kwargs):
                    if document_id in document_ids:
                        document_corpus[document_id] = document

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
                            row = (topic_id, document_id, rank + 1, similarity, 'Run_Formula2Vec_0')
                        csv_writer.writerow(row)

            del model, corpus

#!/usr/bin/env python
# -*- coding:utf-8 -*-

import csv
import logging

from gensim.models import Doc2Vec
from gensim.models.phrases import Phrases, Phraser
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from .common import ArXMLivParagraphIterator, read_corpora, get_results
from .configuration import CSV_PARAMETERS, get_doc2vec_configurations


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    for configuration in get_doc2vec_configurations():

        result_type = configuration['result_type']
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
        reader_args = [json_filenames, json_nums_paragraphs]
        reader_kwargs = {
            'discard_math': discard_math,
            'concat_math': concat_math,
            'tagged_documents': True,
        }
        phraser_reader_kwargs = {
            **reader_kwargs,
            'tagged_documents': False,
        }

        try:
            with open(validation_result_filename, 'rt') as f:
                pass
        except IOError:
            phraser = None
            phraser_num_iterations = dataset_parameters['phrases']
            if phraser_num_iterations > 0:
                try:
                    phraser = Phraser.load(phraser_filename.format(phraser_num_iterations))
                except IOError:
                    transformed_paragraphs = ArXMLivParagraphIterator(*reader_args, **phraser_reader_kwargs)
                    for phraser_iteration in tqdm(range(phraser_num_iterations), 'Modeling phrases'):
                        try:
                            phraser = Phraser.load(phraser_filename.format(phraser_iteration + 1))
                        except IOError:
                            phraser = Phraser(Phrases(transformed_paragraphs, **phrases_parameters))
                            phraser.save(phraser_filename.format(phraser_iteration + 1))
                        reader_kwargs['phraser'] = phraser
                        transformed_paragraphs = ArXMLivParagraphIterator(*reader_args, **phraser_reader_kwargs)
                    del transformed_paragraphs
                reader_kwargs['phraser'] = phraser

            paragraphs = ArXMLivParagraphIterator(*reader_args, **reader_kwargs)

            try:
                model = Doc2Vec.load(doc2vec_filename, mmap='r')
            except IOError:
                model = Doc2Vec(paragraphs, **doc2vec_parameters)
                model.save(doc2vec_filename)
            model.delete_temporary_training_data(keep_doctags_vectors=False, keep_inference=True)

            def topic_and_document_transformer(topic_or_document):
                return model.infer_vector(topic_or_document)
            
            topic_corpus, document_corpus = read_corpora({
                'topic_corpus_filename': topic_corpus_filename,
                'topic_corpus_num_documents': topic_corpus_num_documents,
                'topic_ids': topic_ids,
                'topic_transformer': topic_and_document_transformer,
                'document_corpus_filename': document_corpus_filename,
                'document_corpus_num_documents': document_corpus_num_documents,
                'document_ids': document_ids,
                'document_transformer': topic_and_document_transformer,
                'parallelize_transformers': True,
            }, reader_kwargs)

            def get_results_1N_worker(args):
                topic_id, topic, document_ids, documents = args
                topic = np.array([topic])
                documents = np.array(documents)
                similarities = np.ravel(cosine_similarity(topic, documents))
                assert len(document_ids) == similarities.size
                return (topic_id, document_ids, similarities)

            def get_results_MN_worker(args):
                topic_ids, topics, document_ids, documents = args
                topics = np.array(topics)
                documents = np.array(documents)
                similarities_list = cosine_similarity(topics, documents)
                assert (len(topic_ids), len(document_ids)) == similarities_list.shape
                for topic_index, topic_id in enumerate(topic_ids):
                    similarities = similarities_list[topic_index]
                    assert len(document_ids) == similarities.size
                    yield (topic_id, document_ids, similarities)

            if result_type == 'judged':
                run_name = 'xxx'
            elif result_type == 'task1':
                run_name = 'Run_Formula2Vec_2'
            elif result_type == 'task2':
                run_name = 'Run_Formula2Vec_3'

            with open(validation_result_filename, 'wt') as f:
                csv_writer = csv.writer(f, **CSV_PARAMETERS)
                results = get_results(topic_corpus, document_corpus, topic_judgements, get_results_1N_worker, get_results_MN_worker)
                for topic_id, top_documents_and_similarities in results:
                    if isinstance(topic_id, tuple):
                        _, topic_id = topic_id
                    for rank, (document_id, similarity) in enumerate(top_documents_and_similarities):
                        if judged_results:
                            row = (topic_id, 'xxx', document_id, rank + 1, similarity, run_name)
                        else:
                            if isinstance(document_id, tuple):
                                formula_id, post_id = document_id
                                row = (topic_id, formula_id, post_id, rank + 1, similarity, run_name)
                            else:
                                row = (topic_id, document_id, rank + 1, similarity, run_name)
                        csv_writer.writerow(row)

            del model, topic_corpus, document_corpus

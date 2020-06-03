# -*- coding:utf-8 -*-

import gzip
from itertools import chain, repeat
import json
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor
import re

from gensim.models.doc2vec import TaggedDocument
from tqdm import tqdm

from .configuration import POOL_CHUNKSIZE, POOL_NUM_WORKERS, TOPN


JSON_LINE_REGEX = re.compile(r'"(?P<document_name>[^"]*)": (?P<json_document>.*),')


def get_results(topic_corpus, document_corpus, topic_judgements, get_results_1N_worker, get_results_MN_worker=None):
    topic_ids = sorted(topic_corpus.keys())
    topics = [topic_corpus[topic_id] for topic_id in topic_ids]
    if topic_judgements is None:
        document_ids = sorted(document_corpus.keys())
        documents = [document_corpus[document_id] for document_id in document_ids]
        document_ids_list = repeat(document_ids)
        documents_list = repeat(documents)
    else:
        document_ids_list = (sorted(topic_judgements[topic_id]) for topic_id in topic_ids)
        documents_list = (
            [document_corpus[document_id] for document_id in document_ids]
            for document_ids in document_ids_list
        )

    tqdm_kwargs = {
        'desc': 'Getting judged results',
        'total': len(topic_ids),
    }

    if topic_judgements is not None or get_results_MN_worker is None:
        topics_and_documents = zip(topic_ids, topics, document_ids_list, documents_list)
        topics_and_documents = tqdm(topics_and_documents, **tqdm_kwargs)
    else:
        topics_and_documents = (topic_ids, topics, document_ids, documents)

    with Pool(POOL_NUM_WORKERS) as pool:
        if topic_judgements is None:
            if get_results_MN_worker is None:
                ranked_topics_and_documents = map(get_results_1N_worker, topics_and_documents)
            else:
                ranked_topics_and_documents = tqdm(get_results_MN_worker(topics_and_documents), **tqdm_kwargs)
        else:
            ranked_topics_and_documents = pool.imap(get_results_1N_worker, topics_and_documents, POOL_CHUNKSIZE)
        for topic_id, document_ids, similarities in ranked_topics_and_documents:
            documents_and_similarities = zip(document_ids, similarities)
            top_documents_and_similarities = get_topn(documents_and_similarities)
            yield (topic_id, top_documents_and_similarities)


def get_topn(documents_and_similarities):
    documents = (
        (document, float(similarity))
        for document, similarity
        in documents_and_similarities
    )
    top_documents_and_similarities = sorted(documents, key=lambda x: x[1], reverse=True)[:TOPN]
    return top_documents_and_similarities


def read_corpora(configuration, reader_kwargs):
    topic_corpus_filename = configuration['topic_corpus_filename']
    topic_corpus_num_documents = configuration['topic_corpus_num_documents']
    topic_ids = configuration['topic_ids']
    topic_transformer = configuration['topic_transformer']

    document_corpus_filename = configuration['document_corpus_filename']
    document_corpus_num_documents = configuration['document_corpus_num_documents']
    document_ids = configuration['document_ids']
    document_transformer = configuration['document_transformer']

    parallelize_transformers = configuration['parallelize_transformers']

    topic_corpus = dict()
    document_corpus = dict()

    with ProcessPoolExecutor(POOL_NUM_WORKERS) as executor:

        if parallelize_transformers:
            def transform_document(document): return executor.submit(document_transformer, document)
            def transform_topic(topic): return executor.submit(topic_transformer, topic)
        else:
            transform_document = document_transformer
            transform_topic = topic_transformer

        for topic_id, topic in read_json_file(topic_corpus_filename, topic_corpus_num_documents, **reader_kwargs):
            if topic_id in topic_ids:
                topic_corpus[topic_id] = transform_topic(topic)
            if topic_corpus_filename == document_corpus_filename:
                document_id = topic_id
                document = topic
                if document_id in document_ids:
                    document_corpus[document_id] = transform_document(document)

        assert len(topic_ids) == len(topic_corpus)

        if topic_corpus_filename != document_corpus_filename:
            for document_id, document in read_json_file(document_corpus_filename, document_corpus_num_documents, **reader_kwargs):
                if document_id in document_ids:
                    document_corpus[document_id] = transform_document(document)

        assert len(document_ids) == len(document_corpus)

        if parallelize_transformers:
            for corpus, (key, future) in tqdm(
                        chain(
                            zip(repeat(topic_corpus), topic_corpus.items()),
                            zip(repeat(document_corpus), document_corpus.items()),
                        ),
                        desc='Waiting for topic and document transformers to finish',
                        total=len(topic_corpus) + len(document_corpus),
                    ):
                corpus[key] = future.result()

    return (topic_corpus, document_corpus)


def read_json_file(filename, total_number_of_documents, discard_math=False, concat_math=False, phraser=None, **kwargs):
    number_of_documents = 0
    with gzip.open(filename, 'rt') as f:
        with Pool(POOL_NUM_WORKERS) as pool:
            for result in pool.imap(
                        read_json_file_worker,
                        tqdm(
                            zip(f, repeat(discard_math), repeat(concat_math)),
                            desc='Reading documents from {}'.format(filename),
                            total=total_number_of_documents,
                        ),
		        POOL_CHUNKSIZE,
                    ):
                if result is not None:
                    number_of_documents += 1
                    assert number_of_documents <= total_number_of_documents, \
                        'Expected {} documents, but just read document number {}'.format(
                            total_number_of_documents,
                            number_of_documents,
                        )
                    document_name, document = result
                    if phraser is not None:
                        document = phraser[document]
                    yield (document_name, document)
    assert number_of_documents == total_number_of_documents, \
        'Expected {} documents, but read only {}'.format(
            total_number_of_documents,
            number_of_documents,
        )


def read_json_file_worker(args):
    line, discard_math, concat_math = args
    line = line.strip()
    if line in ('{', '}'):
        return None
    match = re.fullmatch(JSON_LINE_REGEX, line)
    document_name = match.group('document_name')
    document = json.loads(match.group('json_document'))

    def concatenate_tokens(tokens):
        if tokens:
            token_type = tokens[0][:5]
            assert all(token.startswith(token_type) for token in tokens)
            concatenated_tokens = [
                '{}{}'.format(
                    token_type,
                    '_'.join(token[5:] for token in tokens),
                )
            ]
        else:
            concatenated_tokens = []
        return concatenated_tokens

    def preprocess_token(token):
        assert token.startswith('text:') or token.startswith('math:'), 'Unknown type of token {}'.format(token)
        if token.startswith('text:'):
            return token[5:].lower()
        else:
            return token[5:].upper()

    if not discard_math and concat_math:
        math_token_buffer = []
        concatenated_document = []
        for token in document:
            if token.startswith('math:'):
                math_token_buffer.append(token)
            else:
                concatenated_math_tokens = concatenate_tokens(math_token_buffer)
                math_token_buffer.clear()
                concatenated_document.extend(concatenated_math_tokens)
                concatenated_document.append(token)
        document = concatenated_document

    document = [
        preprocess_token(token)
        for token
        in document
        if not (discard_math and token.startswith('math:'))
    ]
    return (document_name, document)


class ArXMLivParagraphIterator():
    def __init__(self, filenames, numbers_of_paragraphs, discard_math=False, concat_math=False, phraser=None, tagged_documents=False):
        self.filenames = list(reversed(filenames))
        self.remaining_filenames = list(self.filenames)
        self.numbers_of_paragraphs = list(reversed(numbers_of_paragraphs))
        self.remaining_numbers_of_paragraphs = list(self.numbers_of_paragraphs)
        assert len(self.remaining_filenames) == len(self.numbers_of_paragraphs)
        self.discard_math = discard_math
        self.concat_math = concat_math
        self.phraser = phraser
        self.tagged_documents = tagged_documents
        self.iterable = None

    def __iter__(self):
        self.__init__(
            list(reversed(self.filenames)),
            list(reversed(self.numbers_of_paragraphs)),
            discard_math=self.discard_math,
            concat_math=self.concat_math,
            phraser=self.phraser,
            tagged_documents=self.tagged_documents,
        )
        return self

    def next_file(self):
        if not self.remaining_filenames:
            raise StopIteration()
        self.iterable = read_json_file(
            self.remaining_filenames.pop(),
            self.remaining_numbers_of_paragraphs.pop(),
            discard_math=self.discard_math,
            concat_math=self.concat_math,
            phraser=self.phraser,
        )

    def __next__(self):
        if self.iterable is None:
            self.next_file()
        paragraph = None
        while paragraph is None:
            try:
                paragraph_name, paragraph = next(self.iterable)
            except StopIteration:
                self.next_file()
        if self.tagged_documents:
            return TaggedDocument(words=paragraph, tags=[paragraph_name])
        else:
            return paragraph

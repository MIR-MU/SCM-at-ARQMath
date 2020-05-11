# -*- coding:utf-8 -*-

import gzip
from itertools import repeat
import json
from multiprocessing import Pool
import re

from tqdm import tqdm

from .configuration import POOL_CHUNKSIZE, POOL_NUM_WORKERS


JSON_LINE_REGEX = re.compile(r'"(?P<document_name>[^"]*)": (?P<json_document>.*),')


def read_json_file(filename, total_number_of_documents, discard_math=False):
    number_of_documents = 0
    with gzip.open(filename, 'rt') as f:
        with Pool(POOL_NUM_WORKERS) as pool:
            for result in pool.imap(
                        read_json_file_worker,
                        tqdm(
                            zip(f, repeat(discard_math)),
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
                    yield (document_name, document)
    assert number_of_documents == total_number_of_documents, \
        'Expected {} documents, but read only {}'.format(
            total_number_of_documents,
            number_of_documents,
        )


def read_json_file_worker(args):
    line, discard_math = args
    line = line.strip()
    if line in ('{', '}'):
        return None
    match = re.fullmatch(JSON_LINE_REGEX, line)
    document_name = match.group('document_name')
    document = json.loads(match.group('json_document'))

    def preprocess_token(token):
        assert token.startswith('text:') or token.startswith('math:'), 'Unknown type of token {}'.format(token)
        if token.startswith('text:'):
            return token[5:].lower()
        else:
            return token[5:].upper()

    document = [
        preprocess_token(token)
        for token
        in document
        if not (discard_math and token.startswith('math:'))
    ]
    return (document_name, document)


class ArXMLivParagraphIterator():
    def __init__(self, filenames, numbers_of_paragraphs, discard_math=False):
        self.filenames = list(reversed(filenames))
        self.remaining_filenames = list(self.filenames)
        self.numbers_of_paragraphs = list(reversed(numbers_of_paragraphs))
        self.remaining_numbers_of_paragraphs = list(self.numbers_of_paragraphs)
        assert len(self.remaining_filenames) == len(self.numbers_of_paragraphs)
        self.discard_math = discard_math
        self.iterable = None

    def __iter__(self):
        self.__init__(
            list(reversed(self.filenames)),
            list(reversed(self.numbers_of_paragraphs)),
            self.discard_math,
        )
        return self

    def next_file(self):
        if not self.remaining_filenames:
            raise StopIteration()
        self.iterable = read_json_file(
            self.remaining_filenames.pop(),
            self.remaining_numbers_of_paragraphs.pop(),
            self.discard_math,
        )

    def __next__(self):
        if self.iterable is None:
            self.next_file()
        paragraph = None
        while paragraph is None:
            try:
                _, paragraph = next(self.iterable)
            except StopIteration:
                self.next_file()
        return paragraph

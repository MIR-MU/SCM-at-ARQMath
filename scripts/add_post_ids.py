#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""Readds post ids into an ARQMath task 2 SERP."""

import csv
import logging
import sys

from .common import read_tsv_file
from .configuration import CSV_PARAMETERS, ARQMATH_COLLECTION_FORMULAE_FILENAMES, ARQMATH_COLLECTION_FORMULAE_NUM_FORMULAE


FORMULAE_FILENAME = ARQMATH_COLLECTION_FORMULAE_FILENAMES['slt']
# FORMULAE_FILENAME = '/mnt/storage/ARQMath_CLEF2020/Formulas/latex_representation/all.tsv'
FORMULAE_NUM_FORMULAE = ARQMATH_COLLECTION_FORMULAE_NUM_FORMULAE['slt']
# FORMULAE_NUM_FORMULAE = 28320920
LOGGER = logging.getLogger(__name__)


def add_post_ids(input_filename, output_filename, post_ids):
    rows = []
    with open(input_filename, 'rt', newline='') as csv_file:
        csv_reader = csv.reader(csv_file, **CSV_PARAMETERS)
        for row in csv_reader:
            topic_id, formula_id, _, rank, score, description = row
            rank = int(rank)
            score = float(score)
            post_id = post_ids[formula_id]
            row = (topic_id, formula_id, post_id, rank, score, description)
            rows.append(row)

    with open(output_filename, 'wt', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, **CSV_PARAMETERS)
        for row in rows:
            csv_writer.writerow(row)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    post_ids = {
        formula_id: post_id
        for (formula_id, post_id), _
        in read_tsv_file(FORMULAE_FILENAME, FORMULAE_NUM_FORMULAE) 
    }
    for filename in sys.argv[1:]:
        LOGGER.info('Adding post ids to {}'.format(filename))
        add_post_ids(filename, filename, post_ids)

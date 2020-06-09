#!/usr/bin/env python
# -*- coding:utf-8 -*-

import csv
from statistics import median
import sys

from .configuration import CSV_PARAMETERS, TOPN


RUN_NAME = 'Run_Ensemble_0'


def combine_serps(input_filenames, output_filename, task):
    assert task in ('task1', 'task2')

    num_systems = len(input_filenames)

    def inverse_median_rank(ranks):
        ranks = [*ranks, *[TOPN] * (num_systems - len(ranks))]
        return float(TOPN - median(ranks)) / TOPN

    separate_results = dict()
    for input_filename in input_filenames:
        with open(input_filename, 'rt', newline='') as csv_file:
            csv_reader = csv.reader(csv_file, **CSV_PARAMETERS)
            for row in csv_reader:
                topic_id = row[0]
                if topic_id not in separate_results:
                    separate_results[topic_id] = dict()
                if task == 'task1':
                    _, post_id, rank, __, ___ = row
                    identifier = (post_id,)
                elif task == 'task2':
                    _, formula_id, post_id, rank, __, ___ = row
                    identifier = (formula_id, post_id)
                rank = int(rank) - 1
                assert rank >= 0
                assert rank < TOPN
                if identifier not in separate_results[topic_id]:
                    separate_results[topic_id][identifier] = []
                separate_results[topic_id][identifier].append(rank)

    with open(output_filename, 'wt', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, **CSV_PARAMETERS)
        for topic_id, identifiers in separate_results.items():
            ensemble_results = dict()
            for identifier, ranks in identifiers.items():
                ensemble_results[identifier] = inverse_median_rank(ranks)
            ensemble_results = sorted(ensemble_results.items(), key=lambda x: (x[1], x[0]), reverse=True)
            for rank, (identifier, score) in enumerate(ensemble_results[:TOPN]):
                csv_writer.writerow((topic_id, *identifier, rank + 1, score, RUN_NAME))


if __name__ == '__main__':
    input_filenames = sys.argv[1:-1]
    output_filename = sys.argv[-1]
    if '-task1-' in input_filename or any('-task1-' in filename for filename in output_filenames):
        task = 'task1'
    elif '-task2-' in input_filename or any('-task2-' in filename for filename in output_filenames):
        task = 'task2'
    else:
        raise ValueError('Task of SERPs cannot be guessed')
    combine_serps(input_filenames, output_filename, task)

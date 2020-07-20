#!/usr/bin/env python
# -*- coding:utf-8 -*-

import csv
import logging
from statistics import median
import sys

import numpy as np

from .configuration import CSV_PARAMETERS, TOPN


LOGGER = logging.getLogger(__name__)
RUN_NAME = 'Run_Ensemble_0'


def combine_serps(input_filenames, output_filename, task):
    assert task in ('task1', 'task2')

    num_systems = len(input_filenames)

    def inverse_rank(rank):
        return float(TOPN - rank) / TOPN

    def inverse_median_rank(ranks):
        ranks = [*ranks, *[TOPN] * (num_systems - len(ranks))]
        return inverse_rank(median(ranks))

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

    percentages_tied_1 = []
    percentages_tied_2 = []
    percentages_tied_3 = []
    nums_results = []
    with open(output_filename, 'wt', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, **CSV_PARAMETERS)
        for topic_id, identifiers in separate_results.items():
            ensemble_results = dict()
            for identifier_index, (identifier, ranks) in enumerate(sorted(identifiers.items())):
                striped_rank = ranks[identifier_index % len(ranks)]
                ensemble_results[identifier] = (inverse_median_rank(ranks), len(ranks), inverse_rank(striped_rank))
            ensemble_results = sorted(ensemble_results.items(), key=lambda x: (x[1], x[0]), reverse=True)
            ensemble_results = ensemble_results[:TOPN]
            num_clusters_1 = len(set((x for _, (x, y, z) in ensemble_results)))
            num_clusters_2 = len(set(((x, y) for _, (x, y, z) in ensemble_results)))
            num_clusters_3 = len(set(((x, y, z) for _, (x, y, z) in ensemble_results)))
            num_clusters_4 = len(ensemble_results)
            percentages_tied_1.append(100.0 * (num_clusters_4 - num_clusters_1) / num_clusters_4)
            percentages_tied_2.append(100.0 * (num_clusters_4 - num_clusters_2) / num_clusters_4)
            percentages_tied_3.append(100.0 * (num_clusters_4 - num_clusters_3) / num_clusters_4)
            nums_results.append(num_clusters_4)
            for rank, (identifier, (score, *_)) in enumerate(ensemble_results):
                csv_writer.writerow((topic_id, *identifier, rank + 1, score, RUN_NAME))
    percentage_tied_1 = np.array(percentages_tied_1).T.dot(np.array(nums_results)) / np.sum(nums_results)
    percentage_tied_2 = np.array(percentages_tied_2).T.dot(np.array(nums_results)) / np.sum(nums_results)
    percentage_tied_3 = np.array(percentages_tied_3).T.dot(np.array(nums_results)) / np.sum(nums_results)
    LOGGER.info('Percentage of tied results when ordering via (M^-1): {:.2f}'.format(percentage_tied_1))
    LOGGER.info('Percentage of tied results when ordering via (M^-1, f): {:.2f}'.format(percentage_tied_2))
    LOGGER.info('Percentage of tied results when ordering via (M^-1, f, S^-1): {:.2f}'.format(percentage_tied_3))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    input_filenames = sys.argv[1:-1]
    output_filename = sys.argv[-1]
    if any('-task1-' in filename for filename in input_filenames) or '-task1-' in output_filename:
        task = 'task1'
    elif any('-task2-' in filename for filename in input_filenames) or '-task2-' in output_filename:
        task = 'task2'
    else:
        raise ValueError('Task of SERPs cannot be guessed')
    combine_serps(input_filenames, output_filename, task)

#!/usr/bin/env python
# -*- coding:utf-8 -*-

import csv
import sys

from .configuration import CSV_PARAMETERS, TOPN


def renumber_ranks(input_filename, output_filename, task):
    assert task in ('task1', 'task2')

    results = dict()
    with open(input_filename, 'rt', newline='') as csv_file:
        csv_reader = csv.reader(csv_file, **CSV_PARAMETERS)
        for row in csv_reader:
            if task == 'task1':
                topic_id, post_id, rank, score, description = row
                identifier = (post_id,)
            elif task == 'task2':
                topic_id, formula_id, post_id, rank, score, description = row
                identifier = (formula_id, post_id)
            if topic_id not in results:
                results[topic_id] = []
            rank = int(rank)
            score = float(score)
            row = (topic_id, *identifier, rank, score, description)
            results[topic_id].append(row)

    with open(output_filename, 'wt', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, **CSV_PARAMETERS)
        for topic_id, topic_results in sorted(results.items()):
            for rank, row in enumerate(sorted(topic_results, key=lambda x: (x[-3], x[-2]))):
                topic_id, *identifier, _, score, description = row
                row = (topic_id, *identifier, rank + 1, score, description)
                csv_writer.writerow(row)


if __name__ == '__main__':
    input_filename = sys.argv[1]
    output_filename = sys.argv[2]
    if '-task1-' in input_filename or '-task1-' in output_filename:
        task = 'task1'
    elif '-task2-' in input_filename or '-task2-' in output_filename:
        task = 'task2'
    else:
        raise ValueError('Task of SERPs cannot be guessed')
    renumber_ranks(input_filename, output_filename, task)

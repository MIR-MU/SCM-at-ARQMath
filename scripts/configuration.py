#!/usr/bin/env python
# -*- coding:utf-8 -*-

from itertools import repeat, chain
from multiprocessing import cpu_count

from arqmath_eval import get_topics, get_judged_documents
from input_data.scripts.configuration import ARXMLIV_NOPROBLEM_JSON_OPT_FILENAME, ARXMLIV_NOPROBLEM_JSON_SLT_FILENAME, ARXMLIV_NOPROBLEM_JSON_PREFIX_FILENAME, ARXMLIV_NOPROBLEM_JSON_INFIX_FILENAME, ARXMLIV_NOPROBLEM_JSON_LATEX_FILENAME, ARXMLIV_NOPROBLEM_HTML5_NUM_PARAGRAPHS, POOL_CHUNKSIZE, POOL_NUM_WORKERS, ARQMATH_COLLECTION_POSTS_NUM_DOCUMENTS, ARQMATH_COLLECTION_POSTS_OPT_FILENAME, ARQMATH_COLLECTION_POSTS_SLT_FILENAME, ARQMATH_COLLECTION_POSTS_PREFIX_FILENAME, ARQMATH_COLLECTION_POSTS_INFIX_FILENAME, ARQMATH_COLLECTION_POSTS_LATEX_FILENAME, CSV_PARAMETERS, ARXMLIV_WARNING1_HTML5_NUM_PARAGRAPHS, ARXMLIV_WARNING1_JSON_OPT_FILENAME, ARXMLIV_WARNING1_JSON_SLT_FILENAME, ARXMLIV_WARNING1_JSON_PREFIX_FILENAME, ARXMLIV_WARNING1_JSON_INFIX_FILENAME, ARXMLIV_WARNING1_JSON_LATEX_FILENAME, ARQMATH_TASK1_TEST_POSTS_OPT_FILENAME, ARQMATH_TASK1_TEST_POSTS_SLT_FILENAME, ARQMATH_TASK1_TEST_POSTS_PREFIX_FILENAME, ARQMATH_TASK1_TEST_POSTS_INFIX_FILENAME, ARQMATH_TASK1_TEST_POSTS_LATEX_FILENAME, ARQMATH_TASK1_TEST_POSTS_NUM_DOCUMENTS, ARQMATH_TRAIN_TSV_OPT_FILENAME, ARQMATH_TRAIN_TSV_SLT_FILENAME, ARQMATH_TRAIN_TSV_PREFIX_FILENAME, ARQMATH_TRAIN_TSV_INFIX_FILENAME, ARQMATH_TRAIN_TSV_OPT_NUM_FORMULAE, ARQMATH_TRAIN_TSV_SLT_NUM_FORMULAE, ARQMATH_TRAIN_TSV_PREFIX_NUM_FORMULAE, ARQMATH_TRAIN_TSV_INFIX_NUM_FORMULAE, ARQMATH_TASK2_TEST_TSV_OPT_FILENAME, ARQMATH_TASK2_TEST_TSV_SLT_FILENAME, ARQMATH_TASK2_TEST_TSV_PREFIX_FILENAME, ARQMATH_TASK2_TEST_TSV_INFIX_FILENAME, ARQMATH_TASK2_TEST_TSV_LATEX_FILENAME, ARQMATH_TASK2_TEST_TSV_OPT_NUM_FORMULAE, ARQMATH_TASK2_TEST_TSV_SLT_NUM_FORMULAE, ARQMATH_TASK2_TEST_TSV_PREFIX_NUM_FORMULAE, ARQMATH_TASK2_TEST_TSV_INFIX_NUM_FORMULAE, ARQMATH_TASK2_TEST_TSV_LATEX_NUM_FORMULAE, ARXMLIV_WARNING2_HTML5_NUM_PARAGRAPHS, ARXMLIV_WARNING2_JSON_OPT_FILENAME, ARXMLIV_WARNING2_JSON_SLT_FILENAME, ARXMLIV_WARNING2_JSON_PREFIX_FILENAME, ARXMLIV_WARNING2_JSON_INFIX_FILENAME, ARXMLIV_WARNING2_JSON_LATEX_FILENAME

from tqdm import tqdm


def parameters_to_string(parameters):
    return '_'.join(
        '='.join((
            str(key).replace('_', '-'),
            str(value),
        ))
        for key, value
        in sorted(parameters.items())
        if key != 'workers'
    )

DATASET_DEFAULT_PARAMETERS = {
    'phrases': 0,
}

DOC2VEC_DEFAULT_PARAMETERS = {
    'dm': 1,
    'dm_concat': 1,
    'vector_size': 400,
    'window': 4,
    'alpha': 0.05,
    'min_alpha': 0,
    'min_count': 5,
    'workers': cpu_count(),
    'epochs': 5,
    'hs': 1,
}

FASTTEXT_DEFAULT_PARAMETERS = {
    'sg': 1,
    'min_count': 5,
    'size': 300,
    'alpha': 0.05,
    'window': 5,
    'sample': 10**-4,
    'workers': cpu_count(),
    'min_alpha': 0,
    'negative': 5,
    'iter': 5,
    'min_n': 3,
    'max_n': 6,
    'bucket': '2M',
}
 
TERMSIM_MATRIX_DEFAULT_PARAMETERS = {
    'symmetric': True,
    'dominant': True,
    'nonzero_limit': 100,
}

TERMSIM_INDEX_DEFAULT_PARAMETERS = {
    'threshold': -1.0,
    'exponent': 4.0,
}

INPUT_DATA_DIRNAME = 'input_data'
OUTPUT_DATA_DIRNAME = 'output_data'

ARXMLIV_OUTPUT_DIRNAME = '{}/arxiv-dataset-arXMLiv-08-2019'.format(OUTPUT_DATA_DIRNAME)
ARXMLIV_OUTPUT_FILENAME = '{}/{{}}_{{}}'.format(ARXMLIV_OUTPUT_DIRNAME)

SERP_OUTPUT_DIRNAME = '{}/MIRMU'.format(OUTPUT_DATA_DIRNAME)
TASK1_OUTPUT_DIRNAME = '{}/Task-1-QA'.format(SERP_OUTPUT_DIRNAME)
TASK2_OUTPUT_DIRNAME = '{}/Task-2-Formulas'.format(SERP_OUTPUT_DIRNAME)

FASTTEXT_OUTPUT_FILENAME = {
    'task1': '{}/MIRMU-task1-SCM-auto-both-P.tsv'.format(TASK1_OUTPUT_DIRNAME),
    'task2': '{}/MIRMU-task2-SCM-auto-math-P.tsv'.format(TASK2_OUTPUT_DIRNAME),
}

DOC2VEC_OUTPUT_FILENAME = {
    'task1': '{}/MIRMU-task1-Formula2Vec-auto-both-P.tsv'.format(TASK1_OUTPUT_DIRNAME),
    'task2': '{}/MIRMU-task2-Formula2Vec-auto-math-P.tsv'.format(TASK2_OUTPUT_DIRNAME),
}

ARQMATH_COLLECTION_POSTS_FILENAMES = {
    'opt': '{}/{}'.format(INPUT_DATA_DIRNAME, ARQMATH_COLLECTION_POSTS_OPT_FILENAME),
    'slt': '{}/{}'.format(INPUT_DATA_DIRNAME, ARQMATH_COLLECTION_POSTS_SLT_FILENAME),
    'prefix': '{}/{}'.format(INPUT_DATA_DIRNAME, ARQMATH_COLLECTION_POSTS_PREFIX_FILENAME),
    'infix': '{}/{}'.format(INPUT_DATA_DIRNAME, ARQMATH_COLLECTION_POSTS_INFIX_FILENAME),
    'latex': '{}/{}'.format(INPUT_DATA_DIRNAME, ARQMATH_COLLECTION_POSTS_LATEX_FILENAME),
}

ARQMATH_TASK2_FORMULAE_FILENAMES = {
    'opt': '{}/{}'.format(INPUT_DATA_DIRNAME, ARQMATH_TASK2_TEST_TSV_OPT_FILENAME),
    'slt': '{}/{}'.format(INPUT_DATA_DIRNAME, ARQMATH_TASK2_TEST_TSV_SLT_FILENAME),
    'prefix': '{}/{}'.format(INPUT_DATA_DIRNAME, ARQMATH_TASK2_TEST_TSV_PREFIX_FILENAME),
    'infix': '{}/{}'.format(INPUT_DATA_DIRNAME, ARQMATH_TASK2_TEST_TSV_INFIX_FILENAME),
    'latex': '{}/{}'.format(INPUT_DATA_DIRNAME, ARQMATH_TASK2_TEST_TSV_LATEX_FILENAME),
}

ARQMATH_TASK2_FORMULAE_NUM_FORMULAE = {
    'opt': ARQMATH_TASK2_TEST_TSV_OPT_NUM_FORMULAE,
    'slt': ARQMATH_TASK2_TEST_TSV_SLT_NUM_FORMULAE,
    'prefix': ARQMATH_TASK2_TEST_TSV_PREFIX_NUM_FORMULAE,
    'infix': ARQMATH_TASK2_TEST_TSV_INFIX_NUM_FORMULAE,
    'latex': ARQMATH_TASK2_TEST_TSV_LATEX_NUM_FORMULAE,
}

ARQMATH_COLLECTION_FORMULAE_FILENAMES = {
    'opt': '{}/{}'.format(INPUT_DATA_DIRNAME, ARQMATH_TRAIN_TSV_OPT_FILENAME),
    'slt': '{}/{}'.format(INPUT_DATA_DIRNAME, ARQMATH_TRAIN_TSV_SLT_FILENAME),
    'prefix': '{}/{}'.format(INPUT_DATA_DIRNAME, ARQMATH_TRAIN_TSV_PREFIX_FILENAME),
    'infix': '{}/{}'.format(INPUT_DATA_DIRNAME, ARQMATH_TRAIN_TSV_INFIX_FILENAME),
}

ARQMATH_COLLECTION_FORMULAE_NUM_FORMULAE = {
    'opt': ARQMATH_TRAIN_TSV_OPT_NUM_FORMULAE,
    'slt': ARQMATH_TRAIN_TSV_SLT_NUM_FORMULAE,
    'prefix': ARQMATH_TRAIN_TSV_PREFIX_NUM_FORMULAE,
    'infix': ARQMATH_TRAIN_TSV_INFIX_NUM_FORMULAE,
}

ARQMATH_TASK1_POSTS_FILENAMES = {
    'opt': '{}/{}'.format(INPUT_DATA_DIRNAME, ARQMATH_TASK1_TEST_POSTS_OPT_FILENAME),
    'slt': '{}/{}'.format(INPUT_DATA_DIRNAME, ARQMATH_TASK1_TEST_POSTS_SLT_FILENAME),
    'prefix': '{}/{}'.format(INPUT_DATA_DIRNAME, ARQMATH_TASK1_TEST_POSTS_PREFIX_FILENAME),
    'infix': '{}/{}'.format(INPUT_DATA_DIRNAME, ARQMATH_TASK1_TEST_POSTS_INFIX_FILENAME),
    'latex': '{}/{}'.format(INPUT_DATA_DIRNAME, ARQMATH_TASK1_TEST_POSTS_LATEX_FILENAME),
}

DATASET_NUMS_PARAGRAPHS = {
    'no_problem': ARXMLIV_NOPROBLEM_HTML5_NUM_PARAGRAPHS,
    'warning_1': ARXMLIV_WARNING1_HTML5_NUM_PARAGRAPHS,
    'warning_2': ARXMLIV_WARNING2_HTML5_NUM_PARAGRAPHS,
    'arqmath': ARQMATH_COLLECTION_POSTS_NUM_DOCUMENTS,
}

DATASET_JSON_FILENAMES = {
    'no_problem': {
        'opt': '{}/{}'.format(INPUT_DATA_DIRNAME, ARXMLIV_NOPROBLEM_JSON_OPT_FILENAME),
        'slt': '{}/{}'.format(INPUT_DATA_DIRNAME, ARXMLIV_NOPROBLEM_JSON_SLT_FILENAME),
        'prefix': '{}/{}'.format(INPUT_DATA_DIRNAME, ARXMLIV_NOPROBLEM_JSON_PREFIX_FILENAME),
        'infix': '{}/{}'.format(INPUT_DATA_DIRNAME, ARXMLIV_NOPROBLEM_JSON_INFIX_FILENAME),
        'latex': '{}/{}'.format(INPUT_DATA_DIRNAME, ARXMLIV_NOPROBLEM_JSON_LATEX_FILENAME),
    },
    'warning_1': {
        'opt': '{}/{}'.format(INPUT_DATA_DIRNAME, ARXMLIV_WARNING1_JSON_OPT_FILENAME),
        'slt': '{}/{}'.format(INPUT_DATA_DIRNAME, ARXMLIV_WARNING1_JSON_SLT_FILENAME),
        'prefix': '{}/{}'.format(INPUT_DATA_DIRNAME, ARXMLIV_WARNING1_JSON_PREFIX_FILENAME),
        'infix': '{}/{}'.format(INPUT_DATA_DIRNAME, ARXMLIV_WARNING1_JSON_INFIX_FILENAME),
        'latex': '{}/{}'.format(INPUT_DATA_DIRNAME, ARXMLIV_WARNING1_JSON_LATEX_FILENAME),
    },
    'warning_2': {
        'opt': '{}/{}'.format(INPUT_DATA_DIRNAME, ARXMLIV_WARNING2_JSON_OPT_FILENAME),
        'slt': '{}/{}'.format(INPUT_DATA_DIRNAME, ARXMLIV_WARNING2_JSON_SLT_FILENAME),
        'prefix': '{}/{}'.format(INPUT_DATA_DIRNAME, ARXMLIV_WARNING2_JSON_PREFIX_FILENAME),
        'infix': '{}/{}'.format(INPUT_DATA_DIRNAME, ARXMLIV_WARNING2_JSON_INFIX_FILENAME),
        'latex': '{}/{}'.format(INPUT_DATA_DIRNAME, ARXMLIV_WARNING2_JSON_LATEX_FILENAME),
    },
    'arqmath': {
        'opt': ARQMATH_COLLECTION_POSTS_FILENAMES['opt'],
        'slt': ARQMATH_COLLECTION_POSTS_FILENAMES['slt'],
        'prefix': ARQMATH_COLLECTION_POSTS_FILENAMES['prefix'],
        'infix': ARQMATH_COLLECTION_POSTS_FILENAMES['infix'],
        'latex': ARQMATH_COLLECTION_POSTS_FILENAMES['latex'],
    },
}

RESULT_TYPES = ('judged', 'task1', 'task2')
TOPN=1000

TASK = {
    'judged': 'task1-votes',
    'task1': 'task1-votes.V1.2',
    'task2': 'task2-topics-formula_ids.V.1.1',
}

SUBSET = {
    'judged': 'validation',
    'task1': 'all',
    'task2': 'all',
}


DOC2VEC_CONFIGURATIONS = {
    'judged': [
        ('prefix', 'no_problem', {'phrases': 2}, {}),  # NDCG' 0.7579
        ('prefix', 'no_problem', {'phrases': 2}, {'dm': 0, 'vector_size': 300, 'negative': 12, 'hs': 0, 'alpha': 0.1, 'window': 8}),  # NDCG' 0.7604 (wins)
        # ('prefix', 'no_problem', {}, {}),
        # ('prefix', 'no_problem', {}, {'vector_size': 300}),
        # ('prefix', 'no_problem', {}, {'window': 5, 'vector_size': 300}),
        # ('prefix', 'no_problem', {}, {'window': 6, 'vector_size': 300}),
    ],
    'task1': [
        # ('prefix', 'no_problem', {'phrases': 2}, {'dm': 0, 'vector_size': 300, 'negative': 12, 'hs': 0, 'alpha': 0.1, 'window': 8}),
        # ('prefix', ['no_problem', 'arqmath'], {'phrases': 2}, {'dm': 0, 'vector_size': 300, 'negative': 12, 'hs': 0, 'alpha': 0.1, 'window': 8, 'epochs': 10}),
        # ('prefix', ['no_problem', 'warning_1', 'arqmath'], {'phrases': 2}, {'dm': 0, 'vector_size': 300, 'negative': 12, 'hs': 0, 'alpha': 0.1, 'window': 8, 'epochs': 2}),
        ('prefix', ['no_problem', 'warning_1', 'warning_2', 'arqmath'], {'phrases': 2}, {'dm': 0, 'vector_size': 300, 'negative': 12, 'hs': 0, 'alpha': 0.1, 'window': 8, 'epochs': 10}),
    ],
    'task2': [
        # ('prefix', 'no_problem', {'phrases': 2}, {'dm': 0, 'vector_size': 300, 'negative': 12, 'hs': 0, 'alpha': 0.1, 'window': 8}),
        # ('prefix', ['no_problem', 'arqmath'], {'phrases': 2}, {'dm': 0, 'vector_size': 300, 'negative': 12, 'hs': 0, 'alpha': 0.1, 'window': 8, 'epochs': 10}),
        # ('prefix', ['no_problem', 'warning_1', 'arqmath'], {'phrases': 2}, {'dm': 0, 'vector_size': 300, 'negative': 12, 'hs': 0, 'alpha': 0.1, 'window': 8, 'epochs': 2}),
        ('prefix', ['no_problem', 'warning_1', 'warning_2', 'arqmath'], {'phrases': 2}, {'dm': 0, 'vector_size': 300, 'negative': 12, 'hs': 0, 'alpha': 0.1, 'window': 8, 'epochs': 10}),
    ],
}

FASTTEXT_CONFIGURATIONS = {
    'judged': [
        # Math representation selection
        ('opt', 'no_problem', {}, {}, {}, {}),     # NDCG' 0.7606
        ('slt', 'no_problem', {}, {}, {}, {}),     # NDCG' 0.7607
        ('prefix', 'no_problem', {}, {}, {}, {}),  # NDCG' 0.7612 (wins)
        ('infix', 'no_problem', {}, {}, {}, {}),   # NDCG' 0.7613 (wins)
        ('latex', 'no_problem', {}, {}, {}, {}),   # NDCG' 0.7602
        ('nomath', 'no_problem', {}, {}, {}, {}),  # NDCG' 0.7600
      
        # Math token concatenation
        # ('infix', 'no_problem', {'concat_math': True}, {}, {}, {}),  # NDCG' 0.7603
      
        # N-gram collocation modeling
        # ('infix', 'no_problem', {'phrases': 0}, {}, {}, {}),  # NDCG' 0.7612
        # ('infix', 'no_problem', {'phrases': 1}, {}, {}, {}),  # NDCG' 0.7614 (wins)
        # ('infix', 'no_problem', {'phrases': 5}, {}, {}, {}),  # NDCG' 0.7612
        ('prefix', 'no_problem', {'phrases': 0}, {}, {}, {}),   # NDCG' 0.7612
        ('prefix', 'no_problem', {'phrases': 1}, {}, {}, {}),   # NDCG' 0.7613
        ('prefix', 'no_problem', {'phrases': 2}, {}, {}, {}),   # NDCG' 0.7614, Mikolov et al. (2013, Distributed Representations of Words and Phrases and their Compositionality) suggests 2-4 iterations (wins)
        ('prefix', 'no_problem', {'phrases': 3}, {}, {}, {}),   # NDCG' 0.7612, Mikolov et al. (2013, Distributed Representations of Words and Phrases and their Compositionality) suggests 2-4 iterations
        ('prefix', 'no_problem', {'phrases': 4}, {}, {}, {}),   # NDCG' 0.7610, Mikolov et al. (2013, Distributed Representations of Words and Phrases and their Compositionality) suggests 2-4 iterations
        ('prefix', 'no_problem', {'phrases': 5}, {}, {}, {}),   # NDCG' 0.7613, Mikolov et al. (2017, Advances in Pre-Training Distributed Word Representations) suggests 5 or 6 iterations
        ('prefix', 'no_problem', {'phrases': 6}, {}, {}, {}),   # NDCG' 0.7613, Mikolov et al. (2017, Advances in Pre-Training Distributed Word Representations) suggests 5 or 6 iterations
        ('prefix', 'no_problem', {'phrases': 10}, {}, {}, {}),  # NDCG' 0.7612
      
        # N-gram collocation modeling without math
        # ('nomath', 'no_problem', {'phrases': 0}, {}, {}, {}),  # NDCG' 0.7600 (wins)
        # ('nomath', 'no_problem', {'phrases': 1}, {}, {}, {}),  # NDCG' 0.7598
        # ('nomath', 'no_problem', {'phrases': 2}, {}, {}, {}),  # NDCG' 0.7596
      
        # Hash bucket size
        ('prefix', 'no_problem', {'phrases': 2}, {'bucket': '1M'}, {}, {}),  # NDCG' 0.7612
        ('prefix', 'no_problem', {'phrases': 2}, {'bucket': '2M'}, {}, {}),  # NDCG' 0.7614 (wins)
        ('prefix', 'no_problem', {'phrases': 2}, {'bucket': '4M'}, {}, {}),  # NDCG' 0.7612
        ('prefix', 'no_problem', {'phrases': 2}, {'bucket': '8M'}, {}, {}),  # NDCG' 0.7611
      
        # Term similarity matrix non-zero limit
        ('prefix', 'no_problem', {'phrases': 2}, {}, {'nonzero_limit': 0}, {}),     # NDCG' 0.7613
      
        ('prefix', 'no_problem', {'phrases': 2}, {}, {'nonzero_limit': 50}, {}),    # NDCG' 0.7614
        ('prefix', 'no_problem', {'phrases': 2}, {}, {'nonzero_limit': 100}, {}),   # NDCG' 0.7614 (wins)
        ('prefix', 'no_problem', {'phrases': 2}, {}, {'nonzero_limit': 200}, {}),   # NDCG' 0.7614
        ('prefix', 'no_problem', {'phrases': 2}, {}, {'nonzero_limit': 400}, {}),   # NDCG' 0.7612
        ('prefix', 'no_problem', {'phrases': 2}, {}, {'nonzero_limit': 800}, {}),   # NDCG' 0.7614
        ('prefix', 'no_problem', {'phrases': 2}, {}, {'nonzero_limit': 1600}, {}),  # NDCG' 0.7612
      
        ('prefix', 'no_problem', {'phrases': 2}, {}, {'symmetric': False, 'nonzero_limit': 50}, {}),    # NDCG' 0.7613
        ('prefix', 'no_problem', {'phrases': 2}, {}, {'symmetric': False, 'nonzero_limit': 100}, {}),   # NDCG' 0.7613
        ('prefix', 'no_problem', {'phrases': 2}, {}, {'symmetric': False, 'nonzero_limit': 200}, {}),   # NDCG' 0.7613
        ('prefix', 'no_problem', {'phrases': 2}, {}, {'symmetric': False, 'nonzero_limit': 400}, {}),   # NDCG' 0.7613
        ('prefix', 'no_problem', {'phrases': 2}, {}, {'symmetric': False, 'nonzero_limit': 800}, {}),   # NDCG' 0.7613
        ('prefix', 'no_problem', {'phrases': 2}, {}, {'symmetric': False, 'nonzero_limit': 1600}, {}),  # NDCG' 0.7613
      
        ('prefix', 'no_problem', {'phrases': 2}, {}, {'dominant': False, 'nonzero_limit': 50}, {}),   # NDCG' 0.7613
        ('prefix', 'no_problem', {'phrases': 2}, {}, {'dominant': False, 'nonzero_limit': 100}, {}),  # NDCG' 0.7610
        ('prefix', 'no_problem', {'phrases': 2}, {}, {'dominant': False, 'nonzero_limit': 200}, {}),  # NDCG' 0.7611
        ('prefix', 'no_problem', {'phrases': 2}, {}, {'dominant': False, 'nonzero_limit': 400}, {}),  # NDCG' 0.7613
        ('prefix', 'no_problem', {'phrases': 2}, {}, {'dominant': False, 'nonzero_limit': 800}, {}),  # NDCG' 0.7609
      
        ('prefix', 'no_problem', {'phrases': 2}, {}, {'symmetric': False, 'dominant': False, 'nonzero_limit': 50}, {}),   # NDCG' 0.7612
        ('prefix', 'no_problem', {'phrases': 2}, {}, {'symmetric': False, 'dominant': False, 'nonzero_limit': 100}, {}),  # NDCG' 0.7610
        ('prefix', 'no_problem', {'phrases': 2}, {}, {'symmetric': False, 'dominant': False, 'nonzero_limit': 200}, {}),  # NDCG' 0.7610
    ],
    'task1': [
        # ('prefix', 'no_problem', {'phrases': 2}, {}, {'nonzero_limit': 100}, {}),
        # ('prefix', ['no_problem', 'arqmath'], {'phrases': 2}, {'iter': 10, 'negative': 10, 'min_n': 4, 'max_n': 5, 'sg': 0}, {'nonzero_limit': 100}, {}),
        # ('prefix', ['no_problem', 'warning_1', 'arqmath'], {'phrases': 2}, {'iter': 2, 'negative': 10, 'min_n': 4, 'max_n': 5, 'sg': 0}, {'nonzero_limit': 100}, {}),
        ('prefix', ['no_problem', 'warning_1', 'warning_2', 'arqmath'], {'phrases': 2}, {'iter': 10, 'negative': 10, 'min_n': 4, 'max_n': 5, 'sg': 0}, {'nonzero_limit': 100}, {}),
    ],
    'task2': [
        # ('prefix', 'no_problem', {'phrases': 2}, {}, {'nonzero_limit': 100}, {}),
        # ('prefix', ['no_problem', 'arqmath'], {'phrases': 2}, {'iter': 10, 'negative': 10, 'min_n': 4, 'max_n': 5, 'sg': 0}, {'nonzero_limit': 100}, {}),
        # ('prefix', ['no_problem', 'warning_1', 'arqmath'], {'phrases': 2}, {'iter': 2, 'negative': 10, 'min_n': 4, 'max_n': 5, 'sg': 0}, {'nonzero_limit': 100}, {}),
        ('prefix', ['no_problem', 'warning_1', 'warning_2', 'arqmath'], {'phrases': 2}, {'iter': 10, 'negative': 10, 'min_n': 4, 'max_n': 5, 'sg': 0}, {'nonzero_limit': 100}, {}),
    ],
}


def get_common_parameters(result_type, math_representation, datasets, dataset_parameters):
    discard_math = math_representation == 'nomath'
    judged_results = result_type == 'judged'

    if result_type == 'judged':
        topic_corpus_filename = ARQMATH_COLLECTION_POSTS_FILENAMES[math_representation if not discard_math else 'latex']
        topic_corpus_num_documents = ARQMATH_COLLECTION_POSTS_NUM_DOCUMENTS
        topic_ids = get_topics(task=TASK[result_type], subset=SUBSET[result_type])
        document_corpus_filename = topic_corpus_filename
        document_corpus_num_documents = topic_corpus_num_documents
        document_ids = get_judged_documents(task=TASK[result_type], subset=SUBSET[result_type])
        topic_judgements = {
            topic_id: get_judged_documents(task=TASK[result_type], subset=SUBSET[result_type], topic=topic_id)
            for topic_id in topic_ids
        }
    else:
        topic_judgements = None
        if result_type == 'task1':
            topic_corpus_filename = ARQMATH_TASK1_POSTS_FILENAMES[math_representation if not discard_math else 'latex']
            topic_corpus_num_documents = ARQMATH_TASK1_TEST_POSTS_NUM_DOCUMENTS
            topic_ids = [
                'A.{}'.format(query_number + 1)
                for query_number in range(100)
                if (query_number + 1) not in (31, 78)
            ]
            document_corpus_filename = ARQMATH_COLLECTION_POSTS_FILENAMES[math_representation if not discard_math else 'latex']
            document_corpus_num_documents = ARQMATH_COLLECTION_POSTS_NUM_DOCUMENTS
            document_ids = get_judged_documents(task=TASK[result_type], subset=SUBSET[result_type])
        elif result_type == 'task2':
            assert not discard_math
            topic_corpus_filename = ARQMATH_TASK2_FORMULAE_FILENAMES[math_representation]
            topic_corpus_num_documents = ARQMATH_TASK2_FORMULAE_NUM_FORMULAE[math_representation]
            topic_ids = set(
                (
                    formula_id,
                    *get_judged_documents(
                        task=TASK[result_type],
                        subset=SUBSET[result_type],
                        topic=formula_id,
                    ),
                )
                for formula_id in get_topics(task=TASK[result_type], subset=SUBSET[result_type])
            )
            document_corpus_filename = ARQMATH_COLLECTION_FORMULAE_FILENAMES[math_representation]
            document_corpus_num_documents = ARQMATH_COLLECTION_FORMULAE_NUM_FORMULAE[math_representation]
            document_ids = None

    if isinstance(datasets, str):
        datasets = [datasets]
    json_filenames = [
        DATASET_JSON_FILENAMES[dataset][math_representation if not discard_math else 'latex']
        for dataset in datasets
    ]
    json_nums_paragraphs = [
        DATASET_NUMS_PARAGRAPHS[dataset]
        for dataset in datasets
    ]

    dataset_parameters = {**DATASET_DEFAULT_PARAMETERS, **dataset_parameters}
    dataset_formattable_parameter_string = parameters_to_string({**dataset_parameters, **{'phrases': '{}'}})
    phraser_filename = ARXMLIV_OUTPUT_FILENAME.format(math_representation, '{}.phraser'.format(dataset_formattable_parameter_string))

    return {
        'judged_results': judged_results,
        'topic_judgements': topic_judgements,
        'json_filenames': json_filenames,
        'json_nums_paragraphs': json_nums_paragraphs,
        'dataset_parameters': dataset_parameters,
        'phraser_filename': phraser_filename,
        'topic_ids': topic_ids,
        'topic_corpus_filename': topic_corpus_filename,
        'topic_corpus_num_documents': topic_corpus_num_documents,
        'document_ids': document_ids,
        'document_corpus_filename': document_corpus_filename,
        'document_corpus_num_documents': document_corpus_num_documents,
        'discard_math': discard_math,
        'result_type': result_type,
    }


def get_doc2vec_configurations():
    doc2vec_configurations = chain(*[
        zip(repeat(result_type), DOC2VEC_CONFIGURATIONS[result_type])
        for result_type in RESULT_TYPES
    ])
    doc2vec_configurations_len = sum(
        len(DOC2VEC_CONFIGURATIONS[result_type])
        for result_type in RESULT_TYPES
    )
    doc2vec_configurations = tqdm(doc2vec_configurations, total=doc2vec_configurations_len, desc='Processing doc2vec configurations')
    for result_type, (math_representation, datasets, dataset_parameters, doc2vec_parameters) in doc2vec_configurations:
        judged_results = result_type == 'judged'
        common_parameters = get_common_parameters(result_type, math_representation, datasets, dataset_parameters)

        dataset_parameter_string = parameters_to_string(common_parameters['dataset_parameters'])

        doc2vec_parameters = {**DOC2VEC_DEFAULT_PARAMETERS, **doc2vec_parameters}
        doc2vec_parameter_string = '_'.join((dataset_parameter_string, parameters_to_string(doc2vec_parameters)))
        doc2vec_filename = ARXMLIV_OUTPUT_FILENAME.format(math_representation, doc2vec_parameter_string)

        if judged_results:
            validation_result_filename = ARXMLIV_OUTPUT_FILENAME.format(math_representation, '{}.tsv'.format(doc2vec_parameter_string))
        else:
            validation_result_filename = DOC2VEC_OUTPUT_FILENAME[result_type]

        yield {
            **common_parameters,
            'doc2vec_parameters': doc2vec_parameters,
            'doc2vec_filename': doc2vec_filename,
            'validation_result_filename': validation_result_filename,
        }


def get_fasttext_configurations():
    fasttext_configurations = chain(*[
        zip(repeat(result_type), FASTTEXT_CONFIGURATIONS[result_type])
        for result_type in RESULT_TYPES
    ])
    fasttext_configurations_len = sum(
        len(FASTTEXT_CONFIGURATIONS[result_type])
        for result_type in RESULT_TYPES
    )
    fasttext_configurations = tqdm(fasttext_configurations, total=fasttext_configurations_len, desc='Processing fastText configurations')
    for result_type, (math_representation, datasets, dataset_parameters, fasttext_parameters, termsim_matrix_parameters, termsim_index_parameters) in fasttext_configurations:
        judged_results = result_type == 'judged'
        common_parameters = get_common_parameters(result_type, math_representation, datasets, dataset_parameters)

        dataset_parameter_string = parameters_to_string(common_parameters['dataset_parameters'])
        dictionary_filename = ARXMLIV_OUTPUT_FILENAME.format(math_representation, '{}.dictionary'.format(dataset_parameter_string))
        topic_tfidf_filename = ARXMLIV_OUTPUT_FILENAME.format(math_representation, '{}.Ttfidf'.format(dataset_parameter_string))
        document_tfidf_filename = ARXMLIV_OUTPUT_FILENAME.format(math_representation, '{}.Dtfidf'.format(dataset_parameter_string))

        fasttext_parameters = {**FASTTEXT_DEFAULT_PARAMETERS, **fasttext_parameters}
        fasttext_parameter_string = '_'.join((dataset_parameter_string, parameters_to_string(fasttext_parameters)))
        if 'bucket' in fasttext_parameters:
            bucket = fasttext_parameters['bucket']
            fasttext_parameters['bucket'] = int(bucket[:-1]) * 10**6
        fasttext_filename = ARXMLIV_OUTPUT_FILENAME.format(math_representation, fasttext_parameter_string)

        termsim_matrix_parameters = {**TERMSIM_MATRIX_DEFAULT_PARAMETERS, **termsim_matrix_parameters}
        termsim_matrix_parameter_string = parameters_to_string(termsim_matrix_parameters)
        termsim_index_parameters = {**TERMSIM_INDEX_DEFAULT_PARAMETERS, **termsim_index_parameters}
        termsim_index_parameter_string = parameters_to_string(termsim_index_parameters)
        scm_parameter_string = '_'.join((fasttext_parameter_string, termsim_matrix_parameter_string, termsim_index_parameter_string))
        scm_filename = ARXMLIV_OUTPUT_FILENAME.format(math_representation, scm_parameter_string)

        if judged_results:
            validation_result_filename = ARXMLIV_OUTPUT_FILENAME.format(math_representation, '{}.tsv'.format(scm_parameter_string))
        else:
            validation_result_filename = FASTTEXT_OUTPUT_FILENAME[result_type]

        yield {
            **common_parameters,
            'fasttext_parameters': fasttext_parameters,
            'fasttext_filename': fasttext_filename,
            'termsim_matrix_parameters': termsim_matrix_parameters,
            'termsim_index_parameters': termsim_index_parameters,
            'scm_filename': scm_filename,
            'dictionary_filename': dictionary_filename,
            'topic_tfidf_filename': topic_tfidf_filename,
            'document_tfidf_filename': document_tfidf_filename,
            'validation_result_filename': validation_result_filename,
        }

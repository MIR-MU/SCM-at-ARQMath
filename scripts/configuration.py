#!/usr/bin/env python
# -*- coding:utf-8 -*-

from multiprocessing import cpu_count

from input_data.scripts.configuration import ARXMLIV_NOPROBLEM_JSON_OPT_FILENAME, ARXMLIV_NOPROBLEM_JSON_SLT_FILENAME, ARXMLIV_NOPROBLEM_JSON_PREFIX_FILENAME, ARXMLIV_NOPROBLEM_JSON_INFIX_FILENAME, ARXMLIV_NOPROBLEM_JSON_LATEX_FILENAME, ARXMLIV_NOPROBLEM_HTML5_NUM_PARAGRAPHS, POOL_CHUNKSIZE, POOL_NUM_WORKERS, ARQMATH_COLLECTION_POSTS_NUM_DOCUMENTS, ARQMATH_COLLECTION_POSTS_OPT_FILENAME, ARQMATH_COLLECTION_POSTS_SLT_FILENAME, ARQMATH_COLLECTION_POSTS_PREFIX_FILENAME, ARQMATH_COLLECTION_POSTS_INFIX_FILENAME, ARQMATH_COLLECTION_POSTS_LATEX_FILENAME

from tqdm import tqdm


def parameters_to_string(parameters):
    return '_'.join(
        '='.join((str(key).replace('_', '-'), str(value)))
        for key, value in sorted(parameters.items())
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
    'bucket': 2 * 10**6,
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

ARQMATH_COLLECTION_POSTS_FILENAMES = {
    'opt': '{}/{}'.format(INPUT_DATA_DIRNAME, ARQMATH_COLLECTION_POSTS_OPT_FILENAME),
    'slt': '{}/{}'.format(INPUT_DATA_DIRNAME, ARQMATH_COLLECTION_POSTS_SLT_FILENAME),
    'prefix': '{}/{}'.format(INPUT_DATA_DIRNAME, ARQMATH_COLLECTION_POSTS_PREFIX_FILENAME),
    'infix': '{}/{}'.format(INPUT_DATA_DIRNAME, ARQMATH_COLLECTION_POSTS_INFIX_FILENAME),
    'latex': '{}/{}'.format(INPUT_DATA_DIRNAME, ARQMATH_COLLECTION_POSTS_LATEX_FILENAME),
}

ARXMLIV_NUMS_PARAGRAPHS = {
    'no_problem': ARXMLIV_NOPROBLEM_HTML5_NUM_PARAGRAPHS,
}

ARXMLIV_JSON_FILENAMES = {
    'no_problem': {
        'opt': '{}/{}'.format(INPUT_DATA_DIRNAME, ARXMLIV_NOPROBLEM_JSON_OPT_FILENAME),
        'slt': '{}/{}'.format(INPUT_DATA_DIRNAME, ARXMLIV_NOPROBLEM_JSON_SLT_FILENAME),
        'prefix': '{}/{}'.format(INPUT_DATA_DIRNAME, ARXMLIV_NOPROBLEM_JSON_PREFIX_FILENAME),
        'infix': '{}/{}'.format(INPUT_DATA_DIRNAME, ARXMLIV_NOPROBLEM_JSON_INFIX_FILENAME),
        'latex': '{}/{}'.format(INPUT_DATA_DIRNAME, ARXMLIV_NOPROBLEM_JSON_LATEX_FILENAME),
    },
}

DOC2VEC_CONFIGURATIONS = [
  ('prefix', 'no_problem', {'phrases': 2}, {})
]

FASTTEXT_CONFIGURATIONS = [
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
  ('prefix', 'no_problem', {'phrases': 2}, {'bucket': 1 * 10**6}, {}, {}),  # NDCG' 0.7612
  ('prefix', 'no_problem', {'phrases': 2}, {'bucket': 2 * 10**6}, {}, {}),  # NDCG' 0.7614 (wins)
  ('prefix', 'no_problem', {'phrases': 2}, {'bucket': 4 * 10**6}, {}, {}),  # NDCG' 0.7612
  ('prefix', 'no_problem', {'phrases': 2}, {'bucket': 8 * 10**6}, {}, {}),  # NDCG' 0.7611

  # Term similarity matrix non-zero limit
  ('prefix', 'no_problem', {'phrases': 2}, {}, {'nonzero_limit': 0}, {}),     # NDCG' 0.7613

  ('prefix', 'no_problem', {'phrases': 2}, {}, {'nonzero_limit': 50}, {}),    # NDCG' 0.7614
  ('prefix', 'no_problem', {'phrases': 2}, {}, {'nonzero_limit': 100}, {}),   # NDCG' 0.7614 (wins)
  ('prefix', 'no_problem', {'phrases': 2}, {}, {'nonzero_limit': 200}, {}),   # NDCG' 0.7614
  ('prefix', 'no_problem', {'phrases': 2}, {}, {'nonzero_limit': 400}, {}),   # NDCG' 0.7612
  ('prefix', 'no_problem', {'phrases': 2}, {}, {'nonzero_limit': 800}, {}),   # NDCG' 0.7614
  ('prefix', 'no_problem', {'phrases': 2}, {}, {'nonzero_limit': 1600}, {}),  # NDCG' 0.7612

# ('prefix', 'no_problem', {'phrases': 2}, {}, {'symmetric': False, 'nonzero_limit': 50}, {}),
# ('prefix', 'no_problem', {'phrases': 2}, {}, {'symmetric': False, 'nonzero_limit': 100}, {}),
# ('prefix', 'no_problem', {'phrases': 2}, {}, {'symmetric': False, 'nonzero_limit': 200}, {}),
# ('prefix', 'no_problem', {'phrases': 2}, {}, {'symmetric': False, 'nonzero_limit': 400}, {}),
# ('prefix', 'no_problem', {'phrases': 2}, {}, {'symmetric': False, 'nonzero_limit': 800}, {}),
# ('prefix', 'no_problem', {'phrases': 2}, {}, {'symmetric': False, 'nonzero_limit': 1600}, {}),

# ('prefix', 'no_problem', {'phrases': 2}, {}, {'dominant': False, 'nonzero_limit': 50}, {}),
# ('prefix', 'no_problem', {'phrases': 2}, {}, {'dominant': False, 'nonzero_limit': 100}, {}),
# ('prefix', 'no_problem', {'phrases': 2}, {}, {'dominant': False, 'nonzero_limit': 200}, {}),
# ('prefix', 'no_problem', {'phrases': 2}, {}, {'dominant': False, 'nonzero_limit': 400}, {}),
# ('prefix', 'no_problem', {'phrases': 2}, {}, {'dominant': False, 'nonzero_limit': 800}, {}),
# ('prefix', 'no_problem', {'phrases': 2}, {}, {'dominant': False, 'nonzero_limit': 1600}, {}),

# ('prefix', 'no_problem', {'phrases': 2}, {}, {'symmetric': False, 'dominant': False, 'nonzero_limit': 50}, {}),
# ('prefix', 'no_problem', {'phrases': 2}, {}, {'symmetric': False, 'dominant': False, 'nonzero_limit': 100}, {}),
# ('prefix', 'no_problem', {'phrases': 2}, {}, {'symmetric': False, 'dominant': False, 'nonzero_limit': 200}, {}),
# ('prefix', 'no_problem', {'phrases': 2}, {}, {'symmetric': False, 'dominant': False, 'nonzero_limit': 400}, {}),
# ('prefix', 'no_problem', {'phrases': 2}, {}, {'symmetric': False, 'dominant': False, 'nonzero_limit': 800}, {}),
# ('prefix', 'no_problem', {'phrases': 2}, {}, {'symmetric': False, 'dominant': False, 'nonzero_limit': 1600}, {}),
]


def get_common_parameters(math_representation, arxmliv, dataset_parameters):
    discard_math = math_representation == 'nomath'

    corpus_filename = ARQMATH_COLLECTION_POSTS_FILENAMES[math_representation if not discard_math else 'latex']
    corpus_num_documents = ARQMATH_COLLECTION_POSTS_NUM_DOCUMENTS

    json_filename = ARXMLIV_JSON_FILENAMES[arxmliv][math_representation if not discard_math else 'latex']
    json_num_paragraphs = ARXMLIV_NUMS_PARAGRAPHS[arxmliv]

    dataset_parameters = {**DATASET_DEFAULT_PARAMETERS, **dataset_parameters}
    dataset_formattable_parameter_string = parameters_to_string({**dataset_parameters, **{'phrases': '{}'}})
    phraser_filename = ARXMLIV_OUTPUT_FILENAME.format(math_representation, '{}.phraser'.format(dataset_formattable_parameter_string))

    return {
        'json_filename': json_filename,
        'json_num_paragraphs': json_num_paragraphs,
        'dataset_parameters': dataset_parameters,
        'phraser_filename': phraser_filename,
        'corpus_filename': corpus_filename,
        'corpus_num_documents': corpus_num_documents,
        'discard_math': discard_math,
    }


def get_doc2vec_configurations():
    for math_representation, arxmliv, dataset_parameters, doc2vec_parameters in tqdm(DOC2VEC_CONFIGURATIONS, desc='Processing doc2vec configurations'):
        common_parameters = get_common_parameters(math_representation, arxmliv, dataset_parameters)

        dataset_parameter_string = parameters_to_string(common_parameters['dataset_parameters'])

        doc2vec_parameters = {**DOC2VEC_DEFAULT_PARAMETERS, **doc2vec_parameters}
        doc2vec_parameter_string = '_'.join((dataset_parameter_string, parameters_to_string(doc2vec_parameters)))
        doc2vec_filename = ARXMLIV_OUTPUT_FILENAME.format(math_representation, doc2vec_parameter_string)

        validation_result_filename = ARXMLIV_OUTPUT_FILENAME.format(math_representation, '{}.tsv'.format(doc2vec_parameter_string))

        yield {
            **common_parameters,
            'doc2vec_parameters': doc2vec_parameters,
            'doc2vec_filename': doc2vec_filename,
            'validation_result_filename': validation_result_filename,
        }


def get_fasttext_configurations():
    for math_representation, arxmliv, dataset_parameters, fasttext_parameters, termsim_matrix_parameters, termsim_index_parameters in tqdm(FASTTEXT_CONFIGURATIONS, desc='Processing fastText configurations'):
        common_parameters = get_common_parameters(math_representation, arxmliv, dataset_parameters)

        dataset_parameter_string = parameters_to_string(common_parameters['dataset_parameters'])
        dictionary_filename = ARXMLIV_OUTPUT_FILENAME.format(math_representation, '{}.dictionary'.format(dataset_parameter_string))
        tfidf_filename = ARXMLIV_OUTPUT_FILENAME.format(math_representation, '{}.tfidf'.format(dataset_parameter_string))

        fasttext_parameters = {**FASTTEXT_DEFAULT_PARAMETERS, **fasttext_parameters}
        fasttext_parameter_string = '_'.join((dataset_parameter_string, parameters_to_string(fasttext_parameters)))
        fasttext_filename = ARXMLIV_OUTPUT_FILENAME.format(math_representation, fasttext_parameter_string)

        termsim_matrix_parameters = {**TERMSIM_MATRIX_DEFAULT_PARAMETERS, **termsim_matrix_parameters}
        termsim_matrix_parameter_string = parameters_to_string(termsim_matrix_parameters)
        termsim_index_parameters = {**TERMSIM_INDEX_DEFAULT_PARAMETERS, **termsim_index_parameters}
        termsim_index_parameter_string = parameters_to_string(termsim_index_parameters)
        scm_parameter_string = '_'.join((fasttext_parameter_string, termsim_matrix_parameter_string, termsim_index_parameter_string))
        scm_filename = ARXMLIV_OUTPUT_FILENAME.format(math_representation, scm_parameter_string)
        validation_result_filename = ARXMLIV_OUTPUT_FILENAME.format(math_representation, '{}.tsv'.format(scm_parameter_string))

        yield {
            **common_parameters,
            'fasttext_parameters': fasttext_parameters,
            'fasttext_filename': fasttext_filename,
            'termsim_matrix_parameters': termsim_matrix_parameters,
            'termsim_index_parameters': termsim_index_parameters,
            'scm_filename': scm_filename,
            'dictionary_filename': dictionary_filename,
            'tfidf_filename': tfidf_filename,
            'validation_result_filename': validation_result_filename,
        }

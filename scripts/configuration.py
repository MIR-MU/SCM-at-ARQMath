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

CONFIGURATIONS = [
  # Math representation selection (infix)
  ('opt', 'no_problem', {}, {}, {}, {}),
  ('slt', 'no_problem', {}, {}, {}, {}),
  ('prefix', 'no_problem', {}, {}, {}, {}),
  ('infix', 'no_problem', {}, {}, {}, {}),
  ('latex', 'no_problem', {}, {}, {}, {}),
  ('nomath', 'no_problem', {}, {}, {}, {}),
]


def get_configurations():
    for math_representation, arxmliv, dataset_parameters, fasttext_parameters, termsim_matrix_parameters, termsim_index_parameters in tqdm(CONFIGURATIONS, desc='Processing configurations'):
        discard_math = math_representation == 'nomath'

        corpus_filename = ARQMATH_COLLECTION_POSTS_FILENAMES[math_representation if not discard_math else 'latex']
        corpus_num_documents = ARQMATH_COLLECTION_POSTS_NUM_DOCUMENTS

        json_filename = ARXMLIV_JSON_FILENAMES[arxmliv][math_representation if not discard_math else 'latex']
        json_num_paragraphs = ARXMLIV_NUMS_PARAGRAPHS[arxmliv]

        dataset_parameters = {**DATASET_DEFAULT_PARAMETERS, **dataset_parameters}
        dataset_parameter_string = parameters_to_string(dataset_parameters)
        dictionary_filename = ARXMLIV_OUTPUT_FILENAME.format(math_representation, '{}.dictionary'.format(dataset_parameter_string))
        tfidf_filename = ARXMLIV_OUTPUT_FILENAME.format(math_representation, '{}.tfidf'.format(dataset_parameter_string))
        phraser_filename = ARXMLIV_OUTPUT_FILENAME.format(math_representation, '{}.phraser'.format(dataset_parameter_string))

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
            'json_filename': json_filename,
            'json_num_paragraphs': json_num_paragraphs,
            'dataset_parameters': dataset_parameters,
            'fasttext_parameters': fasttext_parameters,
            'fasttext_filename': fasttext_filename,
            'termsim_matrix_parameters': termsim_matrix_parameters,
            'termsim_index_parameters': termsim_index_parameters,
            'scm_filename': scm_filename,
            'dictionary_filename': dictionary_filename,
            'tfidf_filename': tfidf_filename,
            'phraser_filename': phraser_filename,
            'corpus_filename': corpus_filename,
            'corpus_num_documents': corpus_num_documents,
            'validation_result_filename': validation_result_filename,
            'discard_math': discard_math,
        }

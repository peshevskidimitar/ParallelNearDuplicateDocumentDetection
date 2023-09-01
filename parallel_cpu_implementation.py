import argparse
import random
import string
import warnings
from functools import wraps
from time import perf_counter_ns

import numba
import numpy as np
from binascii import crc32

from numba import prange, set_num_threads

warnings.filterwarnings('ignore')

set_num_threads(2)


def benchmark(iters, phase):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)

            timing = np.empty(iters)
            for i in range(timing.size):
                start = perf_counter_ns()
                result = func(*args, **kwargs)
                end = perf_counter_ns()
                timing[i] = (end - start)
            timing *= 1e-6
            print(f'Elapsed time for {phase}: {timing.mean():.2f} ± {timing.std():.2f} ms')

            return result

        return wrapper

    return decorator


def parse_args():
    parser = argparse.ArgumentParser(
        description='Benchmarking of parallel CPU implementation for detecting near-duplicate documents.'
    )

    parser.add_argument('-i', '--iterations', type=int, default=101, help='The number of times the code is executed.')
    parser.add_argument('-d', '--documents', type=int, default=1000, help='The number of documents.')

    return parser.parse_args()


arguments = parse_args()

iterations = arguments.iterations
num_docs = arguments.documents


@benchmark(iterations, '1. phase')
def parse_data(filename):
    with open(filename, 'r') as file:
        array = list()
        for line in file:
            doc_id, content = line.split(' ', 1)
            content = content.translate(str.maketrans('', '', string.punctuation))
            content = content.translate(
                str.maketrans('', '', string.whitespace))
            content = content.lower()

            array.append((doc_id, content))

    return array


def shingle_document(text, k):
    return np.unique([crc32(text[i:(i + k)].encode('utf-8')) & 0xffffffff for i in range(len(text) - k + 1)]).astype(
        np.int32)


@benchmark(iterations, '2. phase')
def shingle_documents(docs):
    return [(doc_id, shingle_document(article, 5)) for doc_id, article in docs]


@benchmark(iterations, 'overhead of 2. phase')
def convert_to_crs_format(shingled_documents):
    docs_shingles = np.concatenate([shingled_document for doc_id, shingled_document in shingled_documents])
    docs_indices = [0]
    for index in range(1, len(shingled_documents) + 1):
        docs_indices.append(docs_indices[index - 1] + len(shingled_documents[index - 1][1]))
    docs_indices = np.array(docs_indices)

    return docs_shingles, docs_indices


def get_hash_coefficients(p=2 ** 33 - 355):
    coefficient1 = random.randint(1, p - 1)
    coefficient2 = random.randint(0, p - 1)
    return coefficient1, coefficient2


def generate_hash_coefficients(num_hash):
    return np.array([get_hash_coefficients() for _ in range(num_hash)])


@numba.njit(parallel=True)
def create_signature_matrix(docs_shingles, docs_indices, hash_coefficients, signature_matrix, p, m):
    num_documents = docs_indices.shape[0] - 1
    for doc_index in prange(0, num_documents):
        num_hashes = hash_coefficients.shape[0]
        for hash_index in range(0, num_hashes):
            start_index = docs_indices[doc_index]
            end_index = docs_indices[doc_index + 1]
            min_hash = ((hash_coefficients[hash_index][0] * docs_shingles[start_index] + hash_coefficients[hash_index][
                1]) % p) % m
            for j in range(start_index + 1, end_index):
                min_hash = min(min_hash, ((hash_coefficients[hash_index][0] * docs_shingles[j] +
                                           hash_coefficients[hash_index][1]) % p) % m)
            signature_matrix[doc_index][hash_index] = min_hash


@benchmark(iterations, '3. phase')
def create_signature_matrix_wrapper(docs_shingles, docs_indices, num_hashes):
    hash_coefficients = generate_hash_coefficients(num_hashes)
    signature_matrix = np.zeros((num_docs, num_hashes), dtype=np.int32)
    create_signature_matrix(docs_shingles, docs_indices, hash_coefficients, signature_matrix, p=2 ** 33 - 355,
                            m=4294967295)

    return signature_matrix


@numba.njit(parallel=True)
def estimate_document_similarity(signature_matrix, document_similarities, threshold, count_duplicates):
    num_documents = signature_matrix.shape[0]
    for doc_i in prange(0, num_documents):
        for doc_j in range(0, (num_documents - 1) - doc_i):
            num_hashes = signature_matrix.shape[1]
            count = 0
            for i in range(num_hashes):
                if signature_matrix[doc_i][i] == signature_matrix[(doc_i + 1) + doc_j][i]:
                    count += 1
            start_index = int(doc_i * num_documents - doc_i * (doc_i + 1) / 2)
            estimated_similarity = float(count) / num_hashes
            document_similarities[start_index + doc_j] = estimated_similarity
            if estimated_similarity > threshold:
                count_duplicates[0] += 1


@benchmark(iterations, '4. phase')
def estimate_document_similarity_wrapper(signature_matrix, similarity_threshold):
    num_comparisons = int(num_docs * (num_docs + 1) / 2 - num_docs)
    document_similarities = np.zeros((num_comparisons,), dtype=np.float64)
    count_duplicates = [0]
    estimate_document_similarity(signature_matrix, document_similarities, similarity_threshold, count_duplicates)

    return document_similarities, count_duplicates


@numba.njit(parallel=True)
def extract_duplicate_documents(num_documents, document_similarities, estimated_duplicates, threshold,
                                duplicates_index):
    for doc_index in prange(0, num_documents):
        for index in range(0, (num_documents - 1) - doc_index):
            count1 = int((num_documents - 1) * ((num_documents - 1) + 1) / 2)
            count2 = int(((num_documents - 1) - doc_index) * (((num_documents - 1) - doc_index) + 1) / 2)
            if document_similarities[count1 - count2 + index] > threshold:
                estimated_duplicates[duplicates_index[0]][0] = doc_index
                estimated_duplicates[duplicates_index[0]][1] = doc_index + index + 1
                duplicates_index[0] += 1


@benchmark(iterations, '5. phase')
def extract_duplicate_documents_wrapper(document_similarities, count_duplicates, similarity_threshold, documents):
    duplicates_index = np.zeros((1,), dtype=np.int64)
    estimated_duplicates = np.zeros((count_duplicates[0], 2), dtype=np.int64)
    extract_duplicate_documents(num_docs, document_similarities, estimated_duplicates, similarity_threshold,
                                duplicates_index)

    duplicates = list()
    for i, j in estimated_duplicates:
        id1, article1 = documents[i]
        id2, article2 = documents[j]
        duplicates.append((id1, id2))

    return duplicates


def check(filename, duplicates):
    true_duplicates = []
    with open(filename, 'r') as file:
        for line in file:
            doc_id1, doc_id2 = line.strip().split(' ')
            true_duplicates.append((doc_id1, doc_id2))

    true_duplicates = sorted(true_duplicates)
    duplicates = sorted(duplicates)

    if len(true_duplicates) != len(duplicates):
        return False

    for index in range(len(true_duplicates)):
        if true_duplicates[index][0] == duplicates[index][0] and \
                true_duplicates[index][1] == duplicates[index][1]:
            continue

        return False

    return True


def main():
    documents = parse_data(f'data/articles_{num_docs}.train')
    shingled_documents = shingle_documents(documents)
    docs_shingles, docs_indices = convert_to_crs_format(shingled_documents)
    num_hashes = 64
    signature_matrix = create_signature_matrix_wrapper(docs_shingles, docs_indices, num_hashes)
    threshold = 0.90
    document_similarities, count_duplicates = estimate_document_similarity_wrapper(signature_matrix, threshold)
    duplicates = extract_duplicate_documents_wrapper(document_similarities, count_duplicates, threshold, documents)
    print(check(f'data/articles_{num_docs}.truth', duplicates))


main()

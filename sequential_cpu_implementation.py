import argparse
import random
import string
from functools import wraps
from time import perf_counter_ns

import numpy as np
from binascii import crc32


def benchmark(iters, phase):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = None

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
        description='Benchmarking of sequential CPU implementation for detecting near-duplicate documents.'
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
    return np.unique([crc32(text[i:(i + k)].encode('utf-8')) & 0xffffffff for i in range(len(text) - k + 1)]).astype(np.int32)


@benchmark(iterations, '2. phase')
def shingle_documents(docs):
    return [(doc_id, shingle_document(doc, 5)) for doc_id, doc in docs]


def get_hash_coefficients(p=2 ** 33 - 355):
    coefficient1 = random.randint(1, p - 1)
    coefficient2 = random.randint(0, p - 1)
    return coefficient1, coefficient2


def generate_hash_coefficients(num_hashes):
    return np.array([get_hash_coefficients() for _ in range(num_hashes)])


@benchmark(iterations, '3. phase')
def make_minhash_signature(shingled_documents, num_hashes, p=2 ** 33 - 355, m=4294967295):
    hash_coefficients = generate_hash_coefficients(num_hashes)

    signature_matrix = []
    for doc_id, document in shingled_documents:
        signature = []
        for a, b in hash_coefficients:
            min_hash = (((a * np.array(list(document)) + b) % p) % m).min()
            signature.append(min_hash)
        signature_matrix.append(np.array(signature))
    return signature_matrix


@benchmark(iterations, '4. phase')
def estimate_documents_similarities(num_documents, num_hashes, minhash_signature):
    est_sim_matrix = []
    for i in range(num_documents):
        est_sim_matrix.append([])
        for j in range(i + 1, num_documents):
            count = np.count_nonzero(minhash_signature[i] == minhash_signature[j])
            est_sim_matrix[i].append(float(count) / num_hashes)

    return est_sim_matrix


@benchmark(iterations, '5. phase')
def estimate_document_duplicates(shingled_docs, est_sim_matrix, threshold):
    est_duplicates = []
    for i in range(len(est_sim_matrix)):
        for j in range(len(est_sim_matrix[i])):
            if est_sim_matrix[i][j] > threshold:
                est_duplicates.append((shingled_docs[i][0], shingled_docs[(i + 1) + j][0]))

    return est_duplicates


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
    num_hashes = 64
    minhash_signature = make_minhash_signature(shingled_documents, num_hashes)
    est_sim_matrix = estimate_documents_similarities(num_docs, num_hashes, minhash_signature)
    threshold = 0.90
    est_duplicates = estimate_document_duplicates(shingled_documents, est_sim_matrix, threshold)
    print(check(f'data/articles_{num_docs}.truth', est_duplicates))


main()

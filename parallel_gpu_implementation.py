import argparse
import random
import string
import warnings
from functools import wraps
from time import perf_counter_ns

import numba
import numpy as np
from binascii import crc32

from numba import cuda

warnings.filterwarnings('ignore')


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
    parser.add_argument('-d', '--documents', type=int, default=1000, help='The number of documents')

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


@cuda.jit
def create_signature_matrix_kernel(docs_shingles, docs_indices, hash_coefficients, signature_matrix, p, m):
    block_index = cuda.blockIdx.x
    num_documents = docs_indices.shape[0] - 1
    if block_index < num_documents:
        thread_index = cuda.threadIdx.x
        num_hashes = hash_coefficients.shape[0]
        start_index = docs_indices[block_index]
        end_index = docs_indices[block_index + 1]
        min_hash = ((hash_coefficients[thread_index][0] * docs_shingles[start_index] + hash_coefficients[thread_index][
            1]) % p) % m
        for j in range(start_index + 1, end_index):
            min_hash = min(min_hash,
                           ((hash_coefficients[thread_index][0] * docs_shingles[j] + hash_coefficients[thread_index][
                               1]) % p) % m)
        signature_matrix[block_index][thread_index] = min_hash


@benchmark(iterations, '3. phase')
def create_signature_matrix_wrapper(docs_shingles, docs_indices, num_hashes):
    hash_coefficients = generate_hash_coefficients(num_hashes)

    dev_docs_shingles = cuda.to_device(docs_shingles)
    dev_docs_indices = cuda.to_device(docs_indices)
    dev_hash_coefficients = cuda.to_device(hash_coefficients)
    dev_signature_matrix = cuda.to_device(np.zeros((num_docs, num_hashes), dtype=np.int32))
    create_signature_matrix_kernel[num_docs, num_hashes](dev_docs_shingles, dev_docs_indices, dev_hash_coefficients,
                                                         dev_signature_matrix, 2 ** 33 - 355, 4294967295)
    cuda.synchronize()

    return dev_signature_matrix


@cuda.jit
def estimate_document_similarity_kernel(signature_matrix, document_similarities, threshold, count_duplicates):
    block_index = cuda.blockIdx.x
    num_documents = signature_matrix.shape[0]
    if block_index < num_documents:
        thread_index = cuda.threadIdx.x
        for index in range(thread_index, (cuda.gridDim.x - 1) - block_index, cuda.blockDim.x):
            num_hashes = signature_matrix.shape[1]
            count = 0
            for i in range(num_hashes):
                if signature_matrix[block_index][i] == signature_matrix[(block_index + 1) + index][i]:
                    count += 1
            start_index = int(block_index * cuda.gridDim.x - block_index * (block_index + 1) / 2)
            estimated_similarity = float(count) / num_hashes
            document_similarities[start_index + index] = estimated_similarity
            if estimated_similarity > threshold:
                cuda.atomic.add(count_duplicates, 0, 1)


@benchmark(iterations, '4. phase')
def estimate_document_similarity_wrapper(dev_signature_matrix, threshold):
    num_comparisons = int(num_docs * (num_docs + 1) / 2 - num_docs)

    dev_document_similarities = cuda.to_device(np.zeros((num_comparisons,), dtype=np.float64))
    dev_count_duplicates = cuda.to_device(np.zeros((1,), dtype=np.int64))
    estimate_document_similarity_kernel[num_docs, 512](dev_signature_matrix, dev_document_similarities,
                                                       threshold, dev_count_duplicates)
    count_duplicates = dev_count_duplicates.copy_to_host()
    cuda.synchronize()

    return dev_document_similarities, count_duplicates


@cuda.jit(device=True)
def lock(mutex):
    while cuda.atomic.compare_and_swap(mutex, 0, 1) != 0:
        pass
    cuda.threadfence()


@cuda.jit(device=True)
def unlock(mutex):
    cuda.threadfence()
    cuda.atomic.exch(mutex, 0, 0)


@cuda.jit
def extract_duplicate_documents(num_documents, document_similarities, estimated_duplicates, threshold,
                                duplicates_array_index,
                                mutex):
    block_index = cuda.blockIdx.x
    if block_index < num_documents:
        thread_index = cuda.threadIdx.x
        for index in range(thread_index, (num_documents - 1) - block_index, cuda.blockDim.x):
            count1 = int((num_documents - 1) * ((num_documents - 1) + 1) / 2)
            count2 = int(((num_documents - 1) - block_index) * (((num_documents - 1) - block_index) + 1) / 2)
            if document_similarities[count1 - count2 + index] > threshold:
                lock(mutex)
                estimated_duplicates[duplicates_array_index[0]][0] = block_index
                estimated_duplicates[duplicates_array_index[0]][1] = block_index + index + 1
                duplicates_array_index[0] += 1
                unlock(mutex)


@benchmark(iterations, '5. phase')
def extract_duplicate_documents_wrapper(documents, count_duplicates, dev_document_similarities, similarity_threshold):
    dev_duplicates_array_index = cuda.to_device(np.zeros((1,), dtype=np.int64))
    dev_mutex = cuda.to_device(np.zeros((1,), dtype=np.int64))
    dev_estimated_duplicates = cuda.device_array((count_duplicates[0], 2), dtype=np.int64)
    extract_duplicate_documents[num_docs, 64](num_docs, dev_document_similarities, dev_estimated_duplicates,
                                              similarity_threshold, dev_duplicates_array_index, dev_mutex)
    est_duplicates = dev_estimated_duplicates.copy_to_host()
    cuda.synchronize()

    duplicates = list()
    for i, j in est_duplicates:
        id1, article1 = documents[i]
        id2, article2 = documents[j]
        duplicates.append((id1, id2))

    return duplicates


def main():
    documents = parse_data(f'data/articles_{num_docs}.train')
    shingled_documents = shingle_documents(documents)
    docs_shingles, docs_indices = convert_to_crs_format(shingled_documents)
    num_hashes = 64
    dev_signature_matrix = create_signature_matrix_wrapper(docs_shingles, docs_indices, num_hashes)
    threshold = 0.90
    dev_document_similarities, count_duplicates = estimate_document_similarity_wrapper(dev_signature_matrix,
                                                                                       threshold)
    duplicates = extract_duplicate_documents_wrapper(documents, count_duplicates, dev_document_similarities, threshold)
    print(duplicates)


main()

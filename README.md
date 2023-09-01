# Parallel near-duplicate document detection using general-purpose GPU

Within this project, we have developed three distinct implementations for detecting near-duplicate documents.
These implementations include a sequential CPU approach, a parallel CPU approach, and a parallel GPU approach.

For this project, the programming language Python was used because it is powerful, flexible, and easy to use.
When it comes to utilizing the GPU for the parallelization of tasks, Numba was used as it supports CUDA GPU
programming by directly compiling a restricted subset of Python code into CUDA kernels and device functions
following the CUDA execution model making it possible to leverage massively parallel GPU computing to achieve
faster results and accuracy. With Numba, one can write kernels directly with (a subset of) Python, and Numba
will compile the code on the fly and run it. Numba was also utilized for the parallel implementation on the CPU
because of its support for multithreading.

One can execute any of the implementations from the command line by providing two arguments: one for the 
number of iterations and the other for the number of documents within the dataset.

### Running the sequential CPU implementation
`python sequential_cpu_implementation --iterations 101 --documents 1000`
or

`python sequential_cpu_implementation -i 101 -d 1000`

### Running the parallel CPU implementation
`python parallel_cpu_implementation --iterations 101 --documents 1000`
or

`python parallel_cpu_implementation -i 101 -d 1000`

### Running the parallel GPU implementation
`python parallel_gpu_implementation --iterations 101 --documents 1000`
or

`python parallel_gpu_implementation -i 101 -d 1000`
## SYMnmf Project

This repository implements the Symmetric Nonnegative Matrix Factorization (SymNMF) algorithm as described in the project assignment, as well as compares it to KMEANS. The implementation includes both Python and C components, with bindings through the Python C API. 

## Overview

SymNMF is used for clustering by factorizing a symmetric similarity matrix into a product of low-rank nonnegative matrices. The project includes:

Preprocessing steps: building the similarity matrix, degree matrix, and normalized similarity matrix.

Optimization loop: updating the matrix H until convergence.

Clustering interpretation based on the factorization result .

## Repository Structure
analysis.py — script for testing and analyzing outputs. compares performance between KMEANS and SYMNMF

symnmf.py — main Python program, including high-level interface for SymNMF.

symnmf.c — core C implementation of the SymNMF algorithm.

symnmf.h — header file for the C implementation.

symnmfmodule.c — Python C API wrapper connecting the C code with Python.

setup.py — build script for compiling the C extension.

Makefile — provides shortcuts for compilation and running.

input_example - provides an example for the input points format

## Build and Run

1. Compile the C extension:
  ```
  python setup.py build_ext --inplace 
  ```
or use the provided Makefile.

2. Run the Python program:
   ```
   python symnmf.py <input_file> <goal> <output_file>

   ```
  where <goal> specifies the stage (e.g., similarity, degree, norm, symnmf).



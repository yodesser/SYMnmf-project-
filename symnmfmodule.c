#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "symnmf.h"
#include <stdio.h>
#include <stdlib.h>

/* Build a coordinate linked list (cord) from a Python list of floats */
static cord* build_cord(PyObject* element){
    int d, i;
    cord *head_cord, *curr_cord, *tmp;

    d = (int)PyObject_Length(element);
    head_cord = new_cord(NULL, PyFloat_AsDouble(PyList_GetItem(element, 0)));
    if(head_cord == NULL) return NULL;
    curr_cord = head_cord;

    for(i = 1; i < d; ++i){
        tmp = new_cord(head_cord, PyFloat_AsDouble(PyList_GetItem(element, i)));
        if(tmp == NULL) return NULL;
        curr_cord->next = tmp;
        curr_cord = curr_cord->next;
    }
    return head_cord;
}

/* Build a vector linked list from a 2D Python list */
static vector* build_elements(PyObject* pelements, int N){
    int i;
    vector *all = new_vector(NULL, build_cord(PyList_GetItem(pelements, 0)));
    vector *curr = all, *tmp;
    if(curr == NULL) return NULL;
    for(i = 1; i < N; ++i){
        tmp = new_vector(all, build_cord(PyList_GetItem(pelements, i)));
        if(tmp == NULL) return NULL;
        curr->next = tmp;
        curr = curr->next;
    }
    return all;
}

/* Convert 2D Python list -> newly allocated C matrix (R x C) */
static double** matrixFromPyObject(PyObject* list, int R, int C){
    int i, j;
    double **A = create_matrix(R, C);
    PyObject *row;
    if(A == NULL) return NULL;
    for(i = 0; i < R; ++i){
        row = PyList_GetItem(list, i);
        for(j = 0; j < C; ++j){
            A[i][j] = PyFloat_AsDouble(PyList_GetItem(row, j));
        }
    }
    return A;
}

/* Convert C matrix (R x C) -> new Python 2D list */
static PyObject* matrixToPyObject(double** A, int R, int C){
    int i, j;
    PyObject *outer, *row;
    if(A == NULL) return NULL;
    outer = PyList_New(R);
    if(outer == NULL) return NULL;
    for(i = 0; i < R; ++i){
        row = PyList_New(C);
        if(row == NULL) return NULL;
        for(j = 0; j < C; ++j){
            PyList_SetItem(row, j, PyFloat_FromDouble(A[i][j]));
        }
        PyList_SetItem(outer, i, row);
    }
    return outer;
}

/* Python wrappers */

static PyObject* sym_wrapper(PyObject *self, PyObject *args){
    int N;
    PyObject *datapoints, *res;
    double **A;
    vector *X;

    if(!PyArg_ParseTuple(args, "O", &datapoints)) return NULL;
    N = (int)PyObject_Length(datapoints);
    X = build_elements(datapoints, N);
    A = sym(X, N);
    res = matrixToPyObject(A, N, N);
    delete_list_vector(X, 1);
    free_matrix(A, N);
    return res;
}

static PyObject* ddg_wrapper(PyObject *self, PyObject *args){
    int N;
    PyObject *datapoints, *res;
    double **D;
    vector *X;

    if(!PyArg_ParseTuple(args, "O", &datapoints)) return NULL;
    N = (int)PyObject_Length(datapoints);
    X = build_elements(datapoints, N);
    D = ddg(X, N);
    res = matrixToPyObject(D, N, N);
    delete_list_vector(X, 1);
    free_matrix(D, N);
    return res;
}

static PyObject* norm_wrapper(PyObject *self, PyObject *args){
    int N;
    PyObject *datapoints, *res;
    double **W;
    vector *X;

    if(!PyArg_ParseTuple(args, "O", &datapoints)) return NULL;
    N = (int)PyObject_Length(datapoints);
    X = build_elements(datapoints, N);
    W = norm(X, N);
    res = matrixToPyObject(W, N, N);
    delete_list_vector(X, 1);
    free_matrix(W, N);
    return res;
}

static PyObject* symnmf_wrapper(PyObject *self, PyObject *args){
    int N, k;
    PyObject *H_py, *W_py, *res;
    double **H, **W, **Hf;

    if(!PyArg_ParseTuple(args, "OOi", &H_py, &W_py, &k)) return NULL;
    N = (int)PyObject_Length(H_py);
    H = matrixFromPyObject(H_py, N, k);
    W = matrixFromPyObject(W_py, N, N);
    Hf = symnmf(H, W, N, k);
    res = matrixToPyObject(Hf, N, k);
    free_matrix(H, N);
    free_matrix(W, N);
    free_matrix(Hf, N);
    return res;
}

static PyMethodDef symnmfMethods[] = {
    {"sym",    (PyCFunction)sym_wrapper,    METH_VARARGS, PyDoc_STR("sym(X): return similarity matrix W from datapoints X")},
    {"ddg",    (PyCFunction)ddg_wrapper,    METH_VARARGS, PyDoc_STR("ddg(X): return diagonal degree matrix D from datapoints X")},
    {"norm",   (PyCFunction)norm_wrapper,   METH_VARARGS, PyDoc_STR("norm(X): return normalized similarity matrix D^{-1/2} W D^{-1/2}")},
    {"symnmf", (PyCFunction)symnmf_wrapper, METH_VARARGS, PyDoc_STR("symnmf(H, W, k): multiplicative updates; return refined H")},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef symnmfmodule = {
    PyModuleDef_HEAD_INIT,
    "mysymnmf",
    NULL,
    -1,
    symnmfMethods
};

PyMODINIT_FUNC PyInit_mysymnmf(void){
    PyObject *m = PyModule_Create(&symnmfmodule);
    if(!m) return NULL;
    return m;
}
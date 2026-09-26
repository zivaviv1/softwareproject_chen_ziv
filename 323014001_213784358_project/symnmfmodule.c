#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <limits.h>

#include "symnmf.h"

typedef Matrix *(*MatrixOperation)(const Matrix *);

static int matrix_shape(PyObject *outer, Py_ssize_t *rows, Py_ssize_t *cols);
static int fill_matrix(PyObject *outer, Matrix *matrix);
static Matrix *matrix_from_py(PyObject *object);
static PyObject *matrix_row(const Matrix *matrix, int row);
static PyObject *matrix_to_py(const Matrix *matrix);
static PyObject *run_operation(PyObject *args, MatrixOperation operation);
static PyObject *run_symnmf(PyObject *args);

/* Get the number of rows and columns of a Python list of lists */
static int matrix_shape(PyObject *outer, Py_ssize_t *rows, Py_ssize_t *cols)
{
    PyObject *row;

    *rows = PySequence_Fast_GET_SIZE(outer);
    if (*rows <= 0 || *rows > INT_MAX) {
        PyErr_SetString(PyExc_ValueError, "matrix must have valid rows");
        return 0;
    }
    row = PySequence_Fast(PySequence_Fast_GET_ITEM(outer, 0),
                          "matrix rows must be sequences");
    if (row == NULL) {
        return 0;
    }
    *cols = PySequence_Fast_GET_SIZE(row);
    Py_DECREF(row);
    if (*cols <= 0 || *cols > INT_MAX) {
        PyErr_SetString(PyExc_ValueError, "matrix must have valid columns");
        return 0;
    }
    return 1;
}

/* Copies the Python values into the C matrix, checks every row has the same length */
static int fill_matrix(PyObject *outer, Matrix *matrix)
{
    PyObject *row;
    PyObject *item;
    Py_ssize_t index;
    int column;
    double value;

    for (index = 0; index < matrix->rows; index++) {
        row = PySequence_Fast(PySequence_Fast_GET_ITEM(outer, index),
                              "matrix rows must be sequences");
        if (row == NULL) {
            return 0;
        }
        if ((int)PySequence_Fast_GET_SIZE(row) != matrix->cols) {
            Py_DECREF(row);
            PyErr_SetString(PyExc_ValueError, "matrix rows must match");
            return 0;
        }
        for (column = 0; column < matrix->cols; column++) {
            item = PySequence_Fast_GET_ITEM(row, column);
            value = PyFloat_AsDouble(item);
            if (PyErr_Occurred()) {
                Py_DECREF(row);
                return 0;
            }
            matrix->data[index * matrix->cols + column] = value;
        }
        Py_DECREF(row);
    }
    return 1;
}

/* Converts a Python list of lists into a C Matrix */
static Matrix *matrix_from_py(PyObject *object)
{
    PyObject *outer;
    Matrix *matrix;
    Py_ssize_t rows;
    Py_ssize_t cols;

    outer = PySequence_Fast(object, "matrix must be a sequence");
    if (outer == NULL) {
        return NULL;
    }
    if (!matrix_shape(outer, &rows, &cols)) {
        Py_DECREF(outer);
        return NULL;
    }
    matrix = matrix_create((int)rows, (int)cols);
    if (matrix == NULL) {
        Py_DECREF(outer);
        PyErr_NoMemory();
        return NULL;
    }
    if (!fill_matrix(outer, matrix)) {
        matrix_free(matrix);
        matrix = NULL;
    }
    Py_DECREF(outer);
    return matrix;
}

/* Converts one row of the matrix into a Python list of floats */
static PyObject *matrix_row(const Matrix *matrix, int row)
{
    PyObject *result;
    PyObject *value;
    int column;

    result = PyList_New(matrix->cols);
    if (result == NULL) {
        return NULL;
    }
    for (column = 0; column < matrix->cols; column++) {
        value = PyFloat_FromDouble(matrix->data[row * matrix->cols + column]);
        if (value == NULL) {
            Py_DECREF(result);
            return NULL;
        }
        PyList_SET_ITEM(result, column, value);
    }
    return result;
}

/* Converts a C Matrix into a Python list of lists */
static PyObject *matrix_to_py(const Matrix *matrix)
{
    PyObject *result;
    PyObject *row;
    int index;

    result = PyList_New(matrix->rows);
    if (result == NULL) {
        return NULL;
    }
    for (index = 0; index < matrix->rows; index++) {
        row = matrix_row(matrix, index);
        if (row == NULL) {
            Py_DECREF(result);
            return NULL;
        }
        PyList_SET_ITEM(result, index, row);
    }
    return result;
}

/* Shared code for sym/ddg/norm: parse the points, run the operation,
   return the result as a Python list of lists */
static PyObject *run_operation(PyObject *args, MatrixOperation operation)
{
    PyObject *object;
    PyObject *result;
    Matrix *input;
    Matrix *output;

    if (!PyArg_ParseTuple(args, "O", &object)) {
        return NULL;
    }
    input = matrix_from_py(object);
    if (input == NULL) {
        return NULL;
    }
    output = operation(input);
    matrix_free(input);
    if (output == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "matrix calculation failed");
        return NULL;
    }
    result = matrix_to_py(output);
    matrix_free(output);
    return result;
}

/* Parses (H, W, max_iter, epsilon), runs symnmf and returns the final H */
static PyObject *run_symnmf(PyObject *args)
{
    PyObject *h_object;
    PyObject *w_object;
    PyObject *result;
    Matrix *h;
    Matrix *w;
    Matrix *output;
    int max_iter;
    double epsilon;

    /* defaults in case max_iter and epsilon are not passed */
    max_iter = 300;
    epsilon = 0.0001;
    if (!PyArg_ParseTuple(args, "OO|id", &h_object, &w_object,
                          &max_iter, &epsilon)) {
        return NULL;
    }
    h = matrix_from_py(h_object);
    if (h == NULL) {
        return NULL;
    }
    w = matrix_from_py(w_object);
    if (w == NULL) {
        matrix_free(h);
        return NULL;
    }
    output = symnmf(h, w, max_iter, epsilon);
    matrix_free(h);
    matrix_free(w);
    if (output == NULL) {
        PyErr_SetString(PyExc_ValueError, "invalid symnmf matrices");
        return NULL;
    }
    result = matrix_to_py(output);
    matrix_free(output);
    return result;
}

static PyObject *py_symnmf(PyObject *self, PyObject *args)
{
    (void)self;
    return run_symnmf(args);
}

static PyObject *py_sym(PyObject *self, PyObject *args)
{
    (void)self;
    return run_operation(args, calculate_sym);
}

static PyObject *py_ddg(PyObject *self, PyObject *args)
{
    (void)self;
    return run_operation(args, calculate_ddg);
}

static PyObject *py_norm(PyObject *self, PyObject *args)
{
    (void)self;
    return run_operation(args, calculate_norm);
}

static PyMethodDef symnmf_methods[] = {
    {"symnmf", py_symnmf, METH_VARARGS,
     "symnmf(h, w, max_iter=300, epsilon=0.0001)\n"
     "Takes the initial H (n x k list of lists), the normalized similarity "
     "matrix W (n x n list of lists), the maximum number of iterations and "
     "the convergence epsilon. Returns the optimized H as a list of lists."},
    {"sym", py_sym, METH_VARARGS,
     "sym(points)\n"
     "Takes the data points (n x d list of lists) and returns the n x n "
     "similarity matrix as a list of lists."},
    {"ddg", py_ddg, METH_VARARGS,
     "ddg(points)\n"
     "Takes the data points (n x d list of lists) and returns the n x n "
     "diagonal degree matrix as a list of lists."},
    {"norm", py_norm, METH_VARARGS,
     "norm(points)\n"
     "Takes the data points (n x d list of lists) and returns the n x n "
     "normalized similarity matrix as a list of lists."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef symnmf_module = {
    PyModuleDef_HEAD_INIT,
    "symnmfmodule",
    "SymNMF C extension.",
    -1,
    symnmf_methods,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC PyInit_symnmfmodule(void)
{
    return PyModule_Create(&symnmf_module);
}
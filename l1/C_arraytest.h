/* Header to test of C modules for arrays for Python: C_test.c */

/* ==== Prototypes =================================== */
#include <numpy/arrayobject.h>

// .... Python callable Vector functions ..................
static PyObject *IMGaffine_c(PyObject *self, PyObject *args);

/* .... C vector utility functions ..................*/
int  not_doublematrix(PyArrayObject *mat);
void matrix_multiply(double *x, int x1, int x2, double *y, int y1, int y2, double *z);

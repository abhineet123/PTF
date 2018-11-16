#include <numpy/arrayobject.h>


static PyObject *softresh_c(PyObject *self, PyObject *args);
int  not_doublematrix(PyArrayObject *mat);
double max_num(double a,double b);

/*=================================================================
 *
 * IMGaffine_c.c	This is a C code for Matlab.
 *	        
 *=================================================================*/
#include <Python.h>
#include <math.h>
#include <numpy/arrayobject.h>
#include "softresh_arraytest.h"


/* ==== Set up the methods table ====================== */
static PyMethodDef softreshMethods[] = {
    {"softresh_c", softresh_c, METH_VARARGS},
    {NULL, NULL}     /* Sentinel - marks the end of this structure */
};

/* ==== Initialize the C_test functions ====================== */
void initsoftresh()  {
    (void) Py_InitModule("softresh", softreshMethods);
    import_array();  // Must be present for NumPy.  Called first after above line.
}


/*wrapped function*/
static PyObject* softresh_c(PyObject* self, PyObject* args)
{
	/*input*/
	PyArrayObject *in_x_py, *out_x_py;
	int dims[2],i;
	double *in_x,*out_x;
	double lambda;
	
	/*parse first input array*/ 	
	if (!PyArg_ParseTuple(args, "O!d", &PyArray_Type, &in_x_py, &lambda))
		return NULL;
	if (NULL == in_x_py)	return NULL;
	
	if(not_doublematrix(in_x_py)) return NULL;
	
	dims[0] = in_x_py->dimensions[0];
	dims[1] = in_x_py->dimensions[1];
	in_x = (double*)in_x_py->data;
	
	out_x_py=(PyArrayObject *) PyArray_FromDims(2,dims,NPY_DOUBLE);
	out_x = (double*)out_x_py->data;

	for(i=0;i<dims[0];i++){
		
		*(out_x+i) = max_num(((*(in_x+i)))-lambda,0) - max_num(-((*(in_x+i)))-lambda,0);
		
		}
	
	/* Make a new double matrix of same dims */
	return Py_BuildValue("N",PyArray_Return(out_x_py));
	
}


double max_num(double a,double b){
	
	if(a>b) return a;
	else return b;
	
	
	}
	
	
int  not_doublematrix(PyArrayObject *mat){
     if (mat->descr->type_num != NPY_DOUBLE || mat->nd != 2){
        PyErr_SetString(PyExc_ValueError,"In not_doublematrix: array must be of type Float and 2 dimensional (n x m).");
         return 1;  }
     return 0;
}

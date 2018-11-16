/*=================================================================
 *
 * IMGaffine_c.c	This is a C code for Matlab.
 *	        
 *=================================================================*/
#include <Python.h>
#include <math.h>
#include <numpy/arrayobject.h>
#include "C_arraytest.h"


/* ==== Set up the methods table ====================== */
static PyMethodDef IMGaffineMethods[] = {
    {"IMGaffine_c", IMGaffine_c, METH_VARARGS},
    {NULL, NULL}     /* Sentinel - marks the end of this structure */
};

/* ==== Initialize the C_test functions ====================== */
// Module name must be _C_arraytest in compile and linked
void initIMGaffine()  {
    (void) Py_InitModule("IMGaffine", IMGaffineMethods);
    import_array();  // Must be present for NumPy.  Called first after above line.
}

/* wrapped function*/
static PyObject* IMGaffine_c(PyObject* self, PyObject* args)
{
	/*input*/
	PyArrayObject *in_img_py;
	PyArrayObject *affnv_py;
	PyArrayObject *temp_sz_py;
	PyArrayObject *matout, *matout1;
		
	/*wrap need parameters*/
	double *AFNV;

	/*from old IMGaffine_c.c*/
	int M,N;
	double *temp;
	int M_in,N_in,dims[2];
	double *Rp, *Pp, *Kp, *RINp, *ROUTp;
	double value, value_1, value_2, total_value;
	int i,j,k,Ip;
	double count;
 
	/*parse first input array*/ 	
	if (!PyArg_ParseTuple(args, "O!O!O!", &PyArray_Type, &in_img_py, &PyArray_Type, &affnv_py, &PyArray_Type, &temp_sz_py))
		return NULL;
	if (NULL == in_img_py)	return NULL;
	if (NULL == affnv_py)	return NULL;
	if (NULL == temp_sz_py)	return NULL;
	
	if (not_doublematrix(in_img_py)) return NULL;
	if (not_doublematrix(affnv_py)) return NULL;

	/*size of input and output arrays*/
	M_in = in_img_py->dimensions[0];
	N_in = in_img_py->dimensions[1];
	M = (int)(*(temp_sz_py->data));
	N = (int)(*(temp_sz_py->data + temp_sz_py->strides[1]));
	//M = 12;
	//N = 15;
//	in_img1 = pymatrix_to_C1darrayptrs(in_img);
//	affnv1 = pymatrix_to_C1darrayptrs(affnv);
	
    /*get space for in_img1 and affnv1*/
	Pp = (double* )malloc(3*M*N*sizeof(double));
	Kp = (double* )malloc(3*M*N*sizeof(double));
	Rp = (double* )malloc(9*sizeof(double));	
	temp = (double* )malloc(M*N*sizeof(double));
	
	RINp = (double* )in_img_py->data;
	AFNV = (double* )affnv_py->data;
    
	*Rp = *AFNV;
   	*(Rp+1) = *(AFNV+2);
   	*(Rp+2) = 0;
	*(Rp+3) = *(AFNV+1);
	*(Rp+4) = *(AFNV+3);
    	*(Rp+5) = 0;
    	*(Rp+6) = *(AFNV+4);
    	*(Rp+7) = *(AFNV+5);
    	*(Rp+8) = 1;  
	
	dims[0] = M;
	dims[1] = N;
	matout=(PyArrayObject *) PyArray_FromDims(2,dims,NPY_DOUBLE);
	ROUTp = (double*)matout->data;
    
//	dims[0] = M_in;
//	dims[1] = N_in;
//	M_in = 600;
//	N_in = 800;
//	matout1=(PyArrayObject *) PyArray_FromDims(2,dims,NPY_DOUBLE);
//	temp = (double*) matout1->data;
//	temp = (int* )malloc(9*sizeof(int));	
/*	for(i=0;i<M_in*N_in;i++){
		temp[i] = RINp[i];
	}*/
	for(i=0,j=1,k=1;i<M*N;i++){
        	*(Pp+i*3) = j; j++;  
	        *(Pp+1+i*3) = k;
        	if(j == M+1){
            	    j = 1;
	            k++;
		            }
	        *(Pp+2+i*3) = 1;
    	}
	
	matrix_multiply(Rp,3,3,Pp,3,M*N,Kp);
	
	for(i=0;i<3*M*N;i++){
        	*(Kp+i) = round(*(Kp+i));
    	}
	
	count = 0;
	value = 0;
	total_value = 0;
	for(i=0;i<M*N;i++){
        	*(temp+i) = 0;   
        	//use the first M*N entries of P to restore j.
        	*(Pp+i) = 0;
        	value_1 = *(Kp+0+i*3);
        	value_2 = *(Kp+1+i*3);
        	if((value_1 >= 1) & (value_1 <= M_in)
            		& (value_2 >= 1) & (value_2 < N_in))
       		 {
            	*(Pp+i) = 1;     
            	count++;   
            	//value = *(RINp + (int)((value_2 - 1)*M_in + value_1) - 1);
		value = *(RINp + (int)((value_1-1)*N_in + value_2) - 1);	
           	// value = *(RINp + (ptrdiff_t)((value_2 - 1)*M_in + value_1) - 1);
            	*(temp+i) = value;
            	total_value += value;
       		 }
    	}

	//find mean value
	value = total_value/count;
    	for(i=0;i<M*N;i++)
    	{
        	if(*(Pp+i) == 0)
            		*(temp+i) = value;
    	}

	for(i=0;i<M;i++){
		for(j=0;j<N;j++){
		*(ROUTp+i*N+j) = *(temp+j*M+i);
		}
	}	
	if(count>0)
        	Ip = 1;
    	else 
        	Ip = 0;   
	// Get the output
	
	/*free memory*/
	free(Pp);
	free(Kp);
	free(Rp); 
	free(temp);
	/* Make a new double matrix of same dims */
	return Py_BuildValue("Ni",PyArray_Return(matout),Ip);
}


int  not_doublematrix(PyArrayObject *mat){
     if (mat->descr->type_num != NPY_DOUBLE || mat->nd != 2){
        PyErr_SetString(PyExc_ValueError,"In not_doublematrix: array must be of type Float and 2 dimensional (n x m).");
         return 1;  }
     return 0;
}

void matrix_multiply(double *x, int x1, int x2, double *y, int y1, int y2, double *z)
{
    int i,j,k;
    if(x2!=y1)
        PyErr_SetString(PyExc_ValueError,"Cannot mutiply such matrixs!");
    for(i=0;i<x1;i++)
    {
        for(j=0;j<y2;j++)
        {
            *(z+i+x1*j) = 0;
            for(k=0;k<x2;k++)
                *(z+i+x1*j) += *(x+i+x1*k) * *(y+k+y1*j);
        }
    }
}
double round(double x)
{
    //Matlab's LCC does not contain round(),
    // use floor() to fake one.
    double k = floor(x);
    if( (x - k < 0.5 & k >= 0) | (x - k <= 0.5 & k < 0))
                return k;
            else
                return k+1;
}

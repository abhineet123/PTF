#include <Python.h>
#include <math.h>
#include <numpy/arrayobject.h>
#include "C_arraytest.h"
//#include "softresh_arraytest.h"
//#include<stdlib.h>
//#include<stdio.h>

static PyObject* APGLASSOup_c(PyObject* self, PyObject* args);

static PyMethodDef APGLASSOupMethods[] = {
    {"APGLASSOup_c", APGLASSOup_c, METH_VARARGS},
    {NULL, NULL}     /* Sentinel - marks the end of this structure */
};

void initAPGLASSOup()  {
    (void) Py_InitModule("APGLASSOup", APGLASSOupMethods);
    import_array();  // Must be present for NumPy.  Called first after above line.
}


int  not_doublematrix(PyArrayObject *mat){
     if (mat->descr->type_num != NPY_DOUBLE || mat->nd != 2){
        PyErr_SetString(PyExc_ValueError,"In not_doublematrix: array must be of type Float and 2 dimensional (n x m).");
         return 1;  }
     return 0;
}
double **ptrvector(long n)  {
    double **v;
    v=(double **)malloc((size_t) (n*sizeof(double)));
    if (!v)   {
        printf("In **ptrvector. Allocation of memory for double array failed.");
        exit(0);  }
    return v;
}

double **pymatrix_to_Carrayptrs(PyArrayObject *arrayin)  {
    double **c, *a;
    int i,n,m;

    n=arrayin->dimensions[0];
    m=arrayin->dimensions[1];
    c=ptrvector(n);
    a=(double *) arrayin->data;  /* pointer to arrayin data as double */
    for ( i=0; i<n; i++)  {
        c[i]=a+i*m;  }
    return c;
}
void free_Carrayptrs(double **v) {
   free((char*) v);
}

double max_num(double a,double b){

	if(a>b) return a;
	else return b;
}


static PyObject* APGLASSOup_c(PyObject* self, PyObject* args){

	PyArrayObject *in_b_py, *in_A_py, *in_lambda_py, *out_c_py=NULL;
	//PyArrayObject *out_b_py, *out_A_py, *out_lambda_py;
  	double *b,*b_const, **A,*c,*c_const,*xPrev,*xPrev_const,*temp_lambda,*temp_lambda_const,*tem_y, *tem_y_const,*tem_y1,*tem_y1_const,*lambda;
  	double Lip, lambdaLip, tPrev=1.0,t=1.0,tem_t,temp_yt, tempab, temp_value;
  	int in_Lip, colDim, dims[2],in_nT,nT,in_maxit,maxit,i,j,k,m,n;

  	if (!PyArg_ParseTuple(args, "O!O!O!iii", &PyArray_Type, &in_b_py, &PyArray_Type, &in_A_py,  &PyArray_Type,  &in_lambda_py, &in_Lip, &in_maxit, &in_nT))
		return NULL;

  //printf("here..\n");

  if (NULL == in_b_py)	return NULL;
  if (NULL == in_A_py)	return NULL;
  if (NULL == in_lambda_py)	return NULL;

  if(not_doublematrix(in_b_py)) return NULL;
  if(not_doublematrix(in_A_py)) return NULL;
  if(not_doublematrix(in_lambda_py)) return NULL;

  n = in_A_py->dimensions[0];
  m = in_A_py->dimensions[1];
  A = pymatrix_to_Carrayptrs(in_A_py);


  //printf("here before dims\n");
  dims[0]=colDim = in_b_py->dimensions[0];
  dims[1] = 1;
  b = b_const = (double*)in_b_py->data;
  /* Modified by Jesse */
  xPrev = xPrev_const= (double*)malloc(colDim*sizeof(double));
//  x = (double*)malloc(colDim*sizeof(double));
  tem_y = tem_y_const = (double*)malloc(colDim*sizeof(double));
  tem_y1 = tem_y1_const = (double*)malloc(colDim*sizeof(double));
  //tPrev = 1;
  //t = 1;
  lambda = (double*)in_lambda_py->data;
  Lip = (double)in_Lip;
  //printf("Lip:%f",Lip);
  lambdaLip = lambda[1] / (double)Lip;
  maxit = in_maxit;
  nT = in_nT;
  out_c_py=(PyArrayObject *) PyArray_FromDims(2,dims,NPY_DOUBLE);
  c= c_const = (double*)out_c_py->data;

  temp_lambda = temp_lambda_const = (double*)malloc(colDim*sizeof(double));
  for(i=0;i<colDim;i++){
	  *(c++) = 0.0;
	  //xPrev[i] = 0.0;
	  *(xPrev++) = 0.0;
	  if (i < nT)
	  	//temp_lambda[i] = lambda[0];
		  *(temp_lambda++) = lambda[0];
	  else if (i < colDim - 1)
		  *(temp_lambda++) = 0.0;
	  else
		  *(temp_lambda++) = lambda[0];
	  //printf("temp_lambda[%d]:%f",i,temp_lambda[i]);

  }
  c = c_const;
  xPrev = xPrev_const;
  temp_lambda = temp_lambda_const;
  /* Go to the main loop */

  for(i=0;i<maxit;i++){
	  tem_t = (tPrev-1.0)/t;
	  temp_yt = 0.0;
	  for(j=0;j<colDim;j++){
		  *(tem_y++)= (1.0+tem_t) * *(c++) -tem_t * (*(xPrev++));
	  }
	  tem_y = tem_y_const;
	  c = c_const;
	  xPrev = xPrev_const;
	  for(j=nT;j<=colDim-2;j++){
		    *(temp_lambda++) = lambda[2] * (*(tem_y++));

	  }
	  temp_lambda = temp_lambda_const;
	  tem_y = tem_y_const;
	  for(j=0;j<colDim;j++){
		    *(tem_y1++) = *(tem_y++);
	  }
	  tem_y1 = tem_y1_const;
	  tem_y = tem_y_const;
	  for(j=0;j<=colDim-1;j++){
		  tempab = 0.0;
		  for(k=0;k<=colDim-1;k++){
			  tempab = tempab + A[j][k] * *(tem_y++);
		  }
		  tem_y = tem_y_const;
		  tem_y1[j] = tem_y1[j] - (tempab-*(b++) + *(temp_lambda++))/(double)Lip;

	  }
	  b = b_const;
	  temp_lambda = temp_lambda_const;
	  for(j=0;j<colDim;j++){
		*(tem_y++) = *(tem_y1++);
		*(xPrev++) = *(c++);
	  }
	  tem_y = tem_y_const;
	  tem_y1 = tem_y1_const;
	  xPrev = xPrev_const;
	  c = c_const;
	  for(j=0;j<=nT-1;j++){
		  *(c++) = max_num(*(tem_y++),0.0);
	  }
	  c = c_const;
	  tem_y = tem_y_const;
	  tem_y[colDim-1] = max_num(tem_y[colDim-1],0.0);
	  for(j=nT;j<=colDim-2;j++){
		  c[j] = max_num(tem_y[j]-lambdaLip,0.0)-max_num(-tem_y[j]-lambdaLip,0.0);
		  }
	  tPrev = t;
	  t = (1.0+sqrt(1.0+4.0*t*t)) / 2.0;
  }

  /* free memory*/
//  free(x);
  free(xPrev);
  free(tem_y);
  free(tem_y1);
  free(temp_lambda);
  free_Carrayptrs(A);


  return Py_BuildValue("N",PyArray_Return(out_c_py));

}

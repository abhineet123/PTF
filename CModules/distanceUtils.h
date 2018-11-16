#include <Python.h>
//#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h> 

static PyObject* initStateVars(PyObject* self, PyObject* args);
static PyObject* freeStateVars(PyObject* self, PyObject* args);
static PyObject* isInited(PyObject* self, PyObject* args);

static PyObject* getHistogramsFloor(PyObject* self, PyObject* args);
static PyObject* getHistogramsRound(PyObject* self, PyObject* args);
static PyObject* getfHistograms(PyObject* self, PyObject* args);
static PyObject* getBSplineHistograms(PyObject* self, PyObject* args);

static PyObject* getMIPoints(PyObject* self, PyObject* args);
static PyObject* getMIPoints2(PyObject* self, PyObject* args);
static PyObject* getMIPointsOld(PyObject* self, PyObject* args);
static PyObject* getBSplineMIPoints(PyObject* self, PyObject* args);

static PyObject* getMIMat(PyObject* self, PyObject* args);
static PyObject* getMIMatSSDPoints(PyObject* self, PyObject* args);
static PyObject* getBSplineMIMatSSDPoints(PyObject* self, PyObject* args);


static PyObject* getSCVPoints(PyObject* self, PyObject* args);
static PyObject* getSCVPoints2(PyObject* self, PyObject* args);

static PyObject* getCCREPoints(PyObject* self, PyObject* args);

static PyObject* getHistSSDPoints(PyObject* self, PyObject* args);
static PyObject* getJointHistTracePoints(PyObject* self, PyObject* args);

#ifdef ENABLE_CV
static PyObject* getCorrelationVariancePoints(PyObject* self, PyObject* args);
#endif

static PyObject* getBSplineFKLDPoints(PyObject* self, PyObject* args);
static PyObject* getBSplineIKLDPoints(PyObject* self, PyObject* args);
static PyObject* getBSplineMKLDPoints(PyObject* self, PyObject* args);

static PyObject* getBSplineCHISPoints(PyObject* self, PyObject* args);


static unsigned int  n_pix;
static unsigned int  n_bins;
static unsigned int  img_width;
static unsigned int  img_height;

static int initialized = 0;
static float log_n_pix;


static PyArrayObject *hist1_py, *hist2_py, *hist12_py;
static unsigned int* hist12_data;
static unsigned int* hist1_data;
static unsigned int* hist2_data;

static PyArrayObject *fhist1_py, *fhist2_py, *fhist12_py;
static double* fhist12_data;
static double* fhist1_data;
static double* fhist2_data;

static PyArrayObject *mi_mat_py;
static double* mi_mat_data;

static unsigned int** linear_idx;

float** hist12_norm;
float* hist1_norm;
float* hist2_norm;

double** bspline_mat1;
double** bspline_mat2;

int **bspline_ids;
int **std_bspl_ids;
int *bspline_id_count;


//static unsigned int** hist12;
//static unsigned int* hist1, *hist2;

static unsigned int** hist12_cum;
static unsigned int* hist1_cum;

//static PyArrayObject *hist_frac1, *hist_frac2, *hist_frac12;
//static  double* hist_frac12_data;
//static  double* hist_frac1_data;
//static  double* hist_frac2_data;

static PyArrayObject *img_scv;
double *img_scv_data;


PyArrayObject *img1, *img2;
double *img1_data;
double *img2_data;

static inline void updateHistograms(){

	//fprintf(stdout, "Starting updateHistograms\n");

	memset(hist12_data, 0, sizeof(unsigned int)*n_bins*n_bins);
	memset(hist1_data, 0, sizeof(unsigned int)*n_bins);
	memset(hist2_data, 0, sizeof(unsigned int)*n_bins);

	//fprintf(stdout, "hist12->strides=%d, %d\n", (int)hist12->strides[0], (int)hist12->strides[1]);
	//fprintf(stdout, "hist1->strides=%d\n", (int)hist1->strides[0]);
	//fprintf(stdout, "hist2->strides=%d\n", (int)hist2->strides[0]);


	int i;
	for(i = 0; i < n_pix; i++) {
		//unsigned int img1_val=(int)rint(img1_data[i]);
		//unsigned int img2_val=(int)rint(img2_data[i]);

		unsigned int img1_val = (int)img1_data[i];
		unsigned int img2_val = (int)img2_data[i];

		//int hist12_id=img1_val*(int)hist12->strides[0]+img2_val*(int)hist12->strides[1];
		//int hist1_id=img1_val*(int)hist1->strides[0];
		//int hist2_id=img2_val*(int)hist2->strides[0];
		//fprintf(stdout, "i=%3d, img1_val=%3d, img2_val=%3d\n", i, img1_val,img2_val);
		//fprintf(stdout, "hist12_id=%3d, hist1_id=%3d, hist2_id=%3d\n", hist12_id, hist1_id,hist2_id);


		hist12_data[linear_idx[img1_val][img2_val]]++;
		hist1_data[img1_val]++;
		hist2_data[img2_val]++;
	}
}

static inline void updateHistograms2(){

	memset(hist12_data, 0, sizeof(unsigned int)*n_bins*n_bins);
	memset(hist1_data, 0, sizeof(unsigned int)*n_bins);
	memset(hist2_data, 0, sizeof(unsigned int)*n_bins);
	int i;
	for(i = 0; i < n_pix; i++) {
		unsigned int img1_val = (int)rint(img1_data[i]);
		unsigned int img2_val = (int)rint(img2_data[i]);
		hist12_data[linear_idx[img1_val][img2_val]]++;
		hist1_data[img1_val]++;
		hist2_data[img2_val]++;
	}
}

static inline void updateNormalizedHistograms(){
	int r, t;
	for(r = 0; r < n_bins; r++) {
		hist1_norm[r] = (float)hist1_data[r] / (float)n_pix;
		hist2_norm[r] = (float)hist2_data[r] / (float)n_pix;

		for(t = 0; t < n_bins; t++) {
			hist12_norm[r][t] = (float)hist12_data[linear_idx[r][t]] / (float)n_pix;
		}
	}

}

static inline void updatefHistograms(){

	memset(fhist12_data, 0, sizeof(double)*n_bins*n_bins);
	memset(fhist1_data, 0, sizeof(double)*n_bins);
	memset(fhist2_data, 0, sizeof(double)*n_bins);
	int i;
	for(i = 0; i < n_pix; i++) {
		unsigned int img1_floor = floor(img1_data[i]);
		unsigned int img2_floor = floor(img2_data[i]);
		unsigned int img1_ceil = ceil(img1_data[i]);
		unsigned int img2_ceil = ceil(img2_data[i]);
		double img1_frac = img1_data[i] - img1_floor;
		double img2_frac = img2_data[i] - img2_floor;
		double img1_frac_inv = 1 - img1_frac;
		double img2_frac_inv = 1 - img2_frac;
		fhist12_data[linear_idx[img1_floor][img2_floor]] += img1_frac_inv*img2_frac_inv;
		fhist12_data[linear_idx[img1_ceil][img2_floor]] += img1_frac*img2_frac_inv;
		fhist12_data[linear_idx[img1_floor][img2_ceil]] += img1_frac_inv*img2_frac;
		fhist12_data[linear_idx[img1_ceil][img2_ceil]] += img1_frac*img2_frac;

		fhist1_data[img1_floor] += img1_frac_inv;
		fhist2_data[img2_floor] += img2_frac_inv;
		fhist1_data[img1_ceil] += img1_frac;
		fhist2_data[img2_ceil] += img2_frac;
	}
}
static inline void updatefHistograms2(){

	memset(fhist12_data, 0, sizeof(double)*n_bins*n_bins);
	memset(fhist1_data, 0, sizeof(double)*n_bins);
	memset(fhist2_data, 0, sizeof(double)*n_bins);
	int i;
	for(i = 0; i < n_pix; i++) {
		unsigned int img1_floor = floor(img1_data[i]);
		unsigned int img2_floor = floor(img2_data[i]);
		double img1_frac = img1_data[i] - img1_floor;
		double img2_frac = img2_data[i] - img2_floor;
		double img1_frac_inv = 1 - img1_frac;
		double img2_frac_inv = 1 - img2_frac;

		fhist12_data[linear_idx[img1_floor][img2_floor]] += img1_frac_inv*img2_frac_inv;
		fhist1_data[img1_floor] += img1_frac_inv;
		fhist2_data[img2_floor] += img2_frac_inv;

		if(img1_frac){
			fhist12_data[linear_idx[img1_floor + 1][img2_floor]] += img1_frac*img2_frac_inv;
			fhist1_data[img1_floor + 1] += img1_frac;

			if(img2_frac)
				fhist12_data[linear_idx[img1_floor + 1][img2_floor + 1]] += img1_frac*img2_frac;
		}
		if(img2_frac){
			fhist12_data[linear_idx[img1_floor][img2_floor + 1]] += img1_frac_inv*img2_frac;
			fhist2_data[img2_floor + 1] += img2_frac;
		}
	}
}

static inline double bSpline(double x){
	//fprintf(stdout, "x=%f\t", x);

	if((x > -2) && (x <= -1)){
		double temp = 2 + x;
		return (temp*temp*temp) / 6;
	} else if((x > -1) && (x <= 0)){
		double temp = x*x;
		return (4 - 6 * temp - 3 * temp*x) / 6;
	} else if((x > 0) && (x <= 1)){
		double temp = x*x;
		return (4 - 6 * temp + 3 * temp*x) / 6;
	} else if((x > 1) && (x < 2)){
		double temp = 2 - x;;
		return (temp*temp*temp) / 6;
	}
	return 0;
}

static inline void getNormalizedBSplineHistograms(){

	memset(fhist1_data, 0, sizeof(double)*n_bins);
	memset(fhist2_data, 0, sizeof(double)*n_bins);

	int i, j;
	int pix;
	for(pix = 0; pix < n_pix; pix++) {
		//for(i = 0; i < n_bins; i++) {
		//	fhist1_data[i] += bSpline((double)i - img1_data[pix]);
		//	fhist2_data[i] += bSpline((double)i - img2_data[pix]);
		//}

		int img1_floor = (int)img1_data[pix];
		int img2_floor = (int)img2_data[pix];
		//printf("img1_floor: %d\n", img1_floor);
		//printf("img2_floor: %d\n", img2_floor);
		int id1, id2;
		for(id1 = std_bspl_ids[img1_floor][0]; id1 <= std_bspl_ids[img1_floor][1]; id1++) {
			fhist1_data[id1] += bSpline((double)id1 - img1_data[pix]);
		}
		for(id2 = std_bspl_ids[img2_floor][0]; id2 <= std_bspl_ids[img2_floor][1]; id2++) {
			fhist2_data[id2] += bSpline((double)id2 - img2_data[pix]);
		}
	}

	//printf("std_bspl_ids:\n");
	//for(i = 0; i < n_bins; i++) {
	//	printf("%d\t%d\n", std_bspl_ids[i][0], std_bspl_ids[i][1]);
	//}
	//printf("\n");

	//printf("Histogram 1:\n");
	//for(i = 0; i < n_bins; i++) {
	//	printf("%15.9f\t", fhist1_data[i]);
	//}
	//printf("\n");

	//printf("Histogram 2:\n");
	//for(i = 0; i < n_bins; i++) {
	//	printf("%15.9f\t", fhist2_data[i]);
	//}
	//printf("\n");

	for(i = 0; i < n_bins; i++){
		fhist1_data[i] /= n_pix;
		fhist2_data[i] /= n_pix;
	}

	//int r, t;
	//printf("Normalized Histogram 1:\n");
	//for(r = 0; r < n_bins; r++) {
	//	printf("%15.9f\t", fhist1_data[r]);
	//}
	//printf("\n");

	//printf("Normalized Histogram 2:\n");
	//for(r = 0; r < n_bins; r++) {
	//	printf("%15.9f\t", fhist2_data[r]);
	//}
	//printf("\n");


}

static inline void updateBSplineHistograms(){

	int bin, pix;
	//for (bin=0; bin<n_bins; bin++){
	//	for(i=0;i<bspline_id_count[img1_floor];i++) {
	//		int id1=bspline_ids[img1_floor][i];
	//	for(pix=0; pix<n_pix; pix++) {
	//		bspline_mat1[bin][pix]=bSpline(bin-img1_data[pix]);
	//		bspline_mat2[bin][pix]=bSpline(bin-img2_data[pix]);
	//	}
	//}
	//fprintf(stdout, "updateBSplineHistograms:: n_pix=%d n_bins=%d\n", n_pix, n_bins);
	memset(fhist12_data, 0, sizeof(double)*n_bins*n_bins);
	memset(fhist1_data, 0, sizeof(double)*n_bins);
	memset(fhist2_data, 0, sizeof(double)*n_bins);

	for(pix = 0; pix < n_pix; pix++) {
		int img1_floor = (int)img1_data[pix];
		int img2_floor = (int)img2_data[pix];

		int i, j;
		//fprintf(stdout, "img2_floor=%d\n", img2_floor);
		for(j = 0; j < bspline_id_count[img2_floor]; j++) {
			int id2 = bspline_ids[img2_floor][j];
			//fprintf(stdout, "id2=%d img2_data[pix]=%f\t", id2, img2_data[pix]);
			bspline_mat2[id2][pix] = bSpline((double)id2 - img2_data[pix]);
			//fprintf(stdout, "b_spline_val=%f\n", bspline_mat2[id2][pix]);
		}
		//fprintf(stdout, "img1_floor=%d\n", img1_floor);
		for(i = 0; i < bspline_id_count[img1_floor]; i++) {
			int id1 = bspline_ids[img1_floor][i];
			//fprintf(stdout, "id1=%d img1_data[pix]=%f\t", id1, img1_data[pix]);
			bspline_mat1[id1][pix] = bSpline((double)id1 - img1_data[pix]);
			//fprintf(stdout, "b_spline_val=%f\n", bspline_mat1[id1][pix]);
			for(j = 0; j < bspline_id_count[img2_floor]; j++) {
				int id2 = bspline_ids[img2_floor][j];
				fhist12_data[linear_idx[id1][id2]] += bspline_mat1[id1][pix] * bspline_mat2[id2][pix];
			}
		}
	}
	int i, j;
	for(i = 0; i < n_bins; i++){
		for(j = 0; j < n_bins; j++){
			fhist1_data[i] += fhist12_data[linear_idx[i][j]];
			fhist2_data[i] += fhist12_data[linear_idx[j][i]];
		}
	}
}








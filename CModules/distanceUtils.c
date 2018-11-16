#include "distanceUtils.h"
#ifdef ENABLE_CV
#include "DSST/CorrelationVariance.h"
#endif

/* ==== Set up the methods table ====================== */
static PyMethodDef distanceUtilsMethods[] = {
	{"initStateVars", initStateVars, METH_VARARGS},
	{"isInited", isInited, METH_VARARGS},
	{"freeStateVars", freeStateVars, METH_VARARGS},

	{"getHistogramsFloor", getHistogramsFloor, METH_VARARGS},
	{"getHistogramsRound", getHistogramsRound, METH_VARARGS},
	{"getfHistograms", getfHistograms, METH_VARARGS},
	{"getBSplineHistograms", getBSplineHistograms, METH_VARARGS},

	{"getMIPoints", getMIPoints, METH_VARARGS},
	{"getMIPoints2", getMIPoints2, METH_VARARGS},
	{"getMIPointsOld", getMIPointsOld, METH_VARARGS},
	{"getBSplineMIPoints", getBSplineMIPoints, METH_VARARGS},


	{"getMIMat", getMIMat, METH_VARARGS},
	{"getMIMatSSDPoints", getMIMatSSDPoints, METH_VARARGS},
	{"getBSplineMIMatSSDPoints", getBSplineMIMatSSDPoints, METH_VARARGS},


	{"getHistSSDPoints", getHistSSDPoints, METH_VARARGS},
	{"getJointHistTracePoints", getJointHistTracePoints, METH_VARARGS},

	{"getCCREPoints", getCCREPoints, METH_VARARGS},

	{"getSCVPoints", getSCVPoints, METH_VARARGS},
	{"getSCVPoints2", getSCVPoints2, METH_VARARGS},
#ifdef ENABLE_CV
	{ "getCorrelationVariancePoints", getCorrelationVariancePoints, METH_VARARGS },
#endif

	{ "getBSplineFKLDPoints", getBSplineFKLDPoints, METH_VARARGS },
	{ "getBSplineIKLDPoints", getBSplineIKLDPoints, METH_VARARGS },
	{ "getBSplineMKLDPoints", getBSplineMKLDPoints, METH_VARARGS },

	{ "getBSplineCHISPoints", getBSplineCHISPoints, METH_VARARGS },

	{NULL, NULL}     /* Sentinel - marks the end of this structure */
};

/* ==== Initialize the C_test functions ====================== */
PyMODINIT_FUNC initdistanceUtils()  {
	(void) Py_InitModule("distanceUtils", distanceUtilsMethods);
	import_array();  // Must be present for NumPy.  Called first after above line.
}

static PyObject* isInited(PyObject* self, PyObject* args) {
	return Py_BuildValue("i", initialized);
}


static PyObject* initStateVars(PyObject* self, PyObject* args) {

	//fprintf(stdout, "Starting initStateVars\n");

	if(!PyArg_ParseTuple(args, "iii", &img_width, &img_height, &n_bins)){
		//fprintf(stdout, "\n----initStateVars: Input arguments could not be parsed----\n\n");
		return NULL;
	}
	n_pix = img_width * img_height;
	//fprintf(stdout, "initStateVars:: n_pix=%d\n", n_pix);
	//fprintf(stdout, "initStateVars:: n_bins=%d\n", n_bins);

	log_n_pix=log(n_pix);

	int dims[2];
	dims[0]=n_bins;
	dims[1]=n_bins;

	hist12_py=(PyArrayObject *) PyArray_FromDims(2,dims,NPY_UINT32);
	hist1_py=(PyArrayObject *) PyArray_FromDims(1,dims,NPY_UINT32);
	hist2_py=(PyArrayObject *) PyArray_FromDims(1,dims,NPY_UINT32);
	hist12_data=(unsigned int*)hist12_py->data;
	hist1_data=(unsigned int*)hist1_py->data;
	hist2_data=(unsigned int*)hist2_py->data;

	fhist12_py=(PyArrayObject *) PyArray_FromDims(2,dims,NPY_FLOAT64);
	fhist1_py=(PyArrayObject *) PyArray_FromDims(1,dims,NPY_FLOAT64);
	fhist2_py=(PyArrayObject *) PyArray_FromDims(1,dims,NPY_FLOAT64);
	fhist12_data=(double*)fhist12_py->data;
	fhist1_data=(double*)fhist1_py->data;
	fhist2_data=(double*)fhist2_py->data;

	mi_mat_py=(PyArrayObject *) PyArray_FromDims(2,dims,NPY_FLOAT64);
	mi_mat_data=(double*)mi_mat_py->data;

	hist12_norm=(float**)malloc(n_bins*sizeof(float*));
	hist1_norm=(float*)malloc(n_bins*sizeof(float)); 
	hist2_norm=(float*)malloc(n_bins*sizeof(float)); 

	hist12_cum=(unsigned int**)malloc(n_bins*sizeof(unsigned int*));
	hist1_cum=(unsigned int*)malloc(n_bins*sizeof(unsigned int)); 
	//hist12=(unsigned int**)malloc(n_bins*sizeof(unsigned int*));

	linear_idx=(unsigned int**)malloc(n_bins*sizeof(unsigned int*));

	bspline_mat1=(double**)malloc(n_bins*sizeof(double*));
	bspline_mat2=(double**)malloc(n_bins*sizeof(double*));
	bspline_ids=(int**)malloc(n_bins*sizeof(int*));
	std_bspl_ids = (int**)malloc(n_bins*sizeof(int*));
	bspline_id_count = (int*)malloc(n_bins*sizeof(int));

	int i, j;
	for(i=0; i<n_bins; i++) {
		hist12_norm[i]=(float*)malloc(n_bins*sizeof(float)); 
		linear_idx[i]=(unsigned int*)malloc(n_bins*sizeof(unsigned int)); 
		hist12_cum[i]=(unsigned int*)malloc(n_bins*sizeof(unsigned int)); 
		for(j=0; j<n_bins; j++) {
			linear_idx[i][j]=i*n_bins+j;
		}
		bspline_mat1[i]=(double*)malloc(n_pix*sizeof(double)); 
		bspline_mat2[i]=(double*)malloc(n_pix*sizeof(double)); 
		bspline_ids[i]=(int*)malloc(n_bins*sizeof(int)); 
		std_bspl_ids[i] = (int*)malloc(2*sizeof(int));
	}

	for(i=0; i<n_bins; i++) {

		std_bspl_ids[i][0] = i > 0 ? i - 1 : 0;
		std_bspl_ids[i][1] = i + 2 < n_bins ? i + 2 : n_bins-1;

		int id_count=0;
		for(j=-1; j<=2; j++) {
			if (i+j<n_bins)
				bspline_ids[i][id_count++]=i+j;
		}
		bspline_id_count[i]=id_count;
	}

	//hist1=(unsigned int*)malloc(n_bins*sizeof(unsigned int)); 
	//hist2=(unsigned int*)malloc(n_bins*sizeof(unsigned int)); 

	dims[0]=n_pix;

	img_scv=(PyArrayObject *) PyArray_FromDims(1,dims,NPY_FLOAT64);	
	img_scv_data=(double*)img_scv->data;

	initialized=1;

	return Py_BuildValue("i", 1);
}

static PyObject* getHistogramsFloor(PyObject* self, PyObject* args) {

	////fprintf(stdout, "Starting getHistograms\n");

	if (!PyArg_ParseTuple(args, "O!O!",  &PyArray_Type, &img1,  &PyArray_Type, &img2))
		return NULL;

	img1_data=(double*)img1->data;
	img2_data=(double*)img2->data;

	updateHistograms();

	return Py_BuildValue("OOO", hist12_py, hist1_py, hist2_py);
}

static PyObject* getHistogramsRound(PyObject* self, PyObject* args) {

	////fprintf(stdout, "Starting getHistograms\n");

	if (!PyArg_ParseTuple(args, "O!O!",  &PyArray_Type, &img1,  &PyArray_Type, &img2))
		return NULL;

	img1_data=(double*)img1->data;
	img2_data=(double*)img2->data;

	updateHistograms2();

	return Py_BuildValue("OOO", hist12_py, hist1_py, hist2_py);
}

static PyObject* getfHistograms(PyObject* self, PyObject* args) {

	////fprintf(stdout, "Starting getHistograms\n");

	if (!PyArg_ParseTuple(args, "O!O!",  &PyArray_Type, &img1,  &PyArray_Type, &img2))
		return NULL;

	img1_data=(double*)img1->data;
	img2_data=(double*)img2->data;

	updatefHistograms2();

	return Py_BuildValue("OOO", fhist12_py, fhist1_py, fhist2_py);
}

static PyObject* getBSplineHistograms(PyObject* self, PyObject* args) {

	//fprintf(stdout, "Starting getBSplineHistograms\n");

	if (!PyArg_ParseTuple(args, "O!O!",  &PyArray_Type, &img1,  &PyArray_Type, &img2)){
		//fprintf(stdout, "getBSplineHistograms:: input args not parsed successfully\n");
		return NULL;
	}

	img1_data=(double*)img1->data;
	img2_data=(double*)img2->data;

	//fprintf(stdout, "getBSplineHistograms:: n_pix=%d n_bins=%d\n", n_pix, n_bins);

	updateBSplineHistograms();

	return Py_BuildValue("OOO", fhist12_py, fhist1_py, fhist2_py);
}


static PyObject* getMIPointsOld(PyObject* self, PyObject* args) {

	if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &img1, &PyArray_Type, &img2))
		return NULL;

	img1_data=(double*)img1->data;
	img2_data=(double*)img2->data;

	updateHistograms();

	int r,t;
	for(r=0; r<n_bins; r++) {	
		hist1_norm[r]=(float)hist1_data[r]/(float)n_pix;
		hist2_norm[r]=(float)hist2_data[r]/(float)n_pix;

		for(t=0; t<n_bins; t++) {	
			hist12_norm[r][t]=(float)hist12_data[linear_idx[r][t]]/(float)n_pix;
		}
	}

	float mi=0;
	for(r=0; r<n_bins; r++) {
		if(!hist1_norm[r])
			continue;
		for(t=0; t<n_bins; t++) {	
			if(!hist2_norm[t] || !hist12_norm[r][t])
				continue;
			mi+=hist12_norm[r][t]*log(hist12_norm[r][t]/(hist1_norm[r]*hist2_norm[t]));
		}
	}
	return Py_BuildValue("f", mi);
}

static PyObject* getMIPoints2(PyObject* self, PyObject* args) {

	if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &img1, &PyArray_Type, &img2))
		return NULL;

	img1_data=(double*)img1->data;
	img2_data=(double*)img2->data;

	updateHistograms2();

	int r,t;
	for(r=0; r<n_bins; r++) {	
		hist1_norm[r]=(float)hist1_data[r]/(float)n_pix;
		hist2_norm[r]=(float)hist2_data[r]/(float)n_pix;

		for(t=0; t<n_bins; t++) {	
			hist12_norm[r][t]=(float)hist12_data[linear_idx[r][t]]/(float)n_pix;
		}
	}

	float mi=0;
	for(r=0; r<n_bins; r++) {
		if(!hist1_norm[r])
			continue;
		for(t=0; t<n_bins; t++) {	
			if(!hist2_norm[t] || !hist12_norm[r][t])
				continue;
			mi+=hist12_norm[r][t]*log(hist12_norm[r][t]/(hist1_norm[r]*hist2_norm[t]));
		}
	}
	return Py_BuildValue("f", mi);
}

static PyObject* getMIPoints(PyObject* self, PyObject* args) {

	if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &img1, &PyArray_Type, &img2))
		return NULL;

	img1_data=(double*)img1->data;
	img2_data=(double*)img2->data;

	updateHistograms2();

	int r,t;
	float mi=0;
	for(r=0; r<n_bins; r++) {
		float hist1_val=hist1_data[r];
		if(!hist1_val){
			continue;
		}
		for(t=0; t<n_bins; t++) {
			float hist2_val=hist2_data[t];
			float hist12_val=hist12_data[linear_idx[r][t]];

			////fprintf(stdout, "\tt=%4d\t hist2=%4d\t hist12=%4d\n", t, hist2[t], hist12[hist_id]);
			if(!hist2_val || !hist12_val)
				continue;
			mi+=hist12_val*(log(hist12_val/(hist1_val*hist2_val))+log_n_pix);

			////fprintf(stdout, "\thist_id=%4d\tmi=%12.9f\n", hist_id, mi);
		}
	}
	return Py_BuildValue("f", mi);
}
static PyObject* getBSplineMIPoints(PyObject* self, PyObject* args) {

	if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &img1, &PyArray_Type, &img2))
		return NULL;

	img1_data=(double*)img1->data;
	img2_data=(double*)img2->data;

	updateBSplineHistograms();

	int r,t;
	double bspline_mi=0;
	for(r=0; r<n_bins; r++) {
		double fhist1_val=fhist1_data[r];
		if(!fhist1_val){
			continue;
		}
		for(t=0; t<n_bins; t++) {
			double fhist2_val=fhist2_data[t];
			double fhist12_val=fhist12_data[linear_idx[r][t]];

			////fprintf(stdout, "\tt=%4d\t fhist2=%4d\t fhist12=%4d\n", t, fhist2[t], fhist12[fhist_id]);
			if(!fhist2_val || !fhist12_val)
				continue;
			bspline_mi+=fhist12_val*log(fhist12_val/(fhist1_val*fhist2_val));

			////fprintf(stdout, "\tfhist_id=%4d\tmi=%12.9f\n", fhist_id, mi);
		}
	}
	return Py_BuildValue("d", bspline_mi);
}

static PyObject* getMIMat(PyObject* self, PyObject* args) {

	if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &img1, &PyArray_Type, &img2))
		return NULL;

	img1_data=(double*)img1->data;
	img2_data=(double*)img2->data;

	updateHistograms2();

	memset(mi_mat_data, 0.0, sizeof(double)*n_bins*n_bins);

	int r,t;
	for(r=0; r<n_bins; r++) {
		float hist1_val=hist1_data[r];
		if(!hist1_val){
			continue;
		}
		for(t=0; t<n_bins; t++) {
			float hist2_val=hist2_data[t];
			float hist12_val=hist12_data[linear_idx[r][t]];
			if(!hist2_val || !hist12_val)
				continue;
			mi_mat_data[linear_idx[r][t]]=hist12_val*(log(hist12_val/(hist1_val*hist2_val))+log_n_pix)/n_pix;
		}
	}
	return Py_BuildValue("O", mi_mat_py);
}

static PyObject* getMIMatSSDPoints(PyObject* self, PyObject* args) {

	if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &img1, &PyArray_Type, &img2))
		return NULL;

	img1_data=(double*)img1->data;
	img2_data=(double*)img2->data;

	updateHistograms2();

	int r,t;
	float mi_mat_ssd=0;
	float norm_factor=n_pix;
	for(r=0; r<n_bins; r++) {
		float hist1_val=(float)hist1_data[r]/norm_factor;
		if(!hist1_val){
			continue;
		}
		for(t=0; t<n_bins; t++) {
			float hist2_val=(float)hist2_data[t]/norm_factor;
			float hist12_val=(float)hist12_data[linear_idx[r][t]]/norm_factor;
			if(!hist2_val || !hist12_val)
				continue;
			float diff=hist12_val*(log(hist12_val/(hist1_val*hist2_val)));
			if (r==t){
				float entropy=hist2_val*log(1/hist2_val);
				diff=entropy-diff;
			}
			mi_mat_ssd+=diff*diff;
		}
	}
	return Py_BuildValue("f", mi_mat_ssd);
}

static PyObject* getBSplineMIMatSSDPoints(PyObject* self, PyObject* args) {

	if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &img1, &PyArray_Type, &img2)){
		//fprintf(stdout, "getBSplineMIMatSSDPoints:: input arguments could not be parsed\n");
		return NULL;
	}

	img1_data=(double*)img1->data;
	img2_data=(double*)img2->data;

	updateBSplineHistograms();

	int r,t;
	float bmi_mat_ssd=0;
	for(r=0; r<n_bins; r++) {
		float fhist1_val=fhist1_data[r];
		if(!fhist1_val){
			continue;
		}
		for(t=0; t<n_bins; t++) {
			float fhist2_val=fhist2_data[t];
			float fhist12_val=fhist12_data[linear_idx[r][t]];
			if(!fhist2_val || !fhist12_val)
				continue;
			float diff=fhist12_val*log(fhist12_val/(fhist1_val*fhist2_val));
			if (r==t){
				float entropy=fhist2_val*log(1/fhist2_val);
				diff=entropy-diff;
			}
			bmi_mat_ssd+=diff*diff;
		}
	}
	return Py_BuildValue("f", bmi_mat_ssd);
}

static PyObject* getHistSSDPoints(PyObject* self, PyObject* args){

	if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &img1, &PyArray_Type, &img2))
		return NULL;

	img1_data=(double*)img1->data;
	img2_data=(double*)img2->data;

	updateHistograms2();
	int r;
	float hist_ssd=0;
	for(r=0; r<n_bins; r++) {
		float diff=((float)hist1_data[r]-(float)hist2_data[r])/(float)n_pix;
		////fprintf(stdout, "\thist1: %d\t hist2: %d\t diff: %f\n", hist1_data[r], hist2_data[r], diff);
		hist_ssd+=diff*diff;
	}
	////fprintf(stdout, "hist_ssd: %f\n", hist_ssd);
	return Py_BuildValue("f", hist_ssd);
}

static PyObject* getJointHistTracePoints(PyObject* self, PyObject* args){

	if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &img1, &PyArray_Type, &img2))
		return NULL;

	img1_data=(double*)img1->data;
	img2_data=(double*)img2->data;

	updateHistograms2();
	int r;
	float h12_trace=0;
	for(r=0; r<n_bins; r++) {
		h12_trace+=hist12_data[linear_idx[r][r]];
		////fprintf(stdout, "hist_ssd: %f\n", hist_ssd);
	}
	h12_trace/=n_pix;	
	return Py_BuildValue("f", h12_trace);
}

static PyObject* getCCREPoints(PyObject* self, PyObject* args) {

	if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &img1, &PyArray_Type, &img2))
		return NULL;

	img1_data=(double*)img1->data;
	img2_data=(double*)img2->data;

	updateHistograms2();

	int r,t;
	//hist1_cum[0]=hist1_data[0];
	//for(t=0; t<n_bins; t++) {
	//	if(t>0){
	//		hist1_cum[t]=hist1_cum[t-1]+hist1_data[t];
	//		////fprintf(stdout, "t=%d\t hist1_cum=%4d\t hist2=%4d\n", t, hist1_cum[t], hist2_data[t]);
	//	}
	//	hist12_cum[0][t]=hist12_data[t];
	//	////fprintf(stdout, "t=%4d r=%4d\t hist12_cum=%4d\n", t, 0, hist12_cum[0][t]);
	//	for(r=1; r<n_bins; r++) {	
	//		hist12_cum[r][t]=hist12_cum[r-1][t]+hist12_data[linear_idx[r][t]];	
	//		////fprintf(stdout, "t=%4d r=%4d\t hist12_cum=%4d\n", t, r, hist12_cum[r][t]);
	//	}
	//}
	hist1_cum[n_bins-1]=hist1_data[n_bins-1];
	for(t=n_bins-1; t>=0; t--) {
		if(t<n_bins-1){
			hist1_cum[t]=hist1_cum[t+1]+hist1_data[t];
			////fprintf(stdout, "t=%d\t hist1_cum=%4d\t hist2=%4d\n", t, hist1_cum[t], hist2_data[t]);
		}
		hist12_cum[n_bins-1][t]=hist12_data[linear_idx[n_bins-1][t]];
		////fprintf(stdout, "t=%4d r=%4d\t hist12_cum=%4d\n", t, 0, hist12_cum[0][t]);
		for(r=n_bins-2; r>=0; r--) {	
			hist12_cum[r][t]=hist12_cum[r+1][t]+hist12_data[linear_idx[r][t]];	
			////fprintf(stdout, "t=%4d r=%4d\t hist12_cum=%4d\n", t, r, hist12_cum[r][t]);
		}
	}
	float ccre=0;
	for(r=0; r<n_bins; r++) {
		float hist1_val=hist1_cum[r];
		//fprintf(stdout, "r=%4d\t hist1_val=%12.6f\n", r, hist1_val);

		if(!hist1_val){
			continue;
		}
		for(t=0; t<n_bins; t++) {
			float hist2_val=hist2_data[t];
			float hist12_val=hist12_cum[r][t];

			//fprintf(stdout, "\tt=%4d\t hist2_val=%12.6f\t hist12_val=%12.6f\n", t, hist2_val, hist12_val);
			if(!hist2_val || !hist12_val)
				continue;

			ccre+=hist12_val*(log(hist12_val/(hist1_val*hist2_val)+log_n_pix));
			//fprintf(stdout, "\t\tccre=%12.6f\n", ccre);
		}
	}
	return Py_BuildValue("f", ccre);
}

static PyObject* getSCVPoints2(PyObject* self, PyObject* args) {

	if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &img1, &PyArray_Type, &img2))
		return NULL;

	img1_data=(double*)img1->data;
	img2_data=(double*)img2->data;

	updateHistograms();
	int x, i;
	float scv=0;
	for(x=0; x<n_pix; x++) {
		int img2_val=(int)img2_data[x];
		float norm_fac=hist2_data[img2_val];
		if (!norm_fac){
			img_scv_data[x]=img2_val;
			continue;
		}else{
			int scv_num=0;
			for(i=0; i<n_bins; i++) {
				scv_num+=i*hist12_data[linear_idx[i][img2_val]];
			}
			img_scv_data[x]=scv_num/norm_fac;
		}
		float diff=img_scv_data[x]-img1_data[x];
		scv+=diff*diff;
	}
	return Py_BuildValue("f", scv);
}

static PyObject* getSCVPoints(PyObject* self, PyObject* args) {

	if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &img1, &PyArray_Type, &img2))
		return NULL;

	img1_data=(double*)img1->data;
	img2_data=(double*)img2->data;

	updateHistograms2();

	int x, i;
	float scv=0;
	for(x=0; x<n_pix; x++) {
		int img2_val=(int)rint(img2_data[x]);
		float norm_fac=hist2_data[img2_val];
		if (!norm_fac){
			img_scv_data[x]=img2_val;
			continue;
		}else{
			int scv_num=0;
			for(i=0; i<n_bins; i++) {
				scv_num+=i*hist12_data[linear_idx[i][img2_val]];
			}
			img_scv_data[x]=scv_num/norm_fac;
		}
		float diff=img_scv_data[x]-img1_data[x];
		scv+=diff*diff;
	}
	return Py_BuildValue("f", scv);
}
#ifdef ENABLE_CV
static PyObject* getCorrelationVariancePoints(PyObject* self, PyObject* args) {

	if(!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &img1, &PyArray_Type, &img2))
		return NULL;

	img1_data = (double*)img1->data;
	img2_data = (double*)img2->data;

	Mat initPatch = cv::Mat(img_height, img_width, CV_64FC1, img2_data);
	Mat currPatch = cv::Mat(img_height, img_width, CV_64FC1, img1_data);

	Mat initPatch8 = convertFloatImg(initPatch);
	Mat currPatch8 = convertFloatImg(currPatch);

	imshow("initPatch ", initPatch8);
	imshow("currPatch ", currPatch8);
	int key = waitKey(0);
	if(key == 27){
		exit(0);
	} else if(key == 32){
		return Py_BuildValue("f", 0);
	}

	//Compute Numerator and Denominator of the correlation filter according to initPatch
	Mat den_trans;
	Mat trans_cos_win;
	Size padded;
	Mat *num_trans = trainOnce(initPatch, img_width, img_height, den_trans, padded, trans_cos_win);

	//Compute correlation with the learned correlation filter
	float variance = computeCorrelationVariance(currPatch, padded, num_trans, den_trans, trans_cos_win);
	delete[] num_trans;
	return Py_BuildValue("f", variance);
}
#endif
static PyObject* getBSplineFKLDPoints(PyObject* self, PyObject* args) {

	if(!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &img1, &PyArray_Type, &img2))
		return NULL;

	img1_data = (double*)img1->data;
	img2_data = (double*)img2->data;

	getNormalizedBSplineHistograms();

	//printf("n_pix: %d\n", n_pix);
	//printf("n_bins: %d\n", n_bins);
	int r;
	double bspline_kld = 0;
	for(r = 0; r < n_bins; r++) {
		double fhist1_val = fhist1_data[r];
		double fhist2_val = fhist2_data[r];
		if(!fhist1_val || !fhist2_val)
			continue;
		bspline_kld += fhist2_val*log(fhist2_val / fhist1_val);
	}
	return Py_BuildValue("d", bspline_kld);
}

static PyObject* getBSplineIKLDPoints(PyObject* self, PyObject* args) {

	if(!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &img1, &PyArray_Type, &img2))
		return NULL;

	img1_data = (double*)img1->data;
	img2_data = (double*)img2->data;

	getNormalizedBSplineHistograms();

	//printf("n_pix: %d\n", n_pix);
	//printf("n_bins: %d\n", n_bins);
	int r;
	double bspline_kld = 0;
	for(r = 0; r < n_bins; r++) {
		double fhist1_val = fhist1_data[r];
		double fhist2_val = fhist2_data[r];
		if(!fhist1_val || !fhist2_val)
			continue;
		bspline_kld += fhist1_val*log(fhist1_val / fhist2_val);
	}
	return Py_BuildValue("d", bspline_kld);
}
// mean of forward and inverse KL divergence
static PyObject* getBSplineMKLDPoints(PyObject* self, PyObject* args) {

	if(!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &img1, &PyArray_Type, &img2))
		return NULL;

	img1_data = (double*)img1->data;
	img2_data = (double*)img2->data;

	getNormalizedBSplineHistograms();

	//printf("n_pix: %d\n", n_pix);
	//printf("n_bins: %d\n", n_bins);
	int r;
	double bspline_kld = 0;
	for(r = 0; r < n_bins; r++) {
		double fhist1_val = fhist1_data[r];
		double fhist2_val = fhist2_data[r];
		if(!fhist1_val || !fhist2_val)
			continue;
		bspline_kld += (fhist2_val - fhist1_val)*log(fhist2_val / fhist1_val) / 2.0;
	}
	return Py_BuildValue("d", bspline_kld);
}
// chi square distance between normalized histograms
static PyObject* getBSplineCHISPoints(PyObject* self, PyObject* args) {

	if(!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &img1, &PyArray_Type, &img2))
		return NULL;

	img1_data = (double*)img1->data;
	img2_data = (double*)img2->data;

	getNormalizedBSplineHistograms();

	//printf("n_pix: %d\n", n_pix);
	//printf("n_bins: %d\n", n_bins);
	int r;
	double bspline_chis = 0;
	for(r = 0; r < n_bins; r++) {
		double sum = fhist1_data[r] + fhist2_data[r];
		if(!sum)
			continue;
		double diff = fhist1_data[r] - fhist2_data[r];
		bspline_chis += diff*diff / sum;
	}
/*	printf("bspline_chis: %15.9f\n", bspline_chis);

	Mat initPatch = cv::Mat(img_height, img_width, CV_64FC1, img2_data);
	Mat currPatch = cv::Mat(img_height, img_width, CV_64FC1, img1_data);

	Mat initPatch8 = convertFloatImg(initPatch);
	Mat currPatch8 = convertFloatImg(currPatch);

	imshow("initPatch ", initPatch8);
	imshow("currPatch ", currPatch8);
	int key = waitKey(0);
	if(key == 27){
		exit(0);
	} */	
	return Py_BuildValue("d", bspline_chis);
}

static PyObject* freeStateVars(PyObject* self, PyObject* args) {
	int i;
	for(i=0; i<n_bins; i++) {
		free(linear_idx[i]); 
		free(hist12_cum[i]); 
		free(hist12_norm[i]); 

		free(bspline_mat1[i]); 
		free(bspline_mat2[i]); 
		free(bspline_ids[i]); 

	}
	free(linear_idx);
	free(hist12_cum);
	free(hist12_norm);
	free(hist1_norm);
	free(hist2_norm);

	free(bspline_mat1);
	free(bspline_mat2);
	free(bspline_ids);
	free(bspline_id_count);

	return Py_BuildValue("i", 1);
}


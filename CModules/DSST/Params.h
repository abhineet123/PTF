#ifndef INC_Params_H
#define INC_Params_H

#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;


struct HOGParams
{
	int binSize;
	int nOrients;
	int softBin;
	float clipHog;

	HOGParams(){}
};

struct target
{
	CvRect init;
	int firstFrame;

	target(int x, int y, int w, int h, int firstF)
	{
		init.x= x;
		init.y= y;
		init.width= w;
		init.height= h;
		firstFrame= firstF;
	}
};

struct GT
{
	double tlx_;
	double tly_;
	double trx_;
	double try_;
	double blx_;
	double bly_;
	double brx_;
	double bry_;
};

struct trackingSetup
{
	cv::Mat trans_cos_win;
	cv::Mat scale_cos_win;

	cv::Mat transFourier;
	cv::Mat scaleFourier;

	int nNumTrans;
	cv::Mat *num_trans;
	cv::Mat den_trans;
	int nNumScale;
	cv::Mat *num_scale;
	cv::Mat den_scale;

	double *scaleFactors;
	Size scale_model_sz;

	float min_scale_factor;
	float max_scale_factor;

	float current_scale_factor;

	Point centroid;
	Size original;
	Size padded;
};

struct DSSTParams
{
	double padding;
	double output_sigma_factor;
	double scale_sigma_factor;
	double lambda;
	double learning_rate;
	int number_scales;
	double scale_step;
	int scale_model_max_area;
	int resize_factor;
	int is_scaling;
	int bin_size;

	DSSTParams(){
		padding = 1;
		output_sigma_factor = 1.0 / 16;
		scale_sigma_factor = 1.0 / 4;
		lambda = 1e-2;
		learning_rate = 0.035;//0.025;
		number_scales = 33;
		scale_step = 1.02;
		resize_factor = 1;
		is_scaling = 1;
		bin_size = 1;
		scale_model_max_area = 512;
	}

	DSSTParams(double padding_, double output_sigma_factor_, double scale_sigma_factor_, double lambda_, 
		double learning_rate_, int number_scales_, double scale_step_, int resize_factor_, int is_scaling_, int bin_size_)
	{
		padding = padding_;
		output_sigma_factor= output_sigma_factor_;
		scale_sigma_factor= scale_sigma_factor_;
		lambda= lambda_;
		learning_rate= learning_rate_;
		number_scales= number_scales_;
		scale_step= scale_step_;
		resize_factor= resize_factor_;
		is_scaling= is_scaling_;
		bin_size= bin_size_;
		scale_model_max_area= 512;
	}

	DSSTParams(DSSTParams *params) {
		if(params){
			padding = params->padding;//1;
			output_sigma_factor= params->output_sigma_factor;//1.0/16;
			scale_sigma_factor= params->scale_sigma_factor;//1.0/4;
			lambda= params->lambda;//1e-2;
			learning_rate= params->learning_rate;//0.035;//0.025;
			number_scales= params->number_scales;//33;
			scale_step= params->scale_step;//1.02;
			resize_factor= params->resize_factor;
			is_scaling= params->is_scaling;
			bin_size= params->bin_size;
			cout<<"Setting params : "<<padding<<" "<<output_sigma_factor<<" "<<scale_sigma_factor<<" "<<lambda;
			cout<<" "<<learning_rate<<" "<<number_scales<<" "<<scale_step<<" "<<resize_factor<<" "<<is_scaling<<" "<<bin_size<<endl;
			scale_model_max_area= 512;
		}
		else
		{
			padding = 1;
			output_sigma_factor= 1.0/16;
			scale_sigma_factor= 1.0/4;
			lambda= 1e-2;
			learning_rate= 0.035;//0.025;
			number_scales= 33;
			scale_step= 1.02;
			resize_factor= 1;
			is_scaling= 1;
			bin_size=1;
			scale_model_max_area= 512;
		}
	}

};

#endif

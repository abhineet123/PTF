#include "Params.h"
#include "HOG.h"

DSSTParams tParams;
HOGParams hParams;

Mat hann(int size) {
    cv::Mat arr(size, 1, CV_32FC1);
    float multiplier;
    for(int i = 0; i < size; i++) {
        multiplier = 0.5 * (1 - cos(2 * M_PI * i / (size - 1)));
        *((float *)(arr.data + i * arr.step[0])) = multiplier;
    }
    return arr;
}

Mat convertNormalizedFloatImg(Mat &img) {
    Mat imgU;
    double minVal;
    double maxVal;
    Point minLoc;
    Point maxLoc;
    minMaxLoc(img, &minVal, &maxVal, &minLoc, &maxLoc);

    img -= minVal;
    img.convertTo(imgU, CV_64FC1, 1.0 / (maxVal - minVal));
    return imgU;
}

Mat convertFloatImg(Mat &img) {
	Mat imgU;
	double minVal;
	double maxVal;
	Point minLoc;
	Point maxLoc;
	minMaxLoc(img, &minVal, &maxVal, &minLoc, &maxLoc);

	img -= minVal;
	img.convertTo(imgU, CV_8UC1, 255.0 / (maxVal - minVal));
	return imgU;
}

double *computeMeanVariance(Mat trans_response) {
    double mean = 0;
    for(int i = 0; i < trans_response.rows; i++) {
        for(int j = 0; j < trans_response.cols; j++) {
            mean += trans_response.at<double>(i, j);
        }
    }
    mean = mean / (trans_response.rows * trans_response.cols);

    double variance = 0;
    for(int i = 0; i < trans_response.rows; i++) {
        for(int j = 0; j < trans_response.cols; j++) {
            variance += pow(trans_response.at<double>(i, j) - mean, 2);
        }
    }
    //cout<<"Variance "<<variance<<endl;
    variance = variance / (trans_response.cols * trans_response.rows);
    //cout<<"Variance again "<<variance<<endl;
    double *params = new double[2];
    params[0] = mean;
    params[1] = sqrt(variance);

    //cout<<"Variance last time"<<params[1]<<endl;

    return params;
}

float *convert1DArray(Mat &patch) {
    float *img = (float*)calloc(patch.rows * patch.cols, sizeof(float));

    int k = 0;
    for(int i = 0; i < patch.cols; i++)
        for(int j = 0; j < patch.rows; j++) {
            img[k] = (float)patch.at<float>(j, i);
            k++;
        }
    return img;
}

Mat inverseFourier(cv::Mat original, int flag = 0) {
	Mat output;
	cv::idft(original, output, DFT_REAL_OUTPUT | DFT_SCALE);  // Applying DFT without padding
	return output;
}

Mat createFourier(cv::Mat original, int flag = 0) {
    Mat planes[] = { Mat_<double>(original), Mat::zeros(original.size(), CV_64F) };
    cv::Mat complexI;
    cv::merge(planes, 2, complexI);
    cv::dft(complexI, complexI, flag);  // Applying DFT without padding
    return complexI;
}

Mat *create_feature_map(Mat& patch, int full, int &nChns, Mat& Gray, bool scaling) {
    int h = patch.rows, w = patch.cols;
    float* M = (float*)calloc(h * w, sizeof(float));
    float* O = (float*)calloc(h * w, sizeof(float));

    float *img = convert1DArray(patch);
    //patch= convert2DImage(img, w, h);
    //Mat patch8U= convertFloatImg(patch);
    //cout<<"Before Computing Gradients"<<endl;
    gradMag(img, M, O, h, w, 1, full);
    //imshow("Patch testing", patch8U);
    //waitKey();
    //cout<<"After Computing Gradients"<<endl;
    if(!scaling) {
        hParams.binSize = tParams.bin_size;//1;
    } else {
        hParams.binSize = 4;
    }
    int hb = h / hParams.binSize;
    int wb = w / hParams.binSize;

    nChns = hParams.nOrients * 3 + 5;
    float *H = (float*)calloc(hb * wb * nChns, sizeof(float));
    //cout<<"Before FHOG SSE"<<endl;
    fhogSSE(M, O, H, h, w, hParams.binSize, hParams.nOrients, hParams.softBin, hParams.clipHog);
    //cout<<"After FHOG SSE"<<endl;
    Mat GrayRes;
    if(!scaling)
        resize(Gray, GrayRes, Size(wb, hb));
    //cout<<"Gray size patch "<<GrayRes.cols<<" "<<GrayRes.rows<<" "<<wb<<" "<<hb<<endl;
    int l = 0;
    Mat *featureMap;
    if(!scaling) {
        nChns = 28;
        featureMap = new Mat[nChns];
        for(int i = 0; i < nChns; i++)
            featureMap[i] = cv::Mat(hb, wb, CV_32FC1);

        GrayRes.convertTo(featureMap[0], CV_32FC1);
        for(int j = 0; j < wb; j++)
            for(int i = 0; i < hb; i++)
                for(int k = 0; k < nChns - 1; k++)
                    featureMap[k + 1].at<float>(i, j) = H[k * (hb * wb) + j * hb + i];

        //cout<<"finished the feature map "<<endl;
    } else {
        nChns = 31;
        featureMap = new Mat[nChns];
        for(int i = 0; i < nChns; i++)
            featureMap[i] = cv::Mat(hb, wb, CV_32FC1);

        for(int j = 0; j < wb; j++)
            for(int i = 0; i < hb; i++)
                for(int k = 0; k < nChns; k++)
                    featureMap[k].at<float>(i, j) = H[k * (hb * wb) + j * hb + i];


    }

    free(img);
    free(H);
    free(M);
    free(O);

    return featureMap;
}

Mat *get_translation_sample_esm(cv::Mat patch, Size padded, int &nDims, Mat trans_cos_win) {
    Mat roi;
    resize(patch, roi, padded);
    Mat roiFl;
    roi.convertTo(roiFl, CV_32FC1);

    Mat roiGrayFlot = roiFl;
    int nChns;
    Mat *featureMap = create_feature_map(roiFl, 1, nChns, roiGrayFlot, false);
    nDims = nChns;

    for(int i = 0; i < nChns; i++)
        featureMap[i] = featureMap[i].mul(trans_cos_win);

    return featureMap;
}

Mat *trainOnce(Mat patch, int height, int width, Mat &den, Size& padded, Mat &trans_cos_win) {
    int nDims = 0;

    //0- A] Compute Size of padded patch
    padded.width = floor(width * (1 + tParams.padding));
    padded.height = floor(height * (1 + tParams.padding));
    int szPadding_w = padded.width / tParams.bin_size;
    int szPadding_h = padded.height / tParams.bin_size;

    //0- B] Create Translation Gaussian Filters
    float transSigma = sqrt(float(width * height)) * tParams.output_sigma_factor;
    cv::Mat transFilter(szPadding_h, szPadding_w, CV_64FC1);
    for(int r = -szPadding_h / 2; r < ceil((double)szPadding_h / 2); r++)
        for(int c = -szPadding_w / 2; c < ceil((double)szPadding_w / 2); c++)
            transFilter.at<double>(r + szPadding_h / 2, c + szPadding_w / 2) = exp(-0.5 * ((double)((r + 1) * (r + 1) + (c + 1) * (c + 1)) / (transSigma * transSigma)));
    Mat transFourier = createFourier(transFilter);

    //0- C] Create Cosine Windows to give less weight to boundarie
    //cv::Mat trans_cosine_win(szPadding_h, szPadding_w, CV_32FC1);
    cv::Mat cos1 = hann(szPadding_h);
    cv::Mat cos2 = hann(szPadding_w);
    trans_cos_win = cos1 * cos2.t();

    //0- D] Setting initial parameters
    hParams.binSize = tParams.bin_size;
    hParams.nOrients = 9;
    hParams.clipHog = 0.2;
    hParams.softBin = -1;

    //1- Compute Feature map of translaation sample
    Mat *feature_map = get_translation_sample_esm(patch, padded, nDims, trans_cos_win);

    //2- Compute Denominator Translation, Numerator Translation
    Mat *feature_map_fourier = new Mat[nDims];
    Mat *num = new Mat[nDims];

    den = cv::Mat(feature_map[0].rows, feature_map[0].cols, CV_64FC2);
    den = cv::Mat::zeros(feature_map[0].rows, feature_map[0].cols, CV_64FC2);

    //cout<<"Feature Map Fourier Translation"<<endl;
    for(int i = 0; i < nDims; i++) {
        Mat feature_map_double(feature_map[i].rows, feature_map[i].cols, CV_64FC1);
        feature_map[i].convertTo(feature_map_double, CV_64FC1);
        feature_map_fourier[i] = createFourier(feature_map_double);
        mulSpectrums(transFourier, feature_map_fourier[i], num[i], 0, true);

        Mat temp;
        mulSpectrums(feature_map_fourier[i], feature_map_fourier[i], temp, 0, true);
        den = den + temp;
    }

    delete[] feature_map;
    delete[] feature_map_fourier;

    return num;
}

double computeCorrelationVariance(Mat originalPatch, Size padded, Mat *num_trans,
                                  Mat den_trans, Mat trans_cos_win) {
    Mat resized;
    resize(originalPatch, resized, Size(padded.width, padded.height));

    //Convert 1D array to 2D patch
    Mat patch(resized.rows, resized.cols, CV_32FC1);
    resized.convertTo(patch, CV_32FC1);
    Mat roiGrayFlot = patch;
	Mat patch8;
	//patch.convertTo(patch8, CV_8UC1);
	patch8= convertFloatImg(patch);
	imshow("testing patch ", patch8);
    waitKey();

    int nDims = 0;
    //Compute HOG  features
    int nChns;
    Mat *featureMap = create_feature_map(patch, 1, nChns, roiGrayFlot, false);
    //Compute Correlation
    nDims = nChns;
    for(int i = 0; i < nChns; i++)
        featureMap[i] = featureMap[i].mul(trans_cos_win);

    Mat *feature_map_fourier = new Mat[nDims];
    for(int i = 0; i < nDims; i++) {
        Mat feature_map_double(featureMap[i].rows, featureMap[i].cols, CV_64FC1);
        featureMap[i].convertTo(feature_map_double, CV_64FC1);
        feature_map_fourier[i] = createFourier(feature_map_double);
    }

    Mat* temp = new Mat[nDims];
    for(int i = 0; i < nDims; i++)
        mulSpectrums(num_trans[i], feature_map_fourier[i], temp[i], 0, false);

    int w = num_trans[0].cols, h = num_trans[0].rows;

    Mat sumDen(h, w, CV_64F);

    for(int j = 0; j < h; j++)
        for(int k = 0; k < w; k++)
            sumDen.at<double>(j, k) = den_trans.at<Vec2d>(j, k)[0] + tParams.lambda;

    Mat sumTemp(h, w, CV_64FC2);
    sumTemp = cv::Mat::zeros(sumTemp.size(), CV_64FC2);
    for(int j = 0; j < h; j++)
        for(int k = 0; k < w; k++) {
            for(int i = 0; i < nDims; i++)
                sumTemp.at<Vec2d>(j, k) += temp[i].at<Vec2d>(j, k);

            sumTemp.at<Vec2d>(j, k) /= sumDen.at<double>(j, k);
        }

    //Compute Final Translation Response
    Mat trans_response = cv::Mat::zeros(sumTemp.rows, sumTemp.cols, CV_64FC1);
    trans_response = inverseFourier(sumTemp);
    Mat trans2 = convertNormalizedFloatImg(trans_response);
    //imshow("Fake Translation Response ", trans2);
    //waitKey();

    double *params = computeMeanVariance(trans2);
    double var = params[1];

    delete[] params;
    delete[] featureMap;
    delete[] feature_map_fourier;
    delete[] temp;

    return var;
}

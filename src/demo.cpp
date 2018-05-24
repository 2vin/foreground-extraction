#include <cstdlib>
#include <cstdio>
#include <ctime>

#include "slic.h"
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;
int main(int argc, char** argv)
{
	if (argc != 4) {
		printf("usage: ./main <filename> <number of superpixels> <scale>\n");
		exit(-1);
	}

	cv::Mat img, result;
	
	//img = imread(argv[1]);
	//int numSuperpixel = atoi(argv[2]);

	img = imread(argv[1]);
	Mat orig = img.clone();
	Mat blurr;
	GaussianBlur( img, blurr, Size(7,7), 1.5, 1.5, BORDER_DEFAULT );

	float scale = std::stoi(argv[3])/100.0;
	resize(img, img, Size(),scale, scale);
	medianBlur(img, img, 5);
	int numSuperpixel = std::stoi(argv[2]);

	SLIC slic;
	
	clock_t clock_begin, clock_end;
	clock_begin = clock();
	
	slic.GenerateSuperpixels(img, numSuperpixel);
	
	
	if (img.channels() == 3) 
		result = slic.GetImgWithContours(cv::Scalar(0, 0, 255));
	else
		result = slic.GetImgWithContours(cv::Scalar(128));

	int *labels;
	double *labelCount;
	double *meanLabR, *meanLabG, *meanLabB;
	int height = img.rows;
	int width = img.cols;
	int sz = height * width;
	labels = new int[sz]; 
	labelCount = new double[sz]; 
	meanLabR = new  double[numSuperpixel];
	meanLabG = new  double[numSuperpixel];
	meanLabB = new  double[numSuperpixel];
	labels = slic.GetLabel();

	slic.SaveSuperpixelLabels(labels, width, height, "", "");

	imshow("img", img);
	for(int i = 0; i<width; i++)
	{
		for(int j=0; j<height; j++)
		{
			meanLabB[labels[j*width+i]] = 0;
			meanLabG[labels[j*width+i]] = 0;
			meanLabR[labels[j*width+i]] = 0;

			labelCount[labels[j*width+i]] = 0;
		}
	}


	for(int i = 0; i<width; i++)
	{
		for(int j=0; j<height; j++)
		{
			meanLabB[labels[j*width+i]] += img.at<Vec3b>(j, i)[0];
			meanLabG[labels[j*width+i]] += img.at<Vec3b>(j, i)[1];
			meanLabR[labels[j*width+i]] += img.at<Vec3b>(j, i)[2];

			labelCount[labels[j*width+i]] += 1.0;
		}
	}

	for(int i = 0; i<numSuperpixel; i++)
	{
		if(labelCount[i]>0)
		{
			meanLabR[i] = (meanLabR[i]*1.0/labelCount[i]);
			meanLabG[i] = (meanLabG[i]*1.0/labelCount[i]);
			meanLabB[i] = (meanLabB[i]*1.0/labelCount[i]);
		}
	}


	for(int i = 0; i<width; i++)
	{
		for(int j=0; j<height; j++)
		{
			result.at<Vec3b>(j, i)[2] = (int)meanLabR[labels[j*width+i]];
			result.at<Vec3b>(j, i)[1] = (int)meanLabG[labels[j*width+i]];
			result.at<Vec3b>(j, i)[0] = (int)meanLabB[labels[j*width+i]];
		}
	}


	clock_end = clock();
	printf("time elapsed: %f (ms), for img size: %dx%d\n", (float)(clock_end - clock_begin) / CLOCKS_PER_SEC * 1000, img.rows, img.cols);


	// for(int i=0; i<sz; i++)
	// {
	// 	cout<<labels[i]<<" \n ";
	// }

	resize(result, result, Size(400, 400*result.rows/result.cols));

	

	// cvtColor(orig, orig, CV_BGR2GRAY);
	// cvtColor(blurr, blurr, CV_BGR2GRAY);
	// orig.convertTo(orig, CV_16S);
	// blurr.convertTo(blurr, CV_16S);
	
	// /// Generate grad_x and grad_y
	// Mat grad_x, grad_y;
	// Mat abs_grad_x, abs_grad_y;

	// Mat grad, gradblur;
	// /// Gradient X
	// //Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
	// Sobel( orig, grad_x, CV_16S, 1, 0, 3, 1, 0, BORDER_DEFAULT );
	// convertScaleAbs( grad_x, abs_grad_x );

	// /// Gradient Y
	// //Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
	// Sobel( orig, grad_y, CV_16S, 0, 1, 3, 1, 0, BORDER_DEFAULT );
	// convertScaleAbs( grad_y, abs_grad_y );

	// /// Total Gradient (approximate)
	// addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );

	// /// Gradient X
	// //Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
	// Sobel( blurr, grad_x, CV_16S, 1, 0, 3, 1, 0, BORDER_DEFAULT );
	// convertScaleAbs( grad_x, abs_grad_x );

	// /// Gradient Y
	// //Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
	// Sobel( blurr, grad_y, CV_16S, 0, 1, 3, 1, 0, BORDER_DEFAULT );
	// convertScaleAbs( grad_y, abs_grad_y );

	// /// Total Gradient (approximate)
	// addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, gradblur );


	// Mat ratio;
	// grad.convertTo(grad, CV_32F);
	// gradblur.convertTo(gradblur, CV_32F);
	
	// ratio = grad/gradblur;

	// Mat sigma;
	// sigma = 1.0/(ratio*ratio - Scalar(1));
	// Mat fin;
	// cv::pow(sigma, 0.5, sigma);
	// // divide(grad, gradblur, ratio);
	// normalize(sigma, sigma, 0,255, NORM_MINMAX);

	
	cv::imwrite("output.jpg", result);
	cv::imshow("result", result);
	// cv::imshow("grad", grad);
	// cv::imshow("gradblur", gradblur);
	// cv::imshow("ratio", ratio);
	// cv::imshow("sigma", sigma);
	cv::waitKey(0);

	return 0;
}

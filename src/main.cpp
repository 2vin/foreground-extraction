#include <iostream>
#include <string>
#include <cstdlib>
#include <cstdio>
#include <ctime>

#include "gc.h"
#include "slic.h"
#include <opencv2/opencv.hpp>
#include "face_x.h"

using namespace std;
using namespace cv;

const string kModelFileName = "./data/model.xml.gz";
const string kAlt2 = "./data/haarcascade_frontalface_alt2.xml";
string kTestImage = "test.jpg";

vector<Rect> totFaces;
 
Mat ColorSegment(Mat& src, int clusterCount)
{
	Mat samples(src.rows * src.cols, 3, CV_32F);
	for( int y = 0; y < src.rows; y++ )
	for( int x = 0; x < src.cols; x++ )
	  for( int z = 0; z < 3; z++)
	    samples.at<float>(y + x*src.rows, z) = src.at<Vec3b>(y,x)[z];

	Mat labels;
	int attempts = 5;
	Mat centers;
	kmeans(samples, clusterCount, labels, TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 10000, 0.0001), attempts, KMEANS_PP_CENTERS, centers );


	Mat new_image( src.size(), src.type() );
	for( int y = 0; y < src.rows; y++ )
	for( int x = 0; x < src.cols; x++ )
	{ 
	  int cluster_idx = labels.at<int>(y + x*src.rows,0);
	  new_image.at<Vec3b>(y,x)[0] = centers.at<float>(cluster_idx, 0);
	  new_image.at<Vec3b>(y,x)[1] = centers.at<float>(cluster_idx, 1);
	  new_image.at<Vec3b>(y,x)[2] = centers.at<float>(cluster_idx, 2);
	}

	return new_image;
}

Mat SuperPixels(Mat img, float& scale, int& numSuperpixel)
{
	Mat result;

	resize(img, img, Size(),scale, scale);
	medianBlur(img, img, 5);

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

	// resize(result, result, Size(400, 400*result.rows/result.cols));


	clock_end = clock();
	printf("Superpixelization: %f seconds, for img size: %dx%d\n", (float)(clock_end - clock_begin) / CLOCKS_PER_SEC , img.rows, img.cols);

	return result;
}


Mat AlignImage(CascadeClassifier face_cascade, Mat img)
{
	Mat image = img.clone();
	// cv::Mat image = cv::imread(kTestImage);
	cv::Mat gray_image;
	cv::cvtColor(image, gray_image, CV_BGR2GRAY);
	equalizeHist( gray_image, gray_image );

	totFaces.clear();	
	face_cascade.detectMultiScale( gray_image, totFaces, 1.2, 2, 0|CV_HAAR_SCALE_IMAGE, Size(60, 60) );


	return image;
}

Mat segmentBody(Mat& image, Mat& colorSeg, float scale)
{
	cvtColor(colorSeg, colorSeg, CV_BGR2GRAY);
	Mat segBody = colorSeg.clone();
	Mat points = Mat::zeros(segBody.size(), CV_8U);
	Mat bin = Mat::zeros(segBody.size(), CV_8U);

	for (cv::Rect face : totFaces)
	{
		Point pt1 = Point(face.x*scale+face.width/2*scale, face.y*scale + face.height*1.2*scale);
		Point pt2 = Point(face.x*scale, min((int)segBody.rows, (int)(pt1.y + face.height*scale)));
		Point pt3 = Point(face.x*scale+face.width*scale, min((int)segBody.rows, (int)(pt1.y + face.height*scale)));
		// cv::rectangle(segBody, Rect(face.x*scale, face.y*scale, face.width*scale, face.height*scale), cv::Scalar(0, 0, 255), 2);
		// cv::line(segBody, pt1, pt2, Scalar(255, 0, 255));
		// cv::line(segBody, pt1, pt3, Scalar(255, 0, 255));

		for(int i = pt2.x; i<pt3.x; i=i+10)
		{
			for(int j = pt1.y; j<pt3.y; j=j+10)
			{
				int x1 = (j-pt1.y)*(pt3.x-pt1.x)*1.0/(pt3.y-pt1.y);

				if(i<pt1.x+x1 && i>pt1.x-x1)
				{
					bin += segBody==segBody.at<uchar>(j,i);
					circle(points, Point(i, j), 2, Scalar(255, 0, 255));
				}
			}
		}
	}

	// Canny(bin, bin, 100, 200, 3);
    /// Find contours   
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    RNG rng(12345);
    findContours( bin, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
    /// Draw contours
    vector<Mat> blobs;
    for( int i = 0; i< contours.size(); i++ )
    {
    	Mat drawing = Mat::zeros( bin.size(), CV_8U );
        drawContours( drawing, contours, i, Scalar(255), CV_FILLED, 8, hierarchy, 0, Point() );
        if(cv::countNonZero(points&drawing == 255))
        {
        	blobs.push_back(drawing);
        }

    }

    bin = Mat::zeros(segBody.size(), CV_8U);

    for( int i = 0; i< blobs.size(); i++ )
    {
    	bin += blobs[i];
    }  

    // bin = drawing.clone();
	imshow("faces", segBody);
	// imshow("bin", bin);
	return bin;
}

int main(int argc, char** argv)
{

	if (argc != 5) {
		printf("usage: ./main <filename> <number of superpixels> <scale*100> <clusters>\n");
		exit(-1);
	}

	bool IMAGE = true;
	GC gc;
	CascadeClassifier face_cascade;
	if( !face_cascade.load("/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml") ){ printf("--(!)Error loading\n"); return -1; };
	

	if(IMAGE)
	{
		FaceX face_x(kModelFileName);
		Mat img = imread(argv[1]);

		std::vector<Mat> planes;
		Mat hsv;
		cv::cvtColor(img, hsv, CV_BGR2HSV);
		cv::split(hsv, planes);
		cv::equalizeHist(planes[2], planes[2]);
		cv::merge(planes, hsv);
		cv::cvtColor(hsv, img, CV_HSV2BGR);		
		
		int numSuperpixel = std::stoi(argv[2]);
		float scale = std::stoi(argv[3])/100.0;
		int clusterCount = std::stoi(argv[4]);
		Mat facialFeat = AlignImage(face_cascade, img);
		Mat pixels = SuperPixels(img, scale, numSuperpixel);
		cvtColor(pixels, pixels, CV_BGR2GRAY);
		cvtColor(pixels, pixels, CV_GRAY2BGR);
		Mat colorSeg = ColorSegment(pixels, clusterCount);

		Mat bodyMask = segmentBody(img, colorSeg, scale);
		Mat imMask = gc.segmentFace(img, face_cascade);

		resize(imMask, imMask, bodyMask.size());

		Mat personBin = imMask | bodyMask;

		cv::imshow("bodyMask", bodyMask);
		cv::imshow("imMask", imMask);
		cv::imshow("person", personBin);
		cv::imshow("color segment", colorSeg);

		cv::imwrite("./results/segmented.jpg", colorSeg);
		cv::imwrite("./results/bodymask.jpg", bodyMask);
		cv::imwrite("./results/facemask.jpg", imMask);
		cv::imwrite("./results/pixels.jpg", pixels);
		cv::imwrite("./results/person.jpg", personBin);
		
		cv::waitKey(0);
	}
}

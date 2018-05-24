#include <iostream>
#include <opencv2/opencv.hpp>
#include "gc.h"
using namespace cv;
using namespace std;

CascadeClassifier face_cascade;

int main()
{	
	Mat frame;
	VideoCapture cap(0);
	cap >> frame;
	GC gc;
	if( !face_cascade.load("/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml") ){ printf("--(!)Error loading\n"); return -1; };
	while(1)
	{
		cap >> frame;

		Mat im = gc.segmentFace(frame, face_cascade);
		imshow("im", im);
		waitKey(1);
	}
	return 0;
}

#include "gc.h"


Mat origFrame;
Mat myMask, myFrame;

int frameCols, frameRows;
int imCols, imRows;

vector<Rect> Labels;

float GCScale = 0.5;

string window = "Background Removal";

Scalar RED = Scalar(0,0,255);
Scalar PINK = Scalar(230,130,255);
Scalar BLUE = Scalar(255,0,0);
Scalar LIGHTBLUE = Scalar(255,255,160);
Scalar GREEN = Scalar(0,255,0);

int BGD_KEY = EVENT_FLAG_CTRLKEY;
int FGD_KEY = EVENT_FLAG_SHIFTKEY;

Mat background_mask_t, foreground_mask_t;

string winName = "image";


static void getBinMask( const Mat& comMask, Mat& binMask )
{
    if( binMask.empty() || binMask.rows!=comMask.rows || binMask.cols!=comMask.cols )
        binMask.create( comMask.size(), CV_8UC1 );
    binMask = comMask & 1;
}


void GCApplication::reset()
{
    if( !mask.empty() )
        mask.setTo(Scalar::all(GC_BGD));
    bgdPxls.clear(); fgdPxls.clear();
    prBgdPxls.clear();  prFgdPxls.clear();

    isInitialized = false;
    rectState = NOT_SET;
    lblsState = NOT_SET;
    prLblsState = NOT_SET;
    iterCount = 0;
}

void GCApplication::setImageAndWinName( const Mat& _image, const string& _winName  )
{
    if( _image.empty() || _winName.empty() )
        return;
    image = &_image;
    winName = &_winName;
    mask.create( image->size(), CV_8UC1);
    reset();
}

Mat res;
void GCApplication::showImage() const
{
    if( image->empty() || winName->empty() )
        return;

    
    Mat binMask;
    if( !isInitialized )
        image->copyTo( res );
    else
    {
        getBinMask( mask, binMask );
        image->copyTo( res, binMask );
    }

    vector<Point>::const_iterator it;
    for( it = bgdPxls.begin(); it != bgdPxls.end(); ++it )
        circle( res, *it, radius, BLUE, thickness );
    for( it = fgdPxls.begin(); it != fgdPxls.end(); ++it )
        circle( res, *it, radius, RED, thickness );
    for( it = prBgdPxls.begin(); it != prBgdPxls.end(); ++it )
        circle( res, *it, radius, LIGHTBLUE, thickness );
    for( it = prFgdPxls.begin(); it != prFgdPxls.end(); ++it )
        circle( res, *it, radius, PINK, thickness );

    if( rectState == IN_PROCESS || rectState == SET )
        rectangle( res, Point( rect.x, rect.y ), Point(rect.x + rect.width, rect.y + rect.height ), GREEN, 2);

    // imshow( *winName, res );
    if(!binMask.empty())
    {
    	myFrame = res.clone();
    	normalize(binMask, myMask, 0, 255, NORM_MINMAX, -1, Mat() );
    }
}

void GCApplication::setRectInMask()
{
    CV_Assert( !mask.empty() );
    mask.setTo( GC_BGD );
    rect.x = max(0, rect.x);
    rect.y = max(0, rect.y);
    rect.width = min(rect.width, image->cols-rect.x);
    rect.height = min(rect.height, image->rows-rect.y);
    (mask(rect)).setTo( Scalar(GC_PR_FGD) );
}

void GCApplication::setLblsInMask( int flags, Point p, bool isPr )
{
    vector<Point> *bpxls, *fpxls;
    uchar bvalue, fvalue;
    if( !isPr )
    {
        bpxls = &bgdPxls;
        fpxls = &fgdPxls;
        bvalue = GC_BGD;
        fvalue = GC_FGD;
    }
    else
    {
        bpxls = &prBgdPxls;
        fpxls = &prFgdPxls;
        bvalue = GC_PR_BGD;
        fvalue = GC_PR_FGD;
    }

    if( flags & BGD_KEY )
    {
        bpxls->push_back(p);
        circle( mask, p, radius, bvalue, thickness );
    }
    if( flags & FGD_KEY )
    {
        fpxls->push_back(p);
        circle( mask, p, radius, fvalue, thickness );
    }

    Mat temp_background_mask = background_mask_t.clone();
    Mat temp_foreground_mask = foreground_mask_t.clone();

    resize(temp_background_mask, temp_background_mask, Size(), GCScale, GCScale);
    resize(temp_foreground_mask, temp_foreground_mask, Size(), GCScale, GCScale);

    

    for(int ic = 0; ic < temp_foreground_mask.cols*3; ic++)
    {
    	for(int jc = 0; jc < temp_foreground_mask.rows; jc++)
    	{
    		if(int(temp_foreground_mask.at<uchar>(jc,ic)) > 100)
    		{
    			fpxls->push_back(Point(ic/3,jc));
        		circle( mask, Point(ic/3,jc), radius, fvalue, thickness );
    		}

    		if(int(temp_background_mask.at<uchar>(jc,ic)) < 100 && (int(temp_foreground_mask.at<uchar>(jc,ic)) < 100))
    		{
    			fpxls->push_back(Point(ic/3,jc));
    			circle( mask, Point(ic/3,jc), 1, GC_PR_FGD, 1 );
    		}

    		if(int(temp_background_mask.at<uchar>(jc,ic)) > 100)
    		{
    			bpxls->push_back(Point(ic/3,jc));
        		circle( mask, Point(ic/3,jc), radius, bvalue, thickness );
    		}

    		
    		
    	}	
    }
}

void GCApplication::mouseClick( int event, int x, int y, int flags, void* )
{

	if( rectState == NOT_SET ) 
    {
        rect = Rect( Point(0, 0), Point(frameCols,frameRows) );
        rectState = SET;
        setRectInMask();
        CV_Assert( bgdPxls.empty() && fgdPxls.empty() && prBgdPxls.empty() && prFgdPxls.empty() );
        //showImage();
    }
    // TODO add bad args check
    switch( event )
    {
    case EVENT_LBUTTONDOWN: // set rect or GC_BGD(GC_FGD) labels
        {
            bool isb = (flags & BGD_KEY) != 0,
                 isf = (flags & FGD_KEY) != 0;
            if( rectState == NOT_SET && !isb && !isf )
            {
                rectState = IN_PROCESS;
                rect = Rect( x, y, 1, 1 );
            }
            if ( (isb || isf) && rectState == SET )
                lblsState = IN_PROCESS;
        }
        break;
    case EVENT_RBUTTONDOWN: // set GC_PR_BGD(GC_PR_FGD) labels
        {
            bool isb = (flags & BGD_KEY) != 0,
                 isf = (flags & FGD_KEY) != 0;
            if ( (isb || isf) && rectState == SET )
                prLblsState = IN_PROCESS;
        }
        break;
    case EVENT_LBUTTONUP:
        if( rectState == IN_PROCESS )
        {
            rect = Rect( Point(rect.x, rect.y), Point(x,y) );
            rectState = SET;
            setRectInMask();
            CV_Assert( bgdPxls.empty() && fgdPxls.empty() && prBgdPxls.empty() && prFgdPxls.empty() );
            showImage();
        }
        if( lblsState == IN_PROCESS )
        {
            setLblsInMask(flags, Point(x,y), false);
            lblsState = SET;
            showImage();
        }
        break;
    case EVENT_RBUTTONUP:
        if( prLblsState == IN_PROCESS )
        {
            setLblsInMask(flags, Point(x,y), true);
            prLblsState = SET;
            showImage();
        }
        break;
    case EVENT_MOUSEMOVE:
        if( rectState == IN_PROCESS )
        {
            rect = Rect( Point(rect.x, rect.y), Point(x,y) );
            CV_Assert( bgdPxls.empty() && fgdPxls.empty() && prBgdPxls.empty() && prFgdPxls.empty() );
            showImage();
        }
        else if( lblsState == IN_PROCESS )
        {
            setLblsInMask(flags, Point(x,y), false);
            showImage();
        }
        else if( prLblsState == IN_PROCESS )
        {
            setLblsInMask(flags, Point(x,y), true);
            showImage();
        }
        break;
    }

   
}

int GCApplication::nextIter()
{
    if( isInitialized )
        grabCut( *image, mask, rect, bgdModel, fgdModel, 1 );
    else
    {
        if( rectState != SET )
            return iterCount;

        if( lblsState == SET || prLblsState == SET )
            grabCut( *image, mask, rect, bgdModel, fgdModel, 1, GC_INIT_WITH_MASK );
        else
            grabCut( *image, mask, rect, bgdModel, fgdModel, 1, GC_INIT_WITH_RECT );

        isInitialized = true;
    }
    iterCount++;

    bgdPxls.clear(); fgdPxls.clear();
    prBgdPxls.clear(); prFgdPxls.clear();

    return iterCount;
}

GCApplication gcapp;

Mat GC::segmentFace(Mat image, CascadeClassifier face_cascade)
{
	if(image.empty())
		return Mat();
	origFrame = image.clone();

    imRows = origFrame.rows;
    imCols = origFrame.cols;

	Mat faceFrame = origFrame.clone();
    int numfaces = detectAndDisplay(faceFrame, face_cascade);

    if(numfaces<1)
    	return Mat();

    resize(origFrame, origFrame, Size(), GCScale, GCScale);
    
    frameCols = origFrame.cols;
    frameRows = origFrame.rows;

    
    gcapp.setImageAndWinName( origFrame, winName );
    gcapp.reset();

	if( gcapp.rectState == gcapp.NOT_SET ) 
    {
        gcapp.rect = Rect( Point(0, 0), Point(frameCols,frameRows) );
        gcapp.rectState = gcapp.SET;
        gcapp.setRectInMask();
        gcapp.lblsState = gcapp.SET;
    
    	gcapp.setLblsInMask(0, Point(0,0), false);
        gcapp.showImage();
    }

    for(int countn = 0; countn<1; countn++)
    {
        int iterCount = gcapp.getIterCount();
        int newIterCount = gcapp.nextIter();
        gcapp.showImage();
	}

    if(!myMask.empty())
    {
    	myFrame = origFrame.clone();
    	resize(myMask,myMask, origFrame.size());
    	
    	Mat temp = myFrame.clone();
		
		Mat tempMask = myMask.clone();
		erode(tempMask, tempMask, Mat());
		erode(tempMask, tempMask, Mat());
		dilate(tempMask, tempMask, Mat());
		dilate(tempMask, tempMask, Mat());
		dilate(tempMask, tempMask, Mat());
		
		// blur(tempMask, tempMask, Size(15,15));
		// // blur(tempMask, tempMask, Size(15,15));
		// // blur(tempMask, tempMask, Size(15,15));
		// cvtColor(tempMask, tempMask, CV_GRAY2BGR);
		// Mat redFrame(origFrame.size(), CV_8UC3, Scalar(0,0,255));
		// for(int i=0; i<origFrame.cols; i++)
		// {
		// 	for(int j=0; j<origFrame.rows; j++)
		// 	{
		// 		temp.at<Vec3b>(j,i)[0] = ((tempMask.at<Vec3b>(j,i)[0])/255.0)*origFrame.at<Vec3b>(j,i)[0]+((255-tempMask.at<Vec3b>(j,i)[0])/255.0)*redFrame.at<Vec3b>(j,i)[0];
		// 		temp.at<Vec3b>(j,i)[1] = ((tempMask.at<Vec3b>(j,i)[1])/255.0)*origFrame.at<Vec3b>(j,i)[1]+((255-tempMask.at<Vec3b>(j,i)[0])/255.0)*redFrame.at<Vec3b>(j,i)[1];
		// 		temp.at<Vec3b>(j,i)[2] = ((tempMask.at<Vec3b>(j,i)[2])/255.0)*origFrame.at<Vec3b>(j,i)[2]+((255-tempMask.at<Vec3b>(j,i)[0])/255.0)*redFrame.at<Vec3b>(j,i)[2];
		// 	}
		// }

		Mat backRemove = tempMask.clone();
		// hconcat(temp, origFrame, backRemove);
		resize(backRemove, backRemove, Size(), 1.0/GCScale, 1.0/GCScale);

		return backRemove;
    }
	return Mat();
}

int GC::detectAndDisplay( Mat& frame, CascadeClassifier face_cascade )
{
	std::vector<Rect> faces;
	std::vector<Rect> Labels;
	Mat frame_gray;

	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );

	face_cascade.detectMultiScale( frame_gray, faces, 1.3, 2, 0|CV_HAAR_SCALE_IMAGE, Size(90, 90) );

	Rect Hair, ROI, Neck, Shoulder, leftSide, rightSide, upSide;

	Mat maskOfFace = Mat::zeros(frame.size(), CV_8UC3);

	Mat profileMask = imread("./data/profileface_mask.png");

	

	for( size_t i = 0; i < faces.size(); i++ )
	{
		Mat profileface_mask_t = profileMask.clone();
		resize(profileface_mask_t, profileface_mask_t, Size(faces[i].width*1.6, faces[i].height*2));

		circle(frame, Point(faces[i].x+faces[i].width/2, faces[i].y+faces[i].height/2), 2,  Scalar(255,0,255), 2);
		Rect maskOfFaceRect;

		maskOfFaceRect.x = faces[i].x-faces[i].width*0.3;
		maskOfFaceRect.y = faces[i].y-faces[i].height*0.25;
		maskOfFaceRect.width = profileface_mask_t.cols;
		maskOfFaceRect.height = profileface_mask_t.rows;

		if(maskOfFaceRect.x<0)
		{
			profileface_mask_t = profileface_mask_t(Rect(-maskOfFaceRect.x, 0, profileface_mask_t.cols+maskOfFaceRect.x, profileface_mask_t.rows));
			maskOfFaceRect.width += maskOfFaceRect.x;
			maskOfFaceRect.x = 0;
		}
		if(maskOfFaceRect.y<0)
		{
			profileface_mask_t = profileface_mask_t(Rect(0, -maskOfFaceRect.y, profileface_mask_t.cols, profileface_mask_t.rows+maskOfFaceRect.y));
			maskOfFaceRect.height += maskOfFaceRect.y;
			maskOfFaceRect.y = 0;
		}
		if(maskOfFaceRect.x+maskOfFaceRect.width > frame.cols)
		{
			maskOfFaceRect.width = frame.cols - maskOfFaceRect.x;
			profileface_mask_t = profileface_mask_t(Rect(0, 0, maskOfFaceRect.width, profileface_mask_t.rows));
		}
		if(maskOfFaceRect.y+maskOfFaceRect.height > frame.rows)
		{
			maskOfFaceRect.height = frame.rows - maskOfFaceRect.y;
			profileface_mask_t = profileface_mask_t(Rect(0, 0, profileface_mask_t.cols, maskOfFaceRect.height ));
		}  	

		profileface_mask_t.copyTo(maskOfFace(maskOfFaceRect), profileface_mask_t);

		maskOfFace(Rect(maskOfFaceRect.x, maskOfFaceRect.y+maskOfFaceRect.height, maskOfFaceRect.width, maskOfFace.rows-(maskOfFaceRect.y+maskOfFaceRect.height))) = Scalar(255,255,255);

		background_mask_t = ~maskOfFace;

		for(int count = 0; count<4+4*(faces[i].width/30.0); count++)
		{
			erode(background_mask_t, background_mask_t, Mat());
		}

		addWeighted(frame, 0.6, maskOfFace, 0.4, 1.0, frame);
		addWeighted(frame, 0.8, background_mask_t, 0.2, 1.0, frame);

	}

	foreground_mask_t = maskOfFace.clone();


	return faces.size();
}
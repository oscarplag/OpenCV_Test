// OpenCV_Test.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <Windows.h>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <iostream>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/nonfree/nonfree.hpp>

//#define DISPLAY_IMAGES

using namespace cv;
using namespace std;

/// Global Variables
int DELAY_CAPTION = 1500;
int DELAY_BLUR = 100;
int MAX_KERNEL_LENGTH = 31;

Mat src; Mat dst;
char window_name[] = "Filter Demo 1";

/// Function headers
int display_caption( char* caption );
int display_dst( int delay );

int diffFilters();
int BilatCanny();
int BilatCannyGPU();
int HomographyTest();

void MatchFeaturesSurf(Mat& img_object, Mat& img_scene);
void MatchFeaturesSift(Mat& img_object, Mat& img_scene);

int BilatTest();


int _tmain(int argc, _TCHAR* argv[])
{
	//HomographyTest();
	

	BilatTest();
	//BilatCanny();
	//BilatCannyGPU();
	//diffFilters();

	return 0;
}

int HomographyTest()
{
	::LARGE_INTEGER freq,t1,t2,t3,t4;

	Mat img_object = imread( "Corrected.png", CV_LOAD_IMAGE_GRAYSCALE );
	resize(img_object, img_object, Size(img_object.cols/4, img_object.rows/4));

	Mat img_scene_orig = imread( "CorrectedRotated2.png", CV_LOAD_IMAGE_GRAYSCALE );
	resize(img_scene_orig, img_scene_orig, Size(img_scene_orig.cols/4, img_scene_orig.rows/4));
	Mat img_scene;
	bilateralFilter(img_scene_orig, img_scene, 5, 10, 3 );

	Mat descriptors_object, descriptors_scene;

	if( !img_object.data || !img_scene.data )
	{ std::cout<< " --(!) Error reading images " << std::endl; return -1; }       
	

	::QueryPerformanceFrequency(&freq);

	::QueryPerformanceCounter(&t1);
	MatchFeaturesSurf(img_object,img_scene);
	::QueryPerformanceCounter(&t2);
	::QueryPerformanceCounter(&t3);
	MatchFeaturesSift(img_object,img_scene);
	::QueryPerformanceCounter(&t4);

	double elapsedTime = (t2.QuadPart - t1.QuadPart) * 1000.0/ freq.QuadPart;
	double elapsedTime2 = (t4.QuadPart - t3.QuadPart) * 1000.0/ freq.QuadPart;

	waitKey(0);
	return 0;
}

void MatchFeaturesSurf(Mat& img_object, Mat& img_scene)
{
	//-- Step 1: Detect the keypoints using SURF Detector
	int minHessian = 400;
	
	SurfFeatureDetector detector( minHessian );
	std::vector<KeyPoint> keypoints_object, keypoints_scene;

	detector.detect( img_object, keypoints_object );
	detector.detect( img_scene, keypoints_scene );

	//-- Step 2: Calculate descriptors (feature vectors)
	SurfDescriptorExtractor extractor;

	Mat descriptors_object, descriptors_scene;

	extractor.compute( img_object, keypoints_object, descriptors_object );
	extractor.compute( img_scene, keypoints_scene, descriptors_scene );

	//-- Step 3: Matching descriptor vectors using FLANN matcher
	FlannBasedMatcher matcher;
	std::vector< DMatch > matches;
	matcher.match( descriptors_object, descriptors_scene, matches );

	double max_dist = 0; double min_dist = 100;

	//-- Quick calculation of max and min distances between keypoints
	for( int i = 0; i < descriptors_object.rows; i++ )
	{ double dist = matches[i].distance;
	if( dist < min_dist ) min_dist = dist;
	if( dist > max_dist ) max_dist = dist;
	}

	//min_dist = 0.05;

	printf("-- Max dist : %f \n", max_dist );
	printf("-- Min dist : %f \n", min_dist );

	//-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
	std::vector< DMatch > good_matches;

	for( int i = 0; i < descriptors_object.rows; i++ )
	{ if( matches[i].distance < 3*min_dist )
		{ good_matches.push_back( matches[i]); }
	}

	Mat img_matches;
	drawMatches( img_object, keypoints_object, img_scene, keypoints_scene,
				good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
				vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

	//-- Localize the object
	std::vector<Point2f> obj;
	std::vector<Point2f> scene;

	for( int i = 0; i < good_matches.size(); i++ )
	{
		//-- Get the keypoints from the good matches
		obj.push_back( keypoints_object[ good_matches[i].queryIdx ].pt );
		scene.push_back( keypoints_scene[ good_matches[i].trainIdx ].pt );
	}

	Mat H = findHomography( obj, scene, CV_RANSAC );

	//-- Get the corners from the image_1 ( the object to be "detected" )
	std::vector<Point2f> obj_corners(4);
	obj_corners[0] = cvPoint(0,0); obj_corners[1] = cvPoint( img_object.cols, 0 );
	obj_corners[2] = cvPoint( img_object.cols, img_object.rows ); obj_corners[3] = cvPoint( 0, img_object.rows );
	std::vector<Point2f> scene_corners(4);

	perspectiveTransform( obj_corners, scene_corners, H);

	//-- Draw lines between the corners (the mapped object in the scene - image_2 )
	line( img_matches, scene_corners[0] + Point2f( img_object.cols, 0), scene_corners[1] + Point2f( img_object.cols, 0), Scalar(0, 255, 0), 4 );
	line( img_matches, scene_corners[1] + Point2f( img_object.cols, 0), scene_corners[2] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );
	line( img_matches, scene_corners[2] + Point2f( img_object.cols, 0), scene_corners[3] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );
	line( img_matches, scene_corners[3] + Point2f( img_object.cols, 0), scene_corners[0] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );

	//-- Show detected matches
	namedWindow("Surf Matches",WINDOW_AUTOSIZE);
	imshow( "Surf Matches", img_matches );
}

void MatchFeaturesSift(Mat& img_object, Mat& img_scene)
{
	//-- Step 1: Detect the keypoints using SURF Detector
	SiftFeatureDetector* detector = new SiftFeatureDetector();
	
	std::vector<KeyPoint> keypoints_object, keypoints_scene;

	detector->detect( img_object, keypoints_object );
	detector->detect( img_scene, keypoints_scene );

	//-- Step 2: Calculate descriptors (feature vectors)
	SiftDescriptorExtractor extractor;

	Mat descriptors_object, descriptors_scene;

	extractor.compute( img_object, keypoints_object, descriptors_object );
	extractor.compute( img_scene, keypoints_scene, descriptors_scene );

	//-- Step 3: Matching descriptor vectors using FLANN matcher
	FlannBasedMatcher matcher;
	std::vector< DMatch > matches;
	matcher.match( descriptors_object, descriptors_scene, matches );

	double max_dist = 0; double min_dist = 100;

	//-- Quick calculation of max and min distances between keypoints
	for( int i = 0; i < descriptors_object.rows; i++ )
	{ double dist = matches[i].distance;
	if( dist < min_dist ) min_dist = dist;
	if( dist > max_dist ) max_dist = dist;
	}

	//min_dist = 0.05;

	printf("-- Max dist : %f \n", max_dist );
	printf("-- Min dist : %f \n", min_dist );

	//-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
	std::vector< DMatch > good_matches;

	for( int i = 0; i < descriptors_object.rows; i++ )
	{ if( matches[i].distance < 3*min_dist )
		{ good_matches.push_back( matches[i]); }
	}

	Mat img_matches;
	drawMatches( img_object, keypoints_object, img_scene, keypoints_scene,
				good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
				vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

	//-- Localize the object
	std::vector<Point2f> obj;
	std::vector<Point2f> scene;

	for( int i = 0; i < good_matches.size(); i++ )
	{
		//-- Get the keypoints from the good matches
		obj.push_back( keypoints_object[ good_matches[i].queryIdx ].pt );
		scene.push_back( keypoints_scene[ good_matches[i].trainIdx ].pt );
	}

	Mat H = findHomography( obj, scene, CV_RANSAC );

	//-- Get the corners from the image_1 ( the object to be "detected" )
	std::vector<Point2f> obj_corners(4);
	obj_corners[0] = cvPoint(0,0); obj_corners[1] = cvPoint( img_object.cols, 0 );
	obj_corners[2] = cvPoint( img_object.cols, img_object.rows ); obj_corners[3] = cvPoint( 0, img_object.rows );
	std::vector<Point2f> scene_corners(4);

	perspectiveTransform( obj_corners, scene_corners, H);

	//-- Draw lines between the corners (the mapped object in the scene - image_2 )
	line( img_matches, scene_corners[0] + Point2f( img_object.cols, 0), scene_corners[1] + Point2f( img_object.cols, 0), Scalar(0, 255, 0), 4 );
	line( img_matches, scene_corners[1] + Point2f( img_object.cols, 0), scene_corners[2] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );
	line( img_matches, scene_corners[2] + Point2f( img_object.cols, 0), scene_corners[3] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );
	line( img_matches, scene_corners[3] + Point2f( img_object.cols, 0), scene_corners[0] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );

	//-- Show detected matches
	namedWindow("Sift Matches",WINDOW_AUTOSIZE);
	imshow( "Sift Matches", img_matches );
}

int BilatCannyGPU()
{
	Mat tmp;
	Mat imgOrig = imread("corrected.png", CV_LOAD_IMAGE_GRAYSCALE);
	if(! imgOrig.data )                              // Check for invalid input
	{
		cout <<  "Could not open or find the image" << std::endl ;
		return -1;
	}
	resize(imgOrig, imgOrig, Size(imgOrig.cols/4, imgOrig.rows/4)); // resized to half size
	gpu::GpuMat img(imgOrig);
	namedWindow( "Original Image", CV_WINDOW_AUTOSIZE  );
	img.download(tmp);
	imshow("Original Image",tmp);

	gpu::GpuMat bilat;	
	gpu::bilateralFilter ( img, bilat, 10, 20, 5 );
	namedWindow("Bilater Filtered Image",CV_WINDOW_AUTOSIZE);
	bilat.download(tmp);
	imshow("Bilater Filtered Image",tmp);

	gpu::GpuMat canOrig;
	int canThresh = 20;
	gpu::Canny(img,canOrig,canThresh,3*canThresh);
	namedWindow("Canny Filtered Image",CV_WINDOW_AUTOSIZE);
	canOrig.download(tmp);
	imshow("Canny Filtered Image",tmp);

	gpu::GpuMat canIm;	
	gpu::Canny(bilat,canIm,canThresh,canThresh*3);
	namedWindow("Bilateral-Canny Filtered Image",CV_WINDOW_AUTOSIZE);
	canIm.download(tmp);
	imshow("Bilateral-Canny Filtered Image",tmp);	
	waitKey(0);

	return 0;
}

int BilatCanny()
{
	Mat img = imread("corrected.png", CV_LOAD_IMAGE_GRAYSCALE);
	if(! img.data )                              // Check for invalid input
	{
		cout <<  "Could not open or find the image" << std::endl ;
		return -1;
	}
	resize(img, img, Size(img.cols/4, img.rows/4)); // resized to half size
	namedWindow( "Original Image", CV_WINDOW_AUTOSIZE  );
	imshow("Original Image",img);

	Mat bilat;	
	bilateralFilter ( img, bilat, 10, 20, 5 );
	namedWindow("Bilater Filtered Image",CV_WINDOW_AUTOSIZE);
	imshow("Bilater Filtered Image",bilat);

	Mat canOrig;
	int canThresh = 20;
	Canny(img,canOrig,canThresh,3*canThresh);
	namedWindow("Canny Filtered Image",CV_WINDOW_AUTOSIZE);
	imshow("Canny Filtered Image",canOrig);

	Mat canIm;	
	Canny(bilat,canIm,canThresh,canThresh*3);
	namedWindow("Bilateral-Canny Filtered Image",CV_WINDOW_AUTOSIZE);
	imshow("Bilateral-Canny Filtered Image",canIm);	
	waitKey(0);

	return 0;
}

int diffFilters()
{
		/// Load the source image
	src = imread( "Corrected.png", CV_LOAD_IMAGE_GRAYSCALE );
	if(! src.data )                              // Check for invalid input
	{
		cout <<  "Could not open or find the image" << std::endl ;
		return -1;
	}

	resize(src, src, Size(src.cols/4, src.rows/4)); // resized to half size
	
	namedWindow( "Display window", CV_WINDOW_AUTOSIZE  );

	if( display_caption( "Original Image" ) != 0 ) { return 0; }

	dst = src.clone();
	if( display_dst( DELAY_CAPTION ) != 0 ) { return 0; }

	/// Applying Homogeneous blur
	if( display_caption( "Homogeneous Blur" ) != 0 ) { return 0; }

	for ( int i = 1; i < MAX_KERNEL_LENGTH; i = i + 2 )
	{ blur( src, dst, Size( i, i ), Point(-1,-1) );
	if( display_dst( DELAY_BLUR ) != 0 ) { return 0; } }

	/// Applying Gaussian blur
	if( display_caption( "Gaussian Blur" ) != 0 ) { return 0; }

	for ( int i = 1; i < MAX_KERNEL_LENGTH; i = i + 2 )
	{ GaussianBlur( src, dst, Size( i, i ), 0, 0 );
	if( display_dst( DELAY_BLUR ) != 0 ) { return 0; } }

	/// Applying Median blur
	if( display_caption( "Median Blur" ) != 0 ) { return 0; }

	for ( int i = 1; i < MAX_KERNEL_LENGTH; i = i + 2 )
	{ medianBlur ( src, dst, i );
	if( display_dst( DELAY_BLUR ) != 0 ) { return 0; } }

	/// Applying Bilateral Filter
	if( display_caption( "Bilateral Blur" ) != 0 ) { return 0; }

	for ( int i = 1; i < MAX_KERNEL_LENGTH; i = i + 2 )
	{ bilateralFilter ( src, dst, i, i*2, i/2 );
	if( display_dst( DELAY_BLUR ) != 0 ) { return 0; } }

	/// Wait until user press a key
	display_caption( "End: Press a key!" );

	waitKey(0);
	return 0;
}

int display_caption( char* caption )
{
	dst = Mat::zeros( src.size(), src.type() );
	putText( dst, caption,
	Point( src.cols/4, src.rows/2),
	CV_FONT_HERSHEY_COMPLEX, 1, Scalar(255, 255, 255) );

	imshow( window_name, dst );
	int c = waitKey( DELAY_CAPTION );
	if( c >= 0 ) { return -1; }
	return 0;
}

int display_dst( int delay )
{
	imshow( window_name, dst );
	int c = waitKey ( delay );
	if( c >= 0 ) { return -1; }
	return 0;
}

int BilatTest()
{
	::LARGE_INTEGER freq,t1,t2,t3,t4;
	::QueryPerformanceFrequency(&freq);

	Mat img = imread("corrected.png", CV_LOAD_IMAGE_GRAYSCALE);
	if(! img.data )                              // Check for invalid input
	{
		cout <<  "Could not open or find the image" << std::endl ;
		return -1;
	}
	gpu::GpuMat imgGPU(img);
	int canThresh = 20;

	Mat bilat, canOrig, canIm;
	gpu::GpuMat bilatGPU, canOrigGPU, canImGPU;	

	::QueryPerformanceCounter(&t1);
	bilateralFilter ( img, bilat, 10, 20, 5 );
	Canny(img,canOrig,canThresh,3*canThresh);
	Canny(bilat,canIm,canThresh,canThresh*3);
	::QueryPerformanceCounter(&t2);
	::QueryPerformanceCounter(&t3);
	gpu::bilateralFilter ( imgGPU, bilatGPU, 10, 20, 5 );
	gpu::Canny(imgGPU,canOrigGPU,canThresh,3*canThresh);
	gpu::Canny(bilatGPU,canImGPU,canThresh,canThresh*3);
	::QueryPerformanceCounter(&t4);

	double elapsedTime = (t2.QuadPart - t1.QuadPart) * 1000.0/ freq.QuadPart;
	double elapsedTime2 = (t4.QuadPart - t3.QuadPart) * 1000.0/ freq.QuadPart;

	printf("CPU version execution time: %fms\n",elapsedTime);
	printf("GPU version execution time: %fms\n",elapsedTime2);	

	waitKey(0);
	return 0;
}
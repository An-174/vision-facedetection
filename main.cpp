#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/objdetect.hpp>
#include <iostream>
using namespace std;
using namespace cv;

int main()
{
	namedWindow("face_detect");
	Mat im = imread("../lena.jpg");
	
	vector<Rect> faces;
	Mat frame_gray;
	cvtColor(im,frame_gray,COLOR_BGR2GRAY);
	equalizeHist(frame_gray,frame_gray);

	CascadeClassifier face_cascade;
	if( !face_cascade.load("../haarcascade_frontalface_alt.xml") )
	{
		cout<<"--(!)Error loading face cascade\n"; 
		return -1; 
	}
	face_cascade.detectMultiScale(frame_gray,faces);
	
	for(size_t i = 0; i < faces.size(); i++ )
		rectangle(im,faces[i],Scalar(0,0,255),2);
	
	imshow("face_detect",im);
	while(static_cast<unsigned char>(waitKey(0)) != 'q');

	return 0;
}

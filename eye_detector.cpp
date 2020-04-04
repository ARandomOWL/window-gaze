/* Simple Iris detector using OpenCV2.
 * Adapted from https://abnerrjo.github.io/blog/2017/01/28/eyeball-tracking-for-mouse-control-in-opencv/
 */

#include <iostream>

#include <core/core.hpp>
#include <highgui/highgui.hpp>
#include <imgproc/imgproc.hpp>
#include <objdetect/objdetect.hpp>

cv::Vec3f getEyeball(cv::Mat &eye, std::vector<cv::Vec3f> &circles) {
	std::vector<int> sums(circles.size(), 0);
	for (int y = 0; y < eye.rows; y++) {
			uchar *ptr = eye.ptr<uchar>(y);
			for (int x = 0; x < eye.cols; x++) {
					int value = static_cast<int>(*ptr);
					for (int i = 0; i < circles.size(); i++) {
							cv::Point center((int)std::round(circles[i][0]),
								             (int)std::round(circles[i][1]));
							int radius = (int)std::round(circles[i][2]);
							if (std::pow(x - center.x, 2) +
								std::pow(y - center.y, 2) < std::pow(radius, 2)) {
									sums[i] += value;
							}
					}
					++ptr;
			}
	}
	int smallestSum = 9999999;
	int smallestSumIndex = -1;
	for (int i = 0; i < circles.size(); i++) {
			if (sums[i] < smallestSum) {
					smallestSum = sums[i];
					smallestSumIndex = i;
			}
	}
	return circles[smallestSumIndex];
}

cv::Rect getLeftmostEye(std::vector<cv::Rect> &eyes) {
	int leftmost = 99999999;
	int leftmostIndex = -1;
	for (int i = 0; i < eyes.size(); i++) {
			if (eyes[i].tl().x < leftmost) {
					leftmost = eyes[i].tl().x;
					leftmostIndex = i;
			}
	}
	return eyes[leftmostIndex];
}

std::vector<cv::Point> centers;
cv::Point lastPoint;
cv::Point mousePoint;

cv::Point stabilize(std::vector<cv::Point> &points, int windowSize) {
	float sumX = 0;
	float sumY = 0;
	int count = 0;
	for (int i = std::max(0, (int)(points.size() - windowSize));
		 i < points.size(); i++) {
			sumX += points[i].x;
			sumY += points[i].y;
			++count;
	}
	if (count > 0) {
			sumX /= count;
			sumY /= count;
	}
	return cv::Point(sumX, sumY);
}

void detectEyes(cv::Mat &frame, cv::CascadeClassifier &faceCascade,
				cv::CascadeClassifier &eyeCascade) {
	/* Convert image to grayscale */
	cv::Mat grayscale;
	cv::cvtColor(frame, grayscale, cv::COLOR_BGR2GRAY);
	/* Enhance image contrast */
	cv::equalizeHist(grayscale, grayscale);

	/* Detect face */
	std::vector<cv::Rect> faces;
	faceCascade.detectMultiScale(grayscale, faces, 1.1, 2,
								 0 | cv::CASCADE_SCALE_IMAGE, cv::Size(150, 150));
	if (faces.size() == 0) return;		/* No face was detected */
	cv::Mat face = grayscale(faces[0]);	/* Crop face */

	/* Detect eyes */
	std::vector<cv::Rect> eyes;
	eyeCascade.detectMultiScale(face, eyes, 1.1, 2,
								0 | cv::CASCADE_SCALE_IMAGE, cv::Size(30, 30));
	rectangle(frame, faces[0].tl(), faces[0].br(), cv::Scalar(255, 0, 0), 2);
	if (eyes.size() != 2) return; /* Both eyes were not detected */
	for (cv::Rect &eye : eyes) {
			rectangle(frame, faces[0].tl() + eye.tl(), faces[0].tl() + eye.br(),
					  cv::Scalar(0, 255, 0), 2);
	}
	cv::Rect eyeRect = getLeftmostEye(eyes);
	cv::Mat eye = face(eyeRect); /* Crop the leftmost eye */
	cv::equalizeHist(eye, eye);

	/* Detect eyeball */
	std::vector<cv::Vec3f> circles;
	cv::HoughCircles(eye, circles, cv::HOUGH_GRADIENT, 1, eye.cols / 8, 250, 15,
					 eye.rows / 8, eye.rows / 3);
	if (circles.size() > 0) {
			cv::Vec3f eyeball = getEyeball(eye, circles);
			cv::Point center(eyeball[0], eyeball[1]);
			centers.push_back(center);
			center = stabilize(centers, 5);
			if (centers.size() > 1) {
					cv::Point diff;
					diff.x = (center.x - lastPoint.x) * 20;
					diff.y = (center.y - lastPoint.y) * -30;
					mousePoint += diff;
			}
			lastPoint = center;
			int radius = (int)eyeball[2];
			cv::circle(frame, faces[0].tl() + eyeRect.tl() + center, radius,
					   cv::Scalar(0, 0, 255), 2);
			cv::circle(eye, center, radius, cv::Scalar(255, 255, 255), 2);
	}
	cv::imshow("Eye", eye);
}

void changeMouse(cv::Mat &frame, cv::Point &location) {
	if (location.x > frame.cols) location.x = frame.cols;
	if (location.x < 0) location.x = 0;
	if (location.y > frame.rows) location.y = frame.rows;
	if (location.y < 0) location.y = 0;
	/* Use external program 'xdotool' to move mouse */
	system(("xdotool mousemove " + std::to_string(location.x) + " " +
			std::to_string(location.y)).c_str());
}

int main(int argc, char **argv) {
	if (argc != 2) {
			std::cerr << "Usage: EyeDetector <WEBCAM_INDEX>" << std::endl;
			return -1;
	}

	/* Load Haar cascades from OpenCV */
	cv::CascadeClassifier faceCascade;
	cv::CascadeClassifier eyeCascade;
	cv::String face_cascade_name = cv::samples::findFile("haarcascades/haarcascade_frontalface_alt.xml");
	cv::String eye_cascade_name = cv::samples::findFile("haarcascades/haarcascade_eye_tree_eyeglasses.xml");
	if (!faceCascade.load(face_cascade_name)) {
			std::cerr << "Could not load face detector." << std::endl;
			return -1;
	}		 
	if (!eyeCascade.load(eye_cascade_name)) {
			std::cerr << "Could not load eye detector." << std::endl;
			return -1;
	}

	/* Open webcam */
	cv::VideoCapture cap(atoi(argv[1]));
	if (!cap.isOpened()) {
			std::cerr << "Webcam not detected." << std::endl;
			return -1;
	}		 
	cv::Mat frame;
	mousePoint = cv::Point(800, 800);

	/* Main loop */
	while (1) {
			cap >> frame; /* Store the webcam image in a cv::Mat */
			if (!frame.data) break;
			detectEyes(frame, faceCascade, eyeCascade);
			// changeMouse(frame, mousePoint);
			cv::imshow("Webcam", frame);
			/* Capture at 33 FPS until any key pressed */
			if (cv::waitKey(30) >= 0) break;
	}
	return 0;
}

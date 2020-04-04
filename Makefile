CPP_FLAGS=-std=c++11
OPENCV_LIBS=-lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_objdetect -lopencv_imgcodecs -lopencv_videoio
LD_FLAGS=-I/usr/include/opencv4 -I/usr/include/opencv4/opencv2 $(OPENCV_LIBS)

target=eye_detector
default: $(target)
$(target): $(target).cpp
	g++ $(CPP_FLAGS) $^ -o $@ $(LD_FLAGS)
clean:
	rm -f $(target)

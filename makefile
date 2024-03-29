CC = g++
CFLAGS = -std=c++17
LIBS = -ltbb -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs
INCLUDES = -I/opt/homebrew/Cellar/tbb/2021.10.0/include -I/opt/homebrew/Cellar/opencv/4.8.1_2/include/opencv4
LDFLAGS = -L/opt/homebrew/Cellar/tbb/2021.10.0/lib -L/opt/homebrew/Cellar/opencv/4.8.1_2/lib

SRCS = image_processing.cpp image_processing_tbb.cpp parallel_algos.cpp ocv.cpp sequential.cpp
EXECS = image_processing.exe image_processing_tbb.exe parallel_algos.exe ocv.exe sequential.exe
OBJS = $(SRCS:.cpp=.o)

all: $(EXECS)

image_processing.exe: image_processing.cpp
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o image_processing.o
	$(CC) $(CFLAGS) -o $@ image_processing.o $(LIBS) $(LDFLAGS)

image_processing_tbb.exe: image_processing_tbb.cpp
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o image_processing_tbb.o
	$(CC) $(CFLAGS) -o $@ image_processing_tbb.o $(LIBS) $(LDFLAGS)

parallel_algos.exe: parallel_algos.cpp
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o parallel_algos.o
	$(CC) $(CFLAGS) -o $@ parallel_algos.o $(LIBS) $(LDFLAGS)

ocv.exe: ocv.cpp
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o ocv.o
	$(CC) $(CFLAGS) -o $@ ocv.o $(LIBS) $(LDFLAGS)

sequential.exe: sequential.cpp
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o sequential.o
	$(CC) $(CFLAGS) -o $@ sequential.o $(LIBS) $(LDFLAGS)

clean:
	rm -f $(OBJS) $(EXECS)

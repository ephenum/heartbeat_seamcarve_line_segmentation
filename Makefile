CC=g++
CFLAGS=-static-libstdc++  --std=c++11
LIBS=-lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc -lopencv_core -lsplinter-static-3-0 -Bdynamic -lpthread -ldl

seamcarve: main.cpp lines.cpp
	$(CC) -o seamcarve main.cpp $(CFLAGS) $(LIBS)

.PHONE: clean

clean:
	find . -name '*.o' -delete
	rm -f seamcarve

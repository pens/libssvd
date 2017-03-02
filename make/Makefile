.PHONY: all clean
all: lib

SRC = ../src/*.cc
INC = -I../include -I/usr/local/magma/include -I/usr/local/cuda/include

FLAGS = -std=c++1y -g -pthread -O3 -march=native -pipe
LIB = -lblas -llapack -llapacke -L/usr/local/magma/lib -lmagma
OUT = -shared -fPIC -o libssvd.so

lib: $(SRC)
	mkdir --parents ../bin/
	g++ $(FLAGS) $(SRC) $(INC) $(OUT) $(LIB)
	mv *.so ../bin/

clean:
	rm -r ../bin/

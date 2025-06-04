#
#/************************************************************************************
#***
#***	Copyright 2024 Dell Du(18588220928@163.com), All Rights Reserved.
#***
#***	File Author: Dell, Wed 17 Jan 2024 10:36:30 PM CST
#***
#************************************************************************************/
#

TARGET := image3d
SOURCE := src/dinov2.c main.c
GGMLRUNTIME_INSTALL_DIR=/home/dell/WDisk/Workspace/2023-07-01/ggml

INCS := \
		-Iinclude -Isrc \
		-I${GGMLRUNTIME_INSTALL_DIR}/include \
		$(shell pkg-config nimage --cflags)

#$(shell pkg-config ggml --libs) -Wl,-rpath=/usr/local/lib
LIBS := \
	$(shell pkg-config nimage --libs) \
	-L${GGMLRUNTIME_INSTALL_DIR}/build/src -lggml -Wl,-rpath=${GGMLRUNTIME_INSTALL_DIR}/build/src \
	-L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas \
	-lm -lstdc++

CFLAGS := -O3 -fPIC 
CXXFLAGS := ${CFLAGS}
LDFLAGS := -fPIC

#****************************************************************************
# Makefile code common to all platforms
#****************************************************************************
DEFINES  := \
	-DGGML_CUDA \
	-D_GLIBCXX_USE_CXX11_ABI=0 -std=c++17

CFLAGS   := ${CFLAGS} ${DEFINES}
CXXFLAGS := ${CXXFLAGS} ${DEFINES}

OBJECT := $(addsuffix .o,$(basename ${SOURCE}))

all: ${TARGET}
# install

#****************************************************************************
# Compile block
#****************************************************************************
${TARGET}: ${OBJECT}
	# make -C lib
	${CXX} ${LDFLAGS} -o ${TARGET} ${OBJECT} ${LIBS}

install:
	${STRIP} ${TARGET}
	install -m 755 ${TARGET} ${INSTALL_DIR}/bin

#****************************************************************************
# common rules
#****************************************************************************
%.o : %.cpp
	${CXX} ${CXXFLAGS} ${INCS} -c $< -o $@

%.o : %.c
	${CXX} ${CXXFLAGS} ${INCS} -c $< -o $@

#****************************************************************************
# Depend block
#****************************************************************************
depend:

clean:
# 	make -C lib clean
	rm -f core ${OBJECT} ${TARGET}

format:
	clang-format --style="{BasedOnStyle: WebKit, TabWidth: 4, IndentWidth: 4, ColumnLimit: 120}" \
	-i include/*.h src/*.h src/*.c main.c

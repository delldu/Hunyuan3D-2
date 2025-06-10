#
#/************************************************************************************
#***
#***	Copyright 2024 Dell Du(18588220928@163.com), All Rights Reserved.
#***
#***	File Author: Dell, Wed 17 Jan 2024 10:36:30 PM CST
#***
#************************************************************************************/
#

# TARGET := image3d
# SOURCE := src/dinov2.c main.c
# GGML_RUNTIME_INSTALL_DIR=/home/dell/WDisk/Workspace/2023-07-01/ggml

# INCS := \
# 		-Iinclude -Isrc \
# 		-I${GGML_RUNTIME_INSTALL_DIR}/include \
# 		$(shell pkg-config nimage --cflags)

# #$(shell pkg-config ggml --libs) -Wl,-rpath=/usr/local/lib
# LIBS := \
# 	$(shell pkg-config nimage --libs) \
# 	-L${GGML_RUNTIME_INSTALL_DIR}/build/src -lggml -Wl,-rpath=${GGML_RUNTIME_INSTALL_DIR}/build/src \
# 	-L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas \
# 	-lm -lstdc++

# CFLAGS := -O3 -fPIC 
# CXXFLAGS := ${CFLAGS}
# LDFLAGS := -fPIC

# #****************************************************************************
# # Makefile code common to all platforms
# #****************************************************************************
# DEFINES  := \
# 	-DGGML_CUDA \
# 	-D_GLIBCXX_USE_CXX11_ABI=0 -std=c++17

# CFLAGS   := ${CFLAGS} ${DEFINES}
# CXXFLAGS := ${CXXFLAGS} ${DEFINES}

# OBJECT := $(addsuffix .o,$(basename ${SOURCE}))

# all: ${TARGET}
# # install

# #****************************************************************************
# # Compile block
# #****************************************************************************
# ${TARGET}: ${OBJECT}
# 	# make -C lib
# 	${CXX} ${LDFLAGS} -o ${TARGET} ${OBJECT} ${LIBS}

# install:
# 	${STRIP} ${TARGET}
# 	install -m 755 ${TARGET} ${INSTALL_DIR}/bin

# #****************************************************************************
# # common rules
# #****************************************************************************
# %.o : %.cpp
# 	${CXX} ${CXXFLAGS} ${INCS} -c $< -o $@

# %.o : %.c
# 	${CXX} ${CXXFLAGS} ${INCS} -c $< -o $@

# #****************************************************************************
# # Depend block
# #****************************************************************************
# depend:

# clean:
# # 	make -C lib clean
# 	rm -f core ${OBJECT} ${TARGET}

# format:
# 	clang-format --style="{BasedOnStyle: WebKit, TabWidth: 4, IndentWidth: 4, ColumnLimit: 120}" \
# 	-i include/*.h src/*.h src/*.c main.c














TARGET := image3d
SOURCE := src/image3d_shape.cpp src/image3d_paint.cpp main.cpp
REDOS_RUNTIME_INSTALL_DIR=/home/dell/WDisk/Workspace/2025-02-05/Redos
GGML_RUNTIME_INSTALL_DIR=/home/dell/WDisk/Workspace/2023-07-01/ggml

INCS	:= -Isrc/ -I/usr/local/include/eigen3 -I${GGML_RUNTIME_INSTALL_DIR}/include -I${REDOS_RUNTIME_INSTALL_DIR}/include
LIBS	:= -L${REDOS_RUNTIME_INSTALL_DIR} -lredos \
	$(shell pkg-config babl --libs) -ljpeg -lpng  \
	-L${GGML_RUNTIME_INSTALL_DIR}/build/src -lggml

CFLAGS := -O3 -fPIC 
CXXFLAGS := ${CFLAGS}
LDFLAGS := -fPIC -Wl,--as-needed

#****************************************************************************
# Makefile code common to all platforms
#****************************************************************************
DEFINES  := -DGGML_CUDA -std=c++17
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

formatc:
	clang-format --style="{BasedOnStyle: WebKit, TabWidth: 4, IndentWidth: 4, ColumnLimit: 120}" \
	-i src/*.h src/*.cpp main.cpp

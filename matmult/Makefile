TARGET	= libmatmult.so

SRC_DIR = src
OBJ_DIR = obj

LIBSRCS	= $(wildcard $(SRC_DIR)/*.cu)
LIBOBJS	= $(addprefix $(OBJ_DIR)/,$(addsuffix .o,$(notdir $(LIBSRCS))))

OPT	= -g -O3
PIC = -fpic
OMP   = -fopenmp
XPIC  = -Xcompiler -fpic -maxrregcount=64
XOPT  = -Xptxas=-v #-lineinfo #for profiler, use -G for debugging
XARCH = -arch=sm_70

CXX	= nvcc
CXXFLAGS = --compiler-options "$(OPT) $(PIC) $(OMP)" $(XARCH) $(XOPT) $(XPIC)

CUDA_PATH ?= /appl/cuda/10.0
INCLUDES = -I$(CUDA_PATH)/include -I$(CUDA_PATH)/samples/common/inc 

SOFLAGS = -shared
XLIBS	= -lcublas -L /usr/lib64/atlas -ltatlas

$(TARGET): $(LIBOBJS)
	$(CXX) -o $@ $(CXXFLAGS) $(SOFLAGS) $(INCLUDES) $^ $(XLIBS)

.SUFFIXES: $(SRC_DIR)/%.cu
$(OBJ_DIR)/%.cu.o: 
	$(CXX) -o $(OBJ_DIR)/$*.cu.o -c $(SRC_DIR)/$*.cu $(CXXFLAGS) $(SOFLAGS) $(INCLUDES)

clean:
	/bin/rm -f $(TARGET) $(LIBOBJS) 

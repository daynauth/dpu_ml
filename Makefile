# DPU compile options
DPU_CC = dpu-upmem-dpurte-clang
DPU_PROGRAM = task

# Host compile options
CC = gcc
CFLAGS = --std=c99
CC_DPU = `dpu-pkg-config --cflags --libs dpu`
PROGRAM = app

all: $(PROGRAM)

$(PROGRAM): $(DPU_PROGRAM)
	$(CC) $(CFLAGS) -DDPU_BINARY=\"./$(DPU_PROGRAM)\" host/app.c -o $(PROGRAM) $(CC_DPU)

tensor.o : dpu/tensor.c
	$(DPU_CC) -c dpu/tensor.c -o tensor.o

arena.o : dpu/arena.c
	$(DPU_CC) -c dpu/arena.c -o arena.o

lstm.o : dpu/lstm.c
	$(DPU_CC) -c dpu/lstm.c -o lstm.o

task.o : dpu/task.c
	$(DPU_CC) -c dpu/task.c -o task.o

$(DPU_PROGRAM): tensor.o arena.o task.o lstm.o
	$(DPU_CC) -o $(DPU_PROGRAM) tensor.o arena.o task.o lstm.o

clean:
	rm -f app $(DPU_PROGRAM) *.o
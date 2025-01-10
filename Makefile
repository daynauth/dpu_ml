# DPU compile options
DPU_CC = dpu-upmem-dpurte-clang
DPU_PROGRAM = task

# Host compile options
CC = gcc
CFLAGS = --std=c99
CC_DPU = `dpu-pkg-config --cflags --libs dpu`
PROGRAM = app

all: $(PROGRAM)

tensor.o: host/tensor.c
	$(CC) $(CFLAGS) -c host/tensor.c -o tensor.o $(CC_DPU)

app.o: host/app.c
	$(CC) $(CFLAGS) -c host/app.c -o app.o $(CC_DPU)

$(PROGRAM): $(DPU_PROGRAM) app.o tensor.o
	$(CC) $(CFLAGS) -DDPU_BINARY=\"./$(DPU_PROGRAM)\" app.o tensor.o -o $(PROGRAM) $(CC_DPU)

arena.o : dpu/arena.c
	$(DPU_CC) -c dpu/arena.c -o arena.o

task.o : dpu/task.c
	$(DPU_CC) -c dpu/task.c -o task.o

$(DPU_PROGRAM): arena.o task.o 
	$(DPU_CC) -o $(DPU_PROGRAM) arena.o task.o

clean:
	rm -f app $(DPU_PROGRAM) *.o
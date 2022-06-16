# How to use me:
# This makefile is made for use with GCC, not MPI.

# Place this file in the root of your project. 
# In the root, also create a file called 'bin', 'header' and 'source'

# use command `make` in the shell to run the makefile.
# All .o files will be placed in the 'bin' file.
# A binary will be generated in the root of the file.

# use ./run to run the program. 

# Compiler Command
CC = g++
CFLAGS = -c -I./header

# collecting object file names
src = $(wildcard source/*.cpp)
src1 = $(src:.cpp=.o)

objects := $(src1:source/%=bin/%)

# Compile object files into binary
all : $(objects)
	$(CC) -o run $(objects) -larmadillo

# Generate object files by compiling .cpp and .h files
bin/%.o : source/%.cpp
	$(CC) $(CFLAGS) $?
	mv *.o bin

# Clean Recipe
.PHONY : clean
clean : 
	rm -rf all $(objects)
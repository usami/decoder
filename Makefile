CPP = g++
LDFLAGS = 
RM = rm -f

SRC = decode.cpp
OBJ = $(SRC:.cpp=.o)
PROG = $(SRC:.cpp=)

FILES = Makefile $(SRC) $(PROG).py tests/*

.SUFFIXES: .o .cpp

all: $(PROG)

$(PROG): $(SRC)
	$(CPP) $(LDFLAGS) $< -o $@

clean:
	$(RM) $(PROG) $(OBJ)

test: $(PROG)
	./$(PROG)



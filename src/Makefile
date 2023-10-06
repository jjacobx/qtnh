CXX = g++
CXXFLAGS = -g -Og -Wall -Wextra -std=c++20
LDFLAGS = 
INCS = 

DEPS = tensor.hpp

ODIR = obj
_OBJ = main.o tensor.o
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))

BDIR = ../bin


$(ODIR)/%.o: %.cpp $(DEPS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(INCS) -c -o $@ $< 

gtnh.out: $(OBJ)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(INCS) -o $(BDIR)/$@ $^

.PHONY: clean

clean:
	rm -f core $(ODIR)/*.o

$(shell mkdir -p $(ODIR))

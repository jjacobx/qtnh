#include <complex>
#include <vector>

typedef std::complex<double> complex;
typedef std::vector<int> indices;

class Tensor {
public:
    Tensor();
    Tensor(std::vector<int> dims);
    Tensor(std::vector<complex> data, std::vector<int> dims);

    int getID();

    std::vector<int> getDims();
    int getSize();
    complex getEl(std::vector<int> coords);
    void setEl(std::vector<int> coords, complex value);

private:
    static unsigned int counter;
    int unsigned id;

    std::vector<complex> data;
    std::vector<int> dims;
};

class Bond {
public:
    Bond(std::pair<Tensor, Tensor> tensors, std::pair<int, int> dims);

    int getID();

    std::pair<Tensor, Tensor> getTensors();
    std::pair<int, int> getDims();
    int getSize();

private:
    static unsigned int counter;
    int unsigned id;

    std::pair<Tensor, Tensor> tensors;
    std::pair<int, int> dims;
};

class TensorNetwork {
public:
    TensorNetwork(std::vector<Tensor> tensors, std::vector<Bond> bonds);

    Tensor getTensor(int n);
    void contract();

private:
    std::vector<Tensor> tensors;
    std::vector<Bond> bonds;
};

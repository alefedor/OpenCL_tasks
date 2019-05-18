#define __CL_ENABLE_EXCEPTIONS

#include <CL/cl.h>
#include "cl.hpp"

#include <vector>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <iterator>

using std::vector;
using std::istream;

struct matrix {
    int size;
    double *data;

    matrix(int n, double value = 0): size(n) {
        data = new double[size * size];
        for (int i = 0; i < size * size; i++)
            data[i] = 0;
    }

    matrix(matrix const& other) : matrix(other.size) {
        for (int i = 0; i < size * size; i++)
            data[i] = other.data[i];
    }

    ~matrix() {
        delete[] data;
    }
};

matrix read_matrix(int n, istream &in) {
    matrix result(n);
    for (int i = 0; i < n * n; i++)
        in >> result.data[i];
    return result;
}

int div_rounded_up(int a, int b) {
    return (a + b - 1) / b;
}

int round_up(int n, int block_size) {
    return div_rounded_up(n, block_size) * block_size;
}

int main() {
#ifndef LOCAL
    std::freopen("input.txt", "r", stdin);
    std::freopen("output.txt", "w", stdout);
#endif

    int n, m;
    std::cin >> n >> m;

    auto A = read_matrix(n, std::cin);
    auto B = read_matrix(m, std::cin);
    auto C = matrix(n);

    std::vector<cl::Platform> platforms;
    std::vector<cl::Device> devices;
    std::vector<cl::Kernel> kernels;

    try {
        // create platform
        cl::Platform::get(&platforms);
        platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);


        // create context
        cl::Context context(devices);

        // create command queue
        cl::CommandQueue queue(context, devices[0]);

        // load opencl source
        std::ifstream cl_file("convolution.cl");
        std::string cl_string(std::istreambuf_iterator<char>(cl_file), (std::istreambuf_iterator<char>()));
        cl::Program::Sources source(1, std::make_pair(cl_string.c_str(),
                                                      cl_string.length() + 1));
        // create program
        cl::Program program(context, source);

        // compile opencl source
        size_t const block_size = 16;
        program.build(devices, ("-D BLOCK_SIZE=" + std::to_string(block_size)).c_str());

        // create a message to send to kernel
        size_t const A_size = n * n;
        size_t const B_size = m * m;

        // allocate device buffer to hold message
        cl::Buffer dev_A(context, CL_MEM_READ_ONLY, sizeof(double) * A_size);
        cl::Buffer dev_B(context, CL_MEM_READ_ONLY, sizeof(double) * B_size);
        cl::Buffer dev_C(context, CL_MEM_WRITE_ONLY, sizeof(double) * A_size);

        // copy from cpu to gpu
        queue.enqueueWriteBuffer(dev_A, CL_TRUE, 0, sizeof(double) * A_size, A.data);
        queue.enqueueWriteBuffer(dev_B, CL_TRUE, 0, sizeof(double) * B_size, B.data);

        int rounded_n = round_up(n, block_size);

        cl::Kernel kernel(program, "convolution");
        cl::KernelFunctor convolve_functor(kernel, queue, cl::NullRange,
                                                      cl::NDRange(rounded_n, rounded_n), cl::NDRange(block_size, block_size));
        convolve_functor(dev_A, dev_B, dev_C, n, m);

        queue.enqueueReadBuffer(dev_C, CL_TRUE, 0, sizeof(double) * A_size, C.data);

        std::cout << std::fixed << std::setprecision(3);

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++)
                std::cout << C.data[i * n + j] << " ";
            std::cout << "\n";
        }
    }
    catch (cl::Error e) {
        std::cout << std::endl << e.what() << " : " << e.err() << std::endl;
    }

    return 0;
}
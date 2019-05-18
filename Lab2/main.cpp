#define __CL_ENABLE_EXCEPTIONS

#include <CL/cl.h>
#include "cl.hpp"

#include <vector>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <iterator>
#include <random>

using std::vector;
using std::istream;

size_t const BLOCK_SIZE = 256;


struct array {
    int size;
    double *data;

    explicit array(int n): size(n) {
        data = new double[size];
        memset(data, 0, sizeof(double) * size);
    }

    array(array const& other) : array(other.size) {
        for (int i = 0; i < size; i++)
            data[i] = other.data[i];
    }

    ~array() {
        delete[] data;
    }
};

array read_array(istream &in) {
    int n;
    in >> n;
    array result(n);
    for (int i = 0; i < n; i++)
        in >> result.data[i];
    return result;
}

int div_rounded_up(int a, int b) {
    return (a + b - 1) / b;
}

int round_up(int n, int block_size) {
    return div_rounded_up(n, block_size) * block_size;
}

/**
 * Copies ends of each block.
 */
void parallel_partial_copy(array const& from, array &to, cl::Context &context, cl::Program &program, cl::CommandQueue &queue) {
    cl::Buffer dev_input(context, CL_MEM_READ_ONLY, sizeof(double) * from.size);
    cl::Buffer dev_output(context, CL_MEM_WRITE_ONLY, sizeof(double) * to.size);

    queue.enqueueWriteBuffer(dev_input, CL_TRUE, 0, sizeof(double) * from.size, from.data);

    int rounded_size = round_up(from.size, BLOCK_SIZE);

    cl::Kernel kernel(program, "partial_copy");
    cl::KernelFunctor convolve_functor(kernel, queue, cl::NullRange, cl::NDRange(rounded_size), cl::NDRange(BLOCK_SIZE));
    convolve_functor(dev_input, dev_output, from.size, to.size);

    queue.enqueueReadBuffer(dev_output, CL_TRUE, 0, sizeof(double) * to.size, to.data);
    to.data[0] = 0.0; // was not initialized before
}

/**
 * Adds elements from first array to corresponding blocks of second array
 */
void parallel_block_add(array const& from, array &to, cl::Context &context, cl::Program &program, cl::CommandQueue &queue) {
    cl::Buffer dev_input_partial(context, CL_MEM_READ_ONLY, sizeof(double) * from.size);
    cl::Buffer dev_input(context, CL_MEM_READ_ONLY, sizeof(double) * to.size);
    cl::Buffer dev_output(context, CL_MEM_WRITE_ONLY, sizeof(double) * to.size);

    queue.enqueueWriteBuffer(dev_input_partial, CL_TRUE, 0, sizeof(double) * from.size, from.data);
    queue.enqueueWriteBuffer(dev_input, CL_TRUE, 0, sizeof(double) * to.size, to.data);

    int rounded_size = round_up(to.size, BLOCK_SIZE);

    cl::Kernel kernel(program, "block_add");
    cl::KernelFunctor convolve_functor(kernel, queue, cl::NullRange, cl::NDRange(rounded_size), cl::NDRange(BLOCK_SIZE));
    convolve_functor(dev_input_partial, dev_input, dev_output, to.size);

    queue.enqueueReadBuffer(dev_output, CL_TRUE, 0, sizeof(double) * to.size, to.data);
}

/**
 * Calculates prefix sum for arr. Result is in arr.
 */
void parallel_prefix_sum(array &arr, cl::Context &context, cl::Program &program, cl::CommandQueue &queue) {
    cl::Buffer dev_input(context, CL_MEM_READ_ONLY, sizeof(double) * arr.size);
    cl::Buffer dev_output(context, CL_MEM_WRITE_ONLY, sizeof(double) * arr.size);

    queue.enqueueWriteBuffer(dev_input, CL_TRUE, 0, sizeof(double) * arr.size, arr.data);

    int rounded_size = round_up(arr.size, BLOCK_SIZE);

    cl::Kernel kernel(program, "block_prefix_sum");
    cl::KernelFunctor convolve_functor(kernel, queue, cl::NullRange, cl::NDRange(rounded_size), cl::NDRange(BLOCK_SIZE));
    convolve_functor(dev_input, dev_output, cl::__local(sizeof(double) * BLOCK_SIZE), cl::__local(sizeof(double) * BLOCK_SIZE), arr.size);

    queue.enqueueReadBuffer(dev_output, CL_TRUE, 0, sizeof(double) * arr.size, arr.data);

    if (arr.size > BLOCK_SIZE) {
        auto sums = array(div_rounded_up(arr.size, BLOCK_SIZE));
        parallel_partial_copy(arr, sums, context, program, queue);
        parallel_prefix_sum(sums, context, program, queue);
        parallel_block_add(sums, arr, context, program, queue);
    }
}

int main() {
#ifndef LOCAL
    std::freopen("input.txt", "r", stdin);
    std::freopen("output.txt", "w", stdout);
#endif

    auto arr = read_array(std::cin);

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
        std::ifstream cl_file("prefix_sum.cl");
        std::string cl_string(std::istreambuf_iterator<char>(cl_file), (std::istreambuf_iterator<char>()));
        cl::Program::Sources source(1, std::make_pair(cl_string.c_str(),
                                                      cl_string.length() + 1));
        // create program
        cl::Program program(context, source);

        program.build(devices);

        parallel_prefix_sum(arr, context, program, queue);
    }
    catch (cl::Error &e) {
        std::cout << std::endl << e.what() << " : " << e.err() << std::endl;
    }

    std::cout << std::fixed << std::setprecision(3);

    for (int i = 0; i < arr.size; i++)
        std::cout << arr.data[i] << " ";

    return 0;
}
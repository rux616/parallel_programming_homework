#include <cstddef>
#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include "image_kernels.h"
#include "utility.h"
using namespace std;

const string SIGNATURE_PGM_BINARY = "P5";

// A small selection of kernels to use.
// const image_kernels::image_kernel_3 & KERNEL = image_kernels::BOX_BLUR_3; const uint32_t KERNEL_SIZE = image_kernels::KERNEL_SIZE_3;
// const image_kernels::image_kernel_5 & KERNEL = image_kernels::BOX_BLUR_5; const uint32_t KERNEL_SIZE = image_kernels::KERNEL_SIZE_5;
const image_kernels::image_kernel_91 & KERNEL = image_kernels::BOX_BLUR_91; const uint32_t KERNEL_SIZE = image_kernels::KERNEL_SIZE_91;

// Declare the offset to be used with the kernel.
const uint32_t KERNEL_OFFSET = KERNEL_SIZE / 2;

int main(int32_t argc, const char * argv[])
{
    uint8_t ** input_data = nullptr, ** output_data = nullptr;

    uint32_t height = 0, width = 0, depth = 0;
    uint32_t to_return = 0;
    uint32_t number_of_threads = 0;
    string input_filename, output_filename, signature;
    fstream input_stream, output_stream;

    chrono::steady_clock::time_point timer_begin, timer_end;
    chrono::duration<double> timer_duration;

    // Determine the action to take based upon the number of parameters.
    switch (argc)
    {
        case 4:
            // Read and process, then write the result back to disk.
            output_filename = argv[3];

        case 3:
            // Read and process, but don't write.
            input_filename = argv[2];
            number_of_threads = stoi(argv[1]);
            break;

        default:
            // No arguments, or more than two arguments.  Spit out the syntax.
            cerr << "Syntax: parallel-convolution.exe <num_threads> <input_filename> [output_filename]\n";
            to_return = 1;
            goto exit_program; // That's right, I'm using goto. Why? Because I can, really. It's good to switch things up occasionally.
    }

    cout << "Number of threads: " << number_of_threads << "\n";
    cout << "Kernel size: " << KERNEL_SIZE << "x" << KERNEL_SIZE << "\n";

    // Attempt to open the specified file for binary input.
    input_stream.open(input_filename, ios_base::in | ios_base::binary);
    if (input_stream.fail())
    {
        cerr << "Something went wrong when opening the requested input file.\n";
        to_return = 1;
        goto exit_program;
    }

    // Get the first line of the PGM, its "magic number."
    getline(input_stream, signature);
    if (signature != SIGNATURE_PGM_BINARY)
    {
        cerr << "The input file is not a binary PGM.\n";
        to_return = 1;
        goto exit_program;
    }

    // Read the width, height, and depth from the input file.
    input_stream >> width >> height >> depth;

    // Eat the next character, which should be '\n'.
    input_stream.get();
    
    // Check the width, height, and depth to make sure they are valid.
    if (width == 0 || height == 0 || depth == 0)
    {
        cerr << "Invalid dimensions or depth in input file.\n";
        to_return = 1;
        goto exit_program;
    }
    cout << "Image dimensions: " << width << "x" << height << "\n";

    // Create the array to read the file into.
    input_data = utility::create_array_2d<uint8_t>(height + KERNEL_SIZE - 1, width + KERNEL_SIZE - 1);

    // Read the input file, then close the input stream.
    for (uint32_t row = KERNEL_OFFSET; row < height + KERNEL_OFFSET; row++)
        input_stream.read((char *)&(input_data[row][KERNEL_OFFSET]), width);
    input_stream.close();

    // Extend the input array.
    timer_begin = chrono::steady_clock::now();
    utility::extend_array(input_data, height, width, KERNEL_SIZE);
    timer_end = chrono::steady_clock::now();
    timer_duration = chrono::duration_cast<chrono::duration<double>>(timer_end - timer_begin);
    cout << "Array extension took " << timer_duration.count() << " second(s).\n";

    // Create the array to write the modified data into.
    output_data = utility::create_array_2d<uint8_t>(height, width);

    // Do the convolutions, and time it.
    timer_begin = chrono::steady_clock::now();
    #pragma omp parallel for num_threads(number_of_threads) default(none) \
        private(data_row, data_column, kernel_row, kernel_column, accumulator) \
        shared(output_data)
    for (uint32_t data_row = KERNEL_OFFSET; data_row < height + KERNEL_OFFSET; data_row++)
    {
        for (uint32_t data_column = KERNEL_OFFSET; data_column < width + KERNEL_OFFSET; data_column++)
        {
            uint32_t accumulator = 0;

            for (uint32_t kernel_row = 0; kernel_row < KERNEL_SIZE; kernel_row++)
            {
                for (uint32_t kernel_column = 0; kernel_column < KERNEL_SIZE; kernel_column++)
                {
                    accumulator += input_data[data_row + kernel_row - KERNEL_OFFSET]
                                             [data_column + kernel_column - KERNEL_OFFSET] * 
                                   KERNEL[kernel_row][kernel_column];
                }
            }

            output_data[data_row - KERNEL_OFFSET][data_column - KERNEL_OFFSET] = accumulator / (KERNEL_SIZE * KERNEL_SIZE);
        }
    }
    timer_end = chrono::steady_clock::now();

    // Calculate and display the time taken specifically for the convolutions.
    timer_duration = chrono::duration_cast<chrono::duration<double>>(timer_end - timer_begin);
    cout << "Convolutions took " << timer_duration.count() << " second(s).\n\n";

    // If an output file is specified, attempt to open it for binary output.
    if (output_filename.empty() == false)
    {
        output_stream.open(output_filename, ios_base::out | ios_base::binary);
        if (output_stream.fail())
        {
            cerr << "Something went wrong when opening the requested output file.\n";
            to_return = 1;
            goto exit_program;
        }

        output_stream << signature << "\n";
        output_stream << width << " " << height << "\n";
        output_stream << depth << "\n";
        for (uint32_t row = 0; row < height; row++)
            output_stream.write((char *)(output_data[row]), width);
    }

    // Clean up the file streams and arrays then exit.
exit_program:
    if (input_stream.is_open())
        input_stream.close();
    if (output_stream.is_open())
        output_stream.close();
    if (input_data != nullptr)
        utility::delete_array_2d(input_data);
    if (output_data != nullptr)
        utility::delete_array_2d(output_data);
    return to_return;
}

#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include "image_kernels.h"
#include "utility.h"

const int DEFAULT_NUM_CHARS = 80;
const std::string MAGIC_NUMBER_PBM_BINARY = "P4";
const std::string MAGIC_NUMBER_PGM_BINARY = "P5";
const std::string MAGIC_NUMBER_PPM_BINARY = "P6";

int main(int argc, const char * argv[])
{
    char * input_data = nullptr, * output_data = nullptr;

    int width = -1, height = -1, depth = -1, to_return = 0;
    std::string input_filename, output_filename, scratch;
    std::fstream input_stream, output_stream;

    // Determine the actions to take based upon the number of parameters.
    switch (argc)
    {
        case 3:
            // Two arguments.  Read and process, then write the result back to disk.
            output_filename = argv[2];

        case 2:
            // One argument.  Read and process, but don't write.
            input_filename = argv[1];
            break;

        default:
            // No arguments, or more than two arguments.  Spit out the syntax.
            std::cerr << "Syntax: parallel-convolution.exe <input_filename> [output_filename]\n";
            to_return = 1;
            goto exit_program; // That's right, I'm using goto. Why? Because I can, really. It's good to switch things up occasionally.
    }

    // Attempt to open the specified file for binary input.
    input_stream.open(input_filename, std::ios_base::in | std::ios_base::binary);
    if (input_stream.fail())
    {
        std::cerr << "Something went wrong when opening the requested input file.\n";
        to_return = 1;
        goto exit_program;
    }

    // Get the first line of the PGM, its "magic number."
    std::getline(input_stream, scratch);
    if (scratch != MAGIC_NUMBER_PBM_BINARY &&
        scratch != MAGIC_NUMBER_PGM_BINARY &&
        scratch != MAGIC_NUMBER_PPM_BINARY)
    {
        std::cerr << "The input file is not a recognized PGM.\n";
        to_return = 1;
        goto exit_program;
    }

    // Read the width, height, and depth from the input file.
    input_stream >> width;
    input_stream >> height;
    input_stream >> depth;

    // Check the width, height, and depth to make sure they are valid.
    if (width <= 0 || height <= 0 || depth <= 0)
    {
        std::cerr << "Invalid dimensions or depth in input file.\n";
        to_return = 1;
        goto exit_program;
    }

    // TODO Extend the array so we can handle the corners and edges.
    // Create the array to read the file into, then check it to make sure it allocated successfully.
    input_data = new char[height * width];
    if (input_data == nullptr)
    {
        std::cerr << "Error allocating memory for the input file's data.\n";
        to_return = 1;
        goto exit_program;
    }

    // Read the input file, then close the input stream.
    input_stream.read(input_data, height * width);
    input_stream.close();

    // Create the array to write the modified data into, then check to make sure it allocated successfully.
    output_data = new char[height * width];
    if (input_data == nullptr)
    {
        std::cerr << "Error allocating memory for the output data.\n";
        to_return = 1;
        goto exit_program;
    }

    // Do the convolutions.
    // TODO Wrap this thing in a timer.
    // TODO Parallelize.
    for (int data_row = 0; data_row < height; data_row++)
    {
        for (int data_column = 0; data_column < width; data_column++)
        {
            int accumulator = 0;

            for (int kernel_row = 0; kernel_row < KERNEL_SIZE; kernel_row++)
            {
                for (int kernel_column = 0; kernel_column < KERNEL_SIZE; kernel_column++)
                {
                    // multiply
                    // add
                }
            }

            output_data[utility::translate_2d_to_1d(data_row, data_column, width)] = accumulator;
        }
    }

    // If an output file is specified, attempt to open it for binary output.
    if (output_filename.empty() == false)
    {
        output_stream.open(output_filename);
        // TODO Finish output file writing.
    }

    // Clean up the file streams and arrays then exit.
exit_program:
    if (input_stream.is_open())
        input_stream.close();
    if (output_stream.is_open())
        output_stream.close();
    if (input_data != nullptr)
        delete[] input_data;
    if (output_data != nullptr)
        delete[] output_data;
    return to_return;
}

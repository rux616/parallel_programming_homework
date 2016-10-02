#include <cstddef>
#include <cstdio>
#include <chrono>
#include <fstream>
#include <omp.h>
#include <string>
#include "utility.h"
using namespace std;

const uint32_t KERNEL_FILL_VALUE = 1;
const string SIGNATURE_PGM_BINARY = "P5";

int main(int32_t argc, const char * argv[])
{
    uint8_t ** input_data = nullptr, ** output_data = nullptr;
    uint32_t ** kernel = nullptr;

    int32_t height = 0, width = 0, depth = 0;
    int32_t kernel_size = 0, kernel_offset = 0;

    uint32_t number_of_threads = 0;

    string input_filename, output_filename, signature;
    fstream input_stream, output_stream;

    uint32_t to_return = 0;

    chrono::steady_clock::time_point timer_begin, timer_end;
    chrono::duration<double> timer_duration;


    // Determine the action to take based upon the number of parameters.
    switch (argc)
    {
        // Read and process, then write the result back to disk.
        case 5:
            output_filename = argv[4];
            // Utilize fall-through to read the rest of the command line arguments.

        // Read and process, but don't write.
        case 4:
            input_filename = argv[3];
            kernel_size = stoi(argv[2]);
            number_of_threads = stoi(argv[1]);
            kernel_offset = kernel_size / 2;
            break;

        // No arguments, or more than the max number of arguments.  Spit out the syntax.
        default:
            fprintf(stderr, "Syntax: parallel-convolution.exe <num_threads> <kernel_size> <input_filename> [output_filename]\n");
            to_return = 1;
            goto exit_program; // Using goto to jump to a central resource deallocation area.
    }


    // Create and fill the kernel.
    kernel = utility::create_array_2d<uint32_t>(kernel_size, kernel_size);
    utility::fill_array_2d(kernel, kernel_size, kernel_size, KERNEL_FILL_VALUE);
    printf("Kernel size: %dx%d\n", kernel_size, kernel_size);


    // Attempt to open the specified file for binary input.
    input_stream.open(input_filename, ios_base::in | ios_base::binary);
    if (input_stream.fail())
    {
        fprintf(stderr, "Something went wrong when opening the requested input file.\n");
        to_return = 1;
        goto exit_program;
    }


    // Get the first line of the PGM, its "magic number."
    getline(input_stream, signature);
    if (signature != SIGNATURE_PGM_BINARY)
    {
        fprintf(stderr, "The input file is not a binary PGM.\n");
        to_return = 1;
        goto exit_program;
    }


    // Read the width, height, and depth from the input file.
    input_stream >> width >> height >> depth;


    // Eat the next character, which should be '\n'.
    input_stream.get();
    

    // Check the width, height, and depth to make sure they are valid.
    if (width <= 0 || height <= 0 || depth <= 0)
    {
        fprintf(stderr, "Invalid dimensions or depth in input file.\n");
        to_return = 1;
        goto exit_program;
    }
    printf("Image dimensions: %dx%d", width, height);
    
    // Create the array to read the file into.
    input_data = utility::create_array_2d<uint8_t>(height + kernel_size - 1, width + kernel_size - 1);


    // Read the input file, then close the input stream.
    for (int32_t row = kernel_offset; row < height + kernel_offset; row++)
        input_stream.read((char *)&(input_data[row][kernel_offset]), width);
    input_stream.close();


    // Extend the input array.
    utility::extend_array(input_data, height, width, kernel_size);


    // Create the array to write the modified data into.
    output_data = utility::create_array_2d<uint8_t>(height, width);


    // Do the convolutions, and time it.
    printf("Number of threads requested: %d\n", number_of_threads);
    timer_begin = chrono::steady_clock::now();
    #pragma omp parallel num_threads(number_of_threads) default(none) \
        shared(height, width, input_data, output_data, kernel, kernel_size, kernel_offset)
    {
        #pragma omp master
        printf("Number of threads created: %d\n", omp_get_num_threads());
        
        #pragma omp for
        // Iterate through the full input data.
        for (int32_t data_row = kernel_offset; data_row < height + kernel_offset; data_row++)
        {
            for (int32_t data_column = kernel_offset; data_column < width + kernel_offset; data_column++)
            {
                int32_t sum = 0;

                // Iterate through the full kernel.
                for (int32_t kernel_row = 0; kernel_row < kernel_size; kernel_row++)
                {
                    for (int32_t kernel_column = 0; kernel_column < kernel_size; kernel_column++)
                    {
                        // Multiply each data element by its corresponding kernel element and add it
                        // to a running sum.
                        sum += input_data[data_row + kernel_row - kernel_offset]
                            [data_column + kernel_column - kernel_offset] *
                            kernel[kernel_row][kernel_column];
                    }
                }

                // Once the summation is done, store that value in the output array.
                output_data[data_row - kernel_offset][data_column - kernel_offset] = sum / (kernel_size * kernel_size);
            }
        }
    }
    timer_end = chrono::steady_clock::now();
    

    // Calculate and display the time taken specifically for the convolutions.
    timer_duration = chrono::duration_cast<chrono::duration<double>>(timer_end - timer_begin);
    printf("Convolutions took %f second(s).\n\n", timer_duration.count());


    // If an output file is specified, attempt to open it for binary output.
    if (output_filename.empty() == false)
    {
        // Open the file.
        output_stream.open(output_filename, ios_base::out | ios_base::binary);
        // Verify that the file was opened.
        if (output_stream.fail())
        {
            fprintf(stderr, "Something went wrong when opening the requested output file.\n");
            to_return = 1;
            goto exit_program;
        }

        // Write the PGM header.
        output_stream << signature << "\n";
        output_stream << width << " " << height << "\n";
        output_stream << depth << "\n";
        // Write the PGM data.
        for (int32_t row = 0; row < height; row++)
            output_stream.write((char *)(output_data[row]), width);
    }


    // Clean up the file streams and arrays then exit.
exit_program:
    if (input_stream.is_open())
        input_stream.close();
    if (output_stream.is_open())
        output_stream.close();
    if (kernel != nullptr)
        utility::delete_array_2d(kernel);
    if (input_data != nullptr)
        utility::delete_array_2d(input_data);
    if (output_data != nullptr)
        utility::delete_array_2d(output_data);
    return to_return;
}

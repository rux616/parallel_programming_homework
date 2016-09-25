#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>

int main(std::int32_t argc, const char * argv[])
{
    std::int32_t ** base_data = nullptr, ** modified_data = nullptr;

    std::int16_t width, height, depth;
    std::string input_filename, output_filename;
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
            std::cout << "Syntax: parallel-convolution.exe <input_filename> [output_filename]\n";
            return 1;
    }






    return 0;
}
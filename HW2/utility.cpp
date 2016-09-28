#include "utility.h"

// Read a location in a 2D array which is mapped to a 1D array.
inline int utility::translate_2d_to_1d(const int row, const int column, const int width)
{
    return (row * width) + column;
}

// Extend an array for kernel convolution.  Assumes that the kernel is an odd height and width.
void utility::extend_array(char * array_to_extend,
                           const int extended_height, const int extended_width,
                           const int current_height, const int current_width,
                           const int current_row, const int current_column)
{

    return;
}
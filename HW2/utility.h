#pragma once
#ifndef UTILITY_H
#define UTILITY_H

#include <cstdint>
#include <cstring>
using namespace std;

namespace utility
{

// Extend an array for kernel convolution.  Assumes that the kernel is an odd height and width.
template <typename T>
void extend_array(T ** array_to_extend, const int32_t height, const int32_t width, const int32_t kernel_size)
{
    int32_t kernel_offset = kernel_size / 2;

    // Extend existing data left.
    for (int32_t row = kernel_offset; row < height + kernel_offset - 1; row++)
    {
        for (int32_t column = kernel_offset; column > 0; column--)
        {
            array_to_extend[row][column - 1] = array_to_extend[row][column];
        }
    }

    // Extend existing data right.
    for (int32_t row = kernel_offset; row < height + kernel_offset - 1; row++)
    {
        for (int32_t column = width + kernel_offset - 1; column < width + kernel_size - 2; column++)
        {
            array_to_extend[row][column + 1] = array_to_extend[row][column];
        }
    }

    // Extend all data upwards.
    for (int32_t row = kernel_offset; row > 0; row--)
    {
        memcpy(array_to_extend[row], array_to_extend[row - 1], (width + kernel_size - 1) * sizeof(T));
    }

    // Extend all data downwards.
    for (int32_t row = height + kernel_offset - 1; row < height + kernel_size - 2; row++)
    {
        memcpy(array_to_extend[row], array_to_extend[row + 1], (width + kernel_size - 1) * sizeof(T));
    }
}

// Taken from http://stackoverflow.com/a/21944048
// Create a 2 dimensional array of contiguous data.
template <typename T>
T ** create_array_2d(int32_t height, int32_t width)
{
    // Allocate pointers.
    T ** ptr = new T*[height];

    // Allocate pool.
    T * pool = new T[height * width];

    // Assign each "row" in the pool to a pointer.
    for (int32_t i = 0; i < height; ++i, pool += width)
    {
        ptr[i] = pool;
    }

    return ptr;
}

// Delete a 2 dimensional array.
template <typename T>
void delete_array_2d(T ** arr)
{
    delete[] arr[0];  // remove the pool
    delete[] arr;     // remove the pointers
}

// Fill a 2 dimensional array.  For safety, this should really only be used with simple data types.
template <typename T>
void fill_array_2d(T ** arr, int32_t height, int32_t width, T fill_data)
{
    for (int32_t row = 0; row < height; row++)
    {
        for (int32_t column = 0; column < width; column++)
        {
            arr[row][column] = fill_data;
        }
    }
}

} // end namespace utility

#endif
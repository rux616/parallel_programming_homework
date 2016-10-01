#pragma once
#ifndef UTILITY_H
#define UTILITY_H

#include <cstdint>
using namespace std;

namespace utility
{

// Extend an array for kernel convolution.  Assumes that the kernel is an odd height and width.
template <typename T>
void extend_array(T ** array_to_extend, const uint32_t height, const uint32_t width,
                           const uint32_t kernel_size)
{
    const uint32_t KERNEL_OFFSET = kernel_size / 2;

    // Extend existing data left.
    for (uint32_t row = KERNEL_OFFSET; row < height + KERNEL_OFFSET - 1; row++)
    {
        for (uint32_t column = KERNEL_OFFSET; column > 0; column--)
        {
            array_to_extend[row][column - 1] = array_to_extend[row][column];
        }
    }

    // Extend existing data right.
    for (uint32_t row = KERNEL_OFFSET; row < height + KERNEL_OFFSET - 1; row++)
    {
        for (uint32_t column = width + KERNEL_OFFSET - 1; column < width + kernel_size - 2; column++)
        {
            array_to_extend[row][column + 1] = array_to_extend[row][column];
        }
    }

    // Extend all data upwards.
    for (uint32_t row = KERNEL_OFFSET; row > 0; row--)
    {
        for (uint32_t column = 0; column < width + kernel_size - 1; column++)
        {
            array_to_extend[row - 1][column] = array_to_extend[row][column];
        }
    }

    // Extend all data downwards.
    for (uint32_t row = height + KERNEL_OFFSET - 1; row < height + kernel_size - 2; row++)
    {
        for (uint32_t column = 0; column < width + kernel_size - 1; column++)
        {
            array_to_extend[row + 1][column] = array_to_extend[row][column];
        }
    }

    //// Extend existing data upwards.
    //for (uint32_t row = KERNEL_OFFSET; row > 0; row--)
    //{
    //    for (uint32_t column = KERNEL_OFFSET; column < width + KERNEL_OFFSET; column++)
    //    {
    //        array_to_extend[row - 1][column] = array_to_extend[row][column];
    //    }
    //}

    //// Extend existing data downwards.
    //for (uint32_t row = height + KERNEL_OFFSET - 1; row < height + kernel_size - 2; row++)
    //{
    //    for (uint32_t column = KERNEL_OFFSET; column < width + KERNEL_OFFSET; column++)
    //    {
    //        array_to_extend[row + 1][column] = array_to_extend[row][column];
    //    }
    //}

    //// Extend all data left.
    //for (uint32_t row = 0; row < height + kernel_size - 1; row++)
    //{
    //    for (uint32_t column = KERNEL_OFFSET; column > 0; column--)
    //    {
    //        array_to_extend[row][column - 1] = array_to_extend[row][column];
    //    }
    //}

    //// Extend all data right.
    //for (uint32_t row = 0; row < height + kernel_size - 1; row++)
    //{
    //    for (uint32_t column = width + KERNEL_OFFSET - 1; column < width + kernel_size - 2; column++)
    //    {
    //        array_to_extend[row][column + 1] = array_to_extend[row][column];
    //    }
    //}

    return;
}

// Taken from http://stackoverflow.com/a/21944048
// Create a 2 dimensional array of contiguous data.
template <typename T>
T ** create_array_2d(uint32_t height, uint32_t width)
{
    // Allocate pointers.
    T ** ptr = new T*[height];

    // Allocate pool.
    T * pool = new T[height * width];

    // Assign each "row" in the pool to a pointer.
    for (uint32_t i = 0; i < height; ++i, pool += width)
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

} // end namespace utility

#endif
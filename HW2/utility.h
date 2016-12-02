/*--------------------------------------------------------------------------------------------------
 * Author:      Dan Cassidy
 * Date:        2016-10-04
 * Assignment:  Homework 2
 * Source File: utility.h
 * Language:    C/C++
 * Course:      CSCI-B 424 Parallel Programming
 * Purpose:     Contains some utility template functions to make my life easier.
--------------------------------------------------------------------------------------------------*/

#pragma once
#ifndef UTILITY_H
#define UTILITY_H

#include <cstdint>
#include <cstring>
using namespace std;

namespace utility
{

/*--------------------------------------------------------------------------------------------------
 * Name:    extend_array
 * Purpose: Copies the edge data of an sub 2d array out to the edges of the containing 2d array.
 * Input:   T ** array_to_extend, holds the pointer to the array which will be worked on.
 * Input:   const int32_t height, holds the height of the sub array.
 * Input:   const int32_t width, holds the width of the sub array.
 * Input:   const int32_t kernel_size, holds the size of the kernel that will be ultimately be used
 *          on the image.
 * Output:  T ** array_to_extend.
--------------------------------------------------------------------------------------------------*/
template <typename T>
void extend_array(T ** array_to_extend, const int32_t height, const int32_t width,
                  const int32_t kernel_size)
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
        memcpy(array_to_extend[row], array_to_extend[row - 1], (width + kernel_size - 1) *
               sizeof(T));
    }

    // Extend all data downwards.
    for (int32_t row = height + kernel_offset - 1; row < height + kernel_size - 2; row++)
    {
        memcpy(array_to_extend[row], array_to_extend[row + 1], (width + kernel_size - 1) *
               sizeof(T));
    }
}



/*--------------------------------------------------------------------------------------------------
 * Name:    create_array_2d
 * Purpose: Creates a 2 dimensional array of contiguous data.
 * Input:   int32_t height, holds the number of rows the array should have.
 * Input:   int32_t width, holds the number of columns the array should have.
 * Output:  T **, holds a pointer to the newly created 2 dimensional array.
 *
 * Note:    Taken from http://stackoverflow.com/a/21944048.
--------------------------------------------------------------------------------------------------*/
template <typename T>
T ** create_array_2d(int32_t height, int32_t width)
{
    // Allocate pointers.
    T ** new_array_pointer = new T*[height];

    // Allocate pool.
    T * pool = new T[height * width];

    // Assign each "row" in the pool to a pointer.
    for (int32_t i = 0; i < height; ++i, pool += width)
    {
        new_array_pointer[i] = pool;
    }

    return new_array_pointer;
}



/*--------------------------------------------------------------------------------------------------
 * Name:    delete_array_2d
 * Purpose: Deletes a 2 dimensional array.
 * Input:   T ** array_to_delete, holds the pointer to the 2 dimensional array that shall be
 *          deleted.
 * Output:  Nothing.
 *
 * Note:    Taken from http://stackoverflow.com/a/21944048.
--------------------------------------------------------------------------------------------------*/
template <typename T>
void delete_array_2d(T ** array_to_delete)
{
    delete[] array_to_delete[0];  // remove the pool
    delete[] array_to_delete;     // remove the pointers
}



/*--------------------------------------------------------------------------------------------------
 * Name:    fill_array_2d
 * Purpose: Fills a 2 dimensional array.  For safety, this should really only be used with primitive
 *          data types.
 * Input:   T ** array_to_fill, holds the pointer to the array which will be filled.
 * Input:   int32_t height, holds the height of the array.
 * Input:   int32_t width, holds the width of the array.
 * Input:   T fill_data, holds the data with which the array shall be filled.
 * Output:  T ** array_to_fill.
--------------------------------------------------------------------------------------------------*/
template <typename T>
void fill_array_2d(T ** array_to_fill, int32_t height, int32_t width, T fill_data)
{
    for (int32_t row = 0; row < height; row++)
    {
        for (int32_t column = 0; column < width; column++)
        {
            array_to_fill[row][column] = fill_data;
        }
    }
}

} // end namespace utility

#endif
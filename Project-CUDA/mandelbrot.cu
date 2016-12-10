/***************************************************************************************************
 * Mandelbrotset.c
 * Copyright Shibin K.Reeny
 * This program is free software; you can redistribute it and/or modify it under the terms of the
 * GNU General Public License as published by the Free Software Foundation; either version 2 of
 * the License, or (at your option) any later version.
 **************************************************************************************************/

#include <GL/gl.h>
#include <GL/glut.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#include <chrono>
#else
#include <ctime>
#endif
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
using namespace std;



// Define an RGB struct to represent the color of a pixel.
struct rgb
{
    float r;
    float g;
    float b;
};



//-------------------
// General Constants
//-------------------
const unsigned int PATTERN_SIZE = 1000;
const float X_RANGE_START = -2.5f;
const float X_RANGE_END = 1.1f;
const float Y_RANGE_START = -1.0f;
const float Y_RANGE_END = 1.1f;
// Default image size.
const unsigned int DEFAULT_IMAGE_WIDTH = 1440;
const unsigned int DEFAULT_IMAGE_HEIGHT = 840;
// Default number of iterations.
const unsigned int DEFAULT_NUM_ITERATIONS = 1000;
// CUDA
const unsigned int DEFAULT_NUM_CUDA_BLOCKS = 1;
const unsigned int DEFAULT_NUM_CUDA_THREADS_PER_BLOCK = 32;

//-----------------
// General Globals
//-----------------
unsigned int image_width;
unsigned int image_height;
unsigned int num_iterations;
float x_increment;
float y_increment;
// Contains the colors of the pixels on the host.
rgb * h_pixels = nullptr;

//---------------------------------
// Implementation-Specific Globals
//---------------------------------
// CUDA
unsigned int num_cuda_blocks;
unsigned int num_cuda_threads_per_block;



// Initialize the pixels array on the GPU.
__global__ void init_pixels_kernel(const unsigned int num_pixels, rgb * d_pixels)
{
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < num_pixels;
         i += blockDim.x * gridDim.x)
    {
        d_pixels[i].r = 1.0f;
        d_pixels[i].g = 1.0f;
        d_pixels[i].b = 1.0f;
    }
}

// Initialize the pattern array on the GPU.
__global__ void init_pattern_kernel(const unsigned int pattern_size, rgb * d_pattern)
{
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < pattern_size;
         i += blockDim.x * gridDim.x)
    {
        if (i > 729)
        {
            d_pattern[i].r = 1.0f;
            d_pattern[i].g = 1.0f;
            d_pattern[i].b = 1.0f;
        }
        else
        {
            d_pattern[i].r = 0.1f + (i % 9) * 0.1f;
            d_pattern[i].g = 0.1f + (i / 81) * 0.1f;
            d_pattern[i].b = 0.1f + ((i / 9) % 9) * 0.1f;
        }
    }
}

// Generate a mandlebrot set and map its colors.
__global__ void mandelbrot_kernel(const unsigned int image_width, const unsigned int image_height,
                                  const float x_range_start, const float y_range_start,
                                  const float x_increment, const float y_increment,
                                  const unsigned int max_iterations,
                                  const unsigned int pattern_size,  rgb * d_pixels,
                                  rgb * d_pattern)
{
    unsigned int num_pixels = image_width * image_height;

    for (unsigned int pixel = blockIdx.x * blockDim.x + threadIdx.x;
         pixel < num_pixels;
         pixel += blockDim.x * gridDim.x)
    {
        // Map y pixel to the imaginary number coordinate.
        float y0 = y_range_start + (pixel / image_width) * y_increment;

        // Map x pixel to the real number coordinate.
        float x0 = x_range_start + (pixel % image_width) * x_increment;

        // Calculate the iterations of a particular point.
        float x = 0.0; //used in mandelbrot calculations
        float y = 0.0; //used in mandelbrot calculations
        float xtemp; //used as a placeholder
        unsigned int iteration = 0; //index for number of iterations
        while ((x * x) + (y * y) < (2 * 2) && iteration < max_iterations)
        {
            xtemp = (x * x) - (y * y) + x0;
            y = (2 * x * y) + y0;
            x = xtemp;
            iteration = iteration + 1;
        }

        // Map each pixel value to the corresponding pattern value.
        unsigned int pattern_map = iteration % pattern_size;
        d_pixels[pixel].r = d_pattern[pattern_map].r;
        d_pixels[pixel].g = d_pattern[pattern_map].g;
        d_pixels[pixel].b = d_pattern[pattern_map].b;
    }
}

cudaError_t Init()
{
    #ifdef SHOW_RESULT
    // Basic Opengl initialization.
    glViewport(0, 0, image_width, image_height);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0, image_width, 0, image_height);
    #endif


    // Declare variables.
    // Declare pointers to hold the addresses of the pixel and pattern arrays on the device.
    rgb * d_pixels = 0;
    rgb * d_pattern = 0;
    // Declare a variable to hold the status of the CUDA device so it can be checked.
    cudaError_t cuda_status;
    // Declare a variable to hold the starting time point of the mandelbrot call.
    #if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    chrono::high_resolution_clock::time_point time_begin;
    #else
    timespec time_begin;
    #endif
    // Declare a variable to hold the ending time point of the mandelbrot call.
    #if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    chrono::high_resolution_clock::time_point time_end;
    #else
    timespec time_end;
    #endif
    // Declare a variable to hold the duration of the mandelbrot call.
    #if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    chrono::duration<double> time_span;
    #else
    double time_span;
    #endif

    // Calculate the increments in the mandlebrot set.
    x_increment = abs(X_RANGE_START - X_RANGE_END) / image_width;
    y_increment = abs(Y_RANGE_START - Y_RANGE_END) / image_height;

    // Allocate memory for the pixel array on the host.
    // h_pixels = new rgb[image_height * image_width];
    h_pixels = (rgb *)malloc(image_width * image_height * sizeof(rgb));
    if (h_pixels == nullptr)
    {
        fprintf(stderr, "Memory allocation failed. (h_pixels)\n");
        goto Error;
    }

    // Choose which GPU to run on, change this on a multi-GPU system.
    cuda_status = cudaSetDevice(0);
    if (cuda_status != cudaSuccess)
    {
        fprintf(stderr, "cudaSetDevice failed.  Do you have a CUDA-capable GPU installed?\n");
        goto Error;
    }
    
    // Allocate memory for the pixel and pattern arrays on the device.
    cuda_status = cudaMalloc(&d_pixels, image_width * image_height * sizeof(rgb));
    if (cuda_status != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed (d_pixels): %s\n", cudaGetErrorString(cuda_status));
        goto Error;
    }
    cuda_status = cudaMalloc(&d_pattern, PATTERN_SIZE * sizeof(rgb));
    if (cuda_status != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed (d_pattern): %s\n", cudaGetErrorString(cuda_status));
        goto Error;
    }

    //Initialize the pixel and pattern arrays on the device.
    init_pixels_kernel<<<num_cuda_blocks, num_cuda_threads_per_block>>>(image_height * image_width,
                                                                        d_pixels);
    cuda_status = cudaDeviceSynchronize();
    // Check for any errors that occurred while launching the kernel.
    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess)
    {
        fprintf(stderr, "init_pixels_kernel launch failed: %s\n", cudaGetErrorString(cuda_status));
        goto Error;
    }
    init_pattern_kernel<<<num_cuda_blocks, num_cuda_threads_per_block>>>(PATTERN_SIZE, d_pattern);
    cuda_status = cudaDeviceSynchronize();
    // Check for any errors that occurred while launching the kernel.
    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess)
    {
        fprintf(stderr, "init_pattern_kernel launch failed: %s\n", cudaGetErrorString(cuda_status));
        goto Error;
    }

    // Record the current (starting) time of the mandelbrot call.
    #if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    time_begin = chrono::high_resolution_clock::now();
    #else
    clock_gettime(CLOCK_REALTIME, &time_begin);
    #endif

    // Call the mandelbrot function on the device.
    mandelbrot_kernel<<<num_cuda_blocks, num_cuda_threads_per_block>>>(image_width, image_height,
                                                                       X_RANGE_START, Y_RANGE_START,
                                                                       x_increment, y_increment,
                                                                       num_iterations, PATTERN_SIZE,
                                                                       d_pixels, d_pattern);
    cuda_status = cudaDeviceSynchronize();

    // Record the current (ending) time of the mandelbrot call.
    #if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    time_end = chrono::high_resolution_clock::now();
    #else
    clock_gettime(CLOCK_REALTIME, &time_end);
    #endif

    // Calculate the duration of the mandelbrot call.
    #if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    time_span = chrono::duration_cast<chrono::duration<double>>(time_end - time_begin);
    #else
    time_span = (1000000000 * (time_end.tv_sec - time_begin.tv_sec) + time_end.tv_nsec - 
                 time_begin.tv_sec) / (double)1000000000;
    #endif

    // Check for any errors that occurred while launching the kernel.
    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess)
    {
        fprintf(stderr, "mandelbrot_kernel launch failed: %s\n", cudaGetErrorString(cuda_status));
        goto Error;
    }

    // Copy the pixel array from the device to the host.
    cuda_status = cudaMemcpy(h_pixels, d_pixels, image_width * image_height * sizeof(rgb),
                             cudaMemcpyDeviceToHost);
    if (cuda_status != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy failed (d_pattern->h_pattern): %s\n",
                cudaGetErrorString(cuda_status));
        goto Error;
    }

    #if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    printf("Performed %d iterations in %f seconds using %d blocks and %d threads per block.\n",
           num_iterations, time_span.count(), num_cuda_blocks, num_cuda_threads_per_block);
    #else
    printf("Performed %d iterations in %f seconds using %d blocks and %d threads per block.\n",
           num_iterations, time_span, num_cuda_blocks, num_cuda_threads_per_block);
    #endif

Error:
    cudaFree(d_pixels);
    cudaFree(d_pattern);

    return cuda_status;
}

#ifdef SHOW_RESULT
void onDisplay()
{
    // Clearing the initial buffer
    glClearColor(1, 1, 1, 0);
    glClear(GL_COLOR_BUFFER_BIT);

    // Draw the complete Mandelbrot set picture
    glDrawPixels(image_width, image_height, GL_RGB, GL_FLOAT, h_pixels);
    glutSwapBuffers();
}
#endif

int main(int argc, char** argv)
{
    //---------------------------------------------
    // Handle general command-line arguments here.
    //---------------------------------------------
    if (argc > 3)
    {
        num_iterations = atoi(argv[1]);
        image_width = atoi(argv[2]);
        image_height = atoi(argv[3]);
    }
    else if (argc > 1)
    {
        num_iterations = atoi(argv[1]);
        image_width = DEFAULT_IMAGE_WIDTH;
        image_height = DEFAULT_IMAGE_HEIGHT;
    }
    else
    {
        num_iterations = DEFAULT_NUM_ITERATIONS;
        image_width = DEFAULT_IMAGE_WIDTH;
        image_height = DEFAULT_IMAGE_HEIGHT;
    }


    //-------------------------------------------------------------
    // Handle implementation-specific command-line arguments here.
    //-------------------------------------------------------------
    if (argc > 5)
    {
        num_cuda_blocks = atoi(argv[4]);
        num_cuda_threads_per_block = atoi(argv[5]);
    }
    else if (argc > 4)
    {
        num_cuda_blocks = atoi(argv[4]);
        num_cuda_threads_per_block = DEFAULT_NUM_CUDA_THREADS_PER_BLOCK;
    }
    else
    {
        num_cuda_blocks = DEFAULT_NUM_CUDA_BLOCKS;
        num_cuda_threads_per_block = DEFAULT_NUM_CUDA_THREADS_PER_BLOCK;
    }

    #ifdef SHOW_RESULT
    // Perform basic OpenGL initialization.
    glutInit(&argc, argv);
    glutInitWindowSize(image_width, image_height);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
    glutInitWindowPosition(100, 100);
    glutCreateWindow("Mandelbrotset by SKR");
    #endif

    // Create a variable to hold the return code.
    int to_return = 0;

    // Call Init().
    if (Init() != cudaSuccess)
    {
        to_return = 1;
    }
    else
    {
        #ifdef SHOW_RESULT
        // Connecting the display function
        glutDisplayFunc(onDisplay);
        // starting the activities
        glutMainLoop();
        #endif
    }

    // Attempt to reset the device.  This is to allow tracing tools (Nsight/Visual Profiler/etc.) to
    // show complete traces.
    if (cudaDeviceReset() != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceReset failed.\n");
        to_return = 1;
    }
    
    
    // Free memory.
    if (h_pixels != nullptr)
        delete[] h_pixels;

    // Return.
    return to_return;
}


/***************************************************************************************************
 * Mandelbrotset.c
 * Copyright Shibin K.Reeny
 * This program is free software; you can redistribute it and/or modify it under the terms of the
 * GNU General Public License as published by the Free Software Foundation; either version 2 of
 * the License, or (at your option) any later version.
 **************************************************************************************************/

#include <GL/gl.h>
#include <GL/glut.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
using namespace std;



// Define an RGB struct to represent the color a pixel.
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

//-----------------------------------
// Implementation-Specific Constants
//-----------------------------------
// CUDA
//const unsigned int DEFAULT_NUM_CUDA_BLOCKS = 1;
//const unsigned int DEFAULT_NUM_CUDA_THREADS_PER_BLOCK = 32;
// MPI
//const unsigned int DEFAULT_NUM_RANKS = 1;
// OpenMP
//const unsigned int DEFAULT_NUM_THREADS = 1;

//-----------------
// General Globals
//-----------------
unsigned int image_width;
unsigned int image_height;
unsigned int num_iterations;
float x_increment;
float y_increment;
// Pixels contains the value of the colors pixel in the picture.
rgb *pixels;
// Pattern stores a predefined color for a particular value.
rgb *pattern;


//---------------------------------
// Implementation-Specific Globals
//---------------------------------
// CUDA
//unsigned int num_cuda_blocks;
//unsigned int num_cuda_threads_per_block;
// MPI
//unsigned int num_ranks;
// OpenMP
//unsigned int num_threads;



// Get the number of iterations
int getIterations(float x0, float y0)
{
    float x = 0.0; //used in mandelbrot calculations
    float y = 0.0; //used in mandelbrot calculations
    float xtemp; //used as a placeholder
    unsigned int iteration = 0; //index for number of iterations

    // mandelbrot calculations
    while ((x * x) + (y * y) < (2 * 2) && iteration < num_iterations)
    {
        xtemp = (x * x) - (y * y) + x0;
        y = (2 * x * y) + y0;
        x = xtemp;
        iteration = iteration + 1;
    }

    return iteration;
}

// Generate a mandlebrot set and map its colors.
void mandelbrotset()
{
    for (unsigned int yPixel = 0; yPixel < image_height; yPixel++)
    {
        // Map y pixel to the imaginary number coordinate.
        float y0 = Y_RANGE_START + yPixel * y_increment;

        for (unsigned int xPixel = 0; xPixel < image_width; xPixel++)
        {
            // Map x pixel to the real number coordinate.
            float x0 = X_RANGE_START + xPixel * x_increment;

            // Calculate the iterations of a particular point.
            unsigned int iteration = getIterations(x0, y0);

            // Map each pixel value to the corresponding pattern value.
            pixels[image_width * yPixel + xPixel].r = pattern[iteration % PATTERN_SIZE].r;
            pixels[image_width * yPixel + xPixel].g = pattern[iteration % PATTERN_SIZE].g;
            pixels[image_width * yPixel + xPixel].b = pattern[iteration % PATTERN_SIZE].b;
        }
    }
}

void Init()
{
#ifdef SHOW_RESULT
    // Basic Opengl initialization.
    // image_width = (-2.5 - 1.1)/0.0025
    // here total x coordinate distance / no of division.
    // 840 = (-1 - 1.1)/0.0025 +1
    // here total y coordinate distance / no of division.
    glViewport(0, 0, image_width, image_height);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0, image_width, 0, image_height);
#endif

    unsigned int i;
    float r, g, b;

    //Initializing all the pixels to white.
    for (i = 0; i < image_height * image_width; i++)
    {
        pixels[i].r = 1;
        pixels[i].g = 1;
        pixels[i].b = 1;
    }

    i = 0;

    // Initializing all the pattern color till 9*9*9
    for (r = 0.1f; r <= 0.9; r = r + 0.1f)
    {
        for (g = 0.1f; g <= 0.9; g = g + 0.1f)
        {
            for (b = 0.1f; b <= 0.9; b = b + 0.1f)
            {
                pattern[i].r = b;
                pattern[i].g = r;
                pattern[i].b = g;
                i++;
            }
        }
    }

    // Initializing the rest of the pattern as 9*9*9 is 729, and we need up to PATTERN_SIZE pattern
    // as the loop bailout condition is 1000.
    for (; i < PATTERN_SIZE; i++)
    {
        pattern[i].r = 1;
        pattern[i].g = 1;
        pattern[i].b = 1;
    }
    chrono::high_resolution_clock::time_point t1 = chrono::high_resolution_clock::now();
    mandelbrotset();
    chrono::high_resolution_clock::time_point t2 = chrono::high_resolution_clock::now();
    chrono::duration<double> time_span = chrono::duration_cast<chrono::duration<double>>(t2 - t1);

    printf("Performed %d iterations in %f seconds.\n", num_iterations, time_span.count());
}

#ifdef SHOW_RESULT
void onDisplay()
{
    // Clearing the initial buffer
    glClearColor(1, 1, 1, 0);
    glClear(GL_COLOR_BUFFER_BIT);

    // Draw the complete Mandelbrot set picture
    glDrawPixels(image_width, image_height, GL_RGB, GL_FLOAT, pixels);
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
    if (argc > 4)
    {
        // ...
    }
    // else if ()



    // Calculate the increments in the mandlebrot set.
    x_increment = abs(X_RANGE_START - X_RANGE_END) / image_width;
    y_increment = abs(Y_RANGE_START - Y_RANGE_END) / image_height;

    // Initialize the pixel and pattern arrays.
    pixels = new rgb[image_height * image_width];
    pattern = new rgb[PATTERN_SIZE];


#ifdef SHOW_RESULT
    // Perform basic OpenGL initialization.
    glutInit(&argc, argv);
    glutInitWindowSize(image_width, image_height);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
    glutInitWindowPosition(100, 100);
    glutCreateWindow("Mandelbrotset by SKR");
#endif


    Init();


#ifdef SHOW_RESULT
    // Connecting the display function
    glutDisplayFunc(onDisplay);
    // starting the activities
    glutMainLoop();
#endif


    // Free memory.
    delete[] pixels;
    delete[] pattern;

    // Return.
    return 0;
}

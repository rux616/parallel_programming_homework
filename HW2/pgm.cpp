#include "pgm.h"

namespace pgm
{
pgm::pgm()
{}

pgm::~pgm()
{
    delete[] data;
}

const std::int32_t pgm::pixel(std::int32_t row, std::int32_t column)
{
    if (row >= 0 && row < height && column >= 0 && column < width)
    {
        return data[row + column * width];
    }
    else
    {
        return -1;
    }
}


}
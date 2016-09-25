#pragma once
#ifndef PGM_H
#define PGM_H

#include <cstdint>
#include <string>

namespace pgm
{
class pgm
{
    private:
    std::int32_t * data = nullptr;
    
    std::int32_t width = -1;
    std::int32_t height = -1;
    std::int16_t depth = -1;

    public:
    pgm();
    ~pgm();

    const std::int32_t pixel(std::int32_t, std::int32_t);
    
    const std::int32_t get_width();
    const std::int32_t get_height();
    const std::int16_t get_depth();

    void read(std::ifstream);
    void write(std::ofstream);
};
}

#endif
#pragma once
#ifndef UTILITY_H
#define UTILITY_H

namespace utility
{
inline int translate_2d_to_1d(const int, const int, const int);

void extend_array(char *,
                  const int, const int,
                  const int, const int,
                  const int, const int);
}
#endif
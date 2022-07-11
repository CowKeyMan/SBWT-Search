#ifndef BIT_VECTOR_UTILS_H
#define BIT_VECTOR_UTILS_H

/**
 * @file BitVectorUtils.h
 * @brief Contains definitions and helper functions for working with the
          BitVector format
 * */

#include <string>
#include <vector>

using std::string;
using std::vector;

const vector<string> acgt_postfixes = { "BWT_A", "BWT_C", "BWT_G", "BWT_T" };
const string c_map_postfix = { "_C" };

#endif

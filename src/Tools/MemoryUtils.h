#ifndef MEMORY_UTILS_H
#define MEMORY_UTILS_H

/**
 * @file MemoryUtils.h
 * @brief Utilities which deal with memory management and querying
 */

// Function credits:
// https://stackoverflow.com/questions/2513505/how-to-get-available-memory-c-g

#include "Tools/TypeDefinitions.h"

namespace memory_utils {

auto get_total_system_memory() -> u64;

}  // namespace memory_utils

#endif

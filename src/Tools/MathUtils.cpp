#include "Tools/MathUtils.hpp"
#include "Tools/TypeDefinitions.h"

namespace math_utils {

const u64 bytes_in_kb = 1024;

auto bits_to_gB(u64 bits) -> double {
  return static_cast<double>(bits)
    / (sizeof(u64) * bytes_in_kb * bytes_in_kb * bytes_in_kb);
}

auto gB_to_bits(double gB) -> u64 {
  return static_cast<u64>(
    gB * sizeof(u64) * bytes_in_kb * bytes_in_kb * bytes_in_kb
  );
}

}  // namespace math_utils

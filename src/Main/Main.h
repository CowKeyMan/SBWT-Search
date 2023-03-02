#ifndef MAIN_H
#define MAIN_H

/**
 * @file Main.h
 * @brief Interface class for main classes
 */

#include <string>
#include <vector>

#include "Tools/TypeDefinitions.h"

namespace sbwt_search {

using std::string;
using std::vector;

class Main {
private:
  u64 threads = 0;

protected:
  [[nodiscard]] auto get_threads() const -> u64;

public:
  virtual auto main(int argc, char **argv) -> int = 0;
  virtual ~Main() = default;
  Main(Main &) = delete;
  Main(Main &&) = delete;
  auto operator=(Main &) = delete;
  auto operator=(Main &&) = delete;

protected:
  Main();
  auto load_threads() -> void;
};

}  // namespace sbwt_search

#endif

#ifndef MAIN_H
#define MAIN_H

/**
 * @file Main.h
 * @brief Interface class for main classes
 */

namespace sbwt_search {

class Main {
public:
  Main() = default;
  virtual auto main(int argc, char **argv) -> int = 0;
  virtual ~Main() = default;
  Main(Main &) = delete;
  Main(Main &&) = delete;
  auto operator=(Main &) = delete;
  auto operator=(Main &&) = delete;
};

}  // namespace sbwt_search

#endif

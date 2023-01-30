#ifndef COLOR_SEARCH_MAIN_H
#define COLOR_SEARCH_MAIN_H

/**
 * @file ColorSearchMain.h
 * @brief The main function for searching for the colors. The 'color' mode of
 * the main executable
 */

#include "Main/Main.h"

namespace sbwt_search {

class ColorSearchMain: public Main {
  auto main(int argc, char **argv) -> int override;
};

};  // namespace sbwt_search
#endif

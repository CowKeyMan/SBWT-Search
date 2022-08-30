#ifndef ERROR_UTILS_H
#define ERROR_UTILS_H

/**
 * @file ErrorUtils.h
 * @brief Helper class for error related functionality
 * */

#include <string>

using std::string;


namespace error_utils {

auto _throw_uninitialised(string file, unsigned int line) -> void;
#define throw_uninitialised() error_utils::_throw_uninitialised(__FILE__, __LINE__)

}  // namespace error_utils


#endif

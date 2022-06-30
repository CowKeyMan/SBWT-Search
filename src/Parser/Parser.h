#ifndef PARSER_H
#define PARSER_H

/**
 * @file Parser.h
 * @brief A parent class which contains methods common to all parsers
 * */

namespace sbwt_search {

class Parser {
private:
  bool has_parsed = false;
protected:
  void check_if_has_parsed();
  Parser(){};
};

}

#endif

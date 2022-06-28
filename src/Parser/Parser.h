#ifndef PARSER_H
#define PARSER_H

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

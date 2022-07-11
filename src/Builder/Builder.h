#ifndef BUILDER_H
#define BUILDER_H

/**
 * @file Builder.h
 * @brief A parent class which contains methods common to
 * anything that builds something as its main task
 * */

namespace sbwt_search {

class Builder {
  private:
    bool has_built = false;

  protected:
    void check_if_has_built();
    Builder(){};
};

}

#endif

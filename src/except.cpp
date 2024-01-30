//
// Created by Denis Ryapolov on 29.01.2024.
//

#include "except.h"
#include <exception>

namespace except {
  void react() {
    try {
      throw;
    } catch(std::exception& e) {
      // handle exceptions known
    } catch(...) {
      // handle exceptions unknown
    }
  }
}

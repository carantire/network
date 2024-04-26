//
// Created by Denis Ryapolov on 29.01.2024.
//

#include "except.h"
#include <exception>
#include <iostream>

namespace except {
void react() {
  try {
    throw;
  } catch (std::exception &e) {
    std::cerr << e.what() << '\n';
  } catch (...) {
  }
}
} // namespace except

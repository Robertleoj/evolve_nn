#include "../include/foundation/example.hpp"
#include <iostream>

int main() {
    int a = 1, b = 2;
    std::cout << "The sum of"
            << " " << a
            << " and"
            << " " << b
            << " is"
            << " " 
            << foundation::add(1, 2) << std::endl;
}
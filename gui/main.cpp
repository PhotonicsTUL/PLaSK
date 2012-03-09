#include <plask/geometry/element.h>
#include <plask/geometry/stack.h>
#include <chrono>
#include <typeindex>
#include <iostream>

int main() {
    
    plask::shared_ptr<plask::GeometryElement> g(new plask::MultiStackContainer<2>());
    plask::shared_ptr<plask::GeometryElement> g2(new plask::StackContainer<2>());
    
    auto t1 = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < 200000; ++i) {
        if (std::type_index(typeid(*g)) !=
                std::type_index(typeid(*g2))) ;
        std::type_index(typeid(*g)) ==
        std::type_index(typeid(*g));
        std::type_index(typeid(*g));
        
        /*plask::dynamic_pointer_cast<plask::MultiStackContainer<2>>(g);
        plask::dynamic_pointer_cast<plask::MultiStackContainer<2>>(g);
        plask::dynamic_pointer_cast<plask::MultiStackContainer<2>>(g);
        
        plask::dynamic_pointer_cast<plask::MultiStackContainer<2>>(g);
        plask::dynamic_pointer_cast<plask::MultiStackContainer<2>>(g);*/
    }
    
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count()
              << " milliseconds / 1 000 000\n";
    
    return 0;
}

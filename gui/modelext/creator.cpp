#include "creator.h"

#include <plask/geometry/leaf.h>
#include <plask/geometry/stack.h>

struct Block2dCreator: public GeometryElementCreator {

    virtual plask::shared_ptr<plask::GeometryElement> getElement() const {
        return plask::make_shared< plask::Block<2> >();
    }

    virtual std::string getName() const { return "block2d"; }

    virtual int getDimensionsCount() const { return 2; }
};

struct Stack2dCreator: public GeometryElementCreator {

    virtual plask::shared_ptr<plask::GeometryElement> getElement() const {
        return plask::make_shared< plask::StackContainer<2> >();
    }

    virtual std::string getName() const { return "stack2d"; }

    virtual int getDimensionsCount() const { return 2; }
};


template <>
std::vector<const GeometryElementCreator*> getCreators<2>() {
    static std::vector<const GeometryElementCreator*> vec = { new Block2dCreator(), new Stack2dCreator() };
    return vec;
}

template <>
std::vector<const GeometryElementCreator*> getCreators<3>() {
    static std::vector<const GeometryElementCreator*> vec = {  };
    return vec;
}

template std::vector<const GeometryElementCreator*> getCreators<2>();
template std::vector<const GeometryElementCreator*> getCreators<3>();

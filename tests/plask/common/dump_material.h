//dump material to use in tests

#include <plask/material/material.h>

struct DumpMaterial: public plask::Material {
    virtual std::string getName() const { return "Dump"; }
};

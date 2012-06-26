#ifndef PLASK_GUI_MATERIAL_H
#define PLASK_GUI_MATERIAL_H

#include <plask/material/material.h>

/**
 * Represent material which holds only a name.
 */
struct NameOnlyMaterial: public plask::Material {

    std::string _name;

public:
    NameOnlyMaterial(const std::string& name): _name(name) {}

    static plask::shared_ptr<Material> getInstance(const std::string& name) {
        return plask::make_shared<NameOnlyMaterial>(name);
    }

    virtual std::string name() const;

    virtual Kind kind() const;

};

#endif // PLASK_GUI_MATERIAL_H

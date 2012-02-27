//dump material to use in tests

#include <plask/material/db.h>

struct DumbMaterial: public plask::Material {
    virtual std::string name() const { return "Dumb"; }
    virtual plask::Material::Kind kind() const { return plask::Material::SEMICONDUCTOR; }
};

inline void initDumbMaterialDb(plask::MaterialsDB& db) {
    db.add<DumbMaterial>("Al");
}

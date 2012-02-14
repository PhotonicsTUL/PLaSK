//dump material to use in tests

#include <plask/material/db.h>

struct DumbMaterial: public plask::Material {
    virtual std::string name() const { return "Dumb"; }
};

inline plask::Material* construct_dumb_material(const plask::Material::Composition& composition, plask::Material::DOPING_AMOUNT_TYPE doping_amount_type, double doping_amount) {
    return new DumbMaterial();
}

inline void initDumbMaterialDb(plask::MaterialsDB& db) {
    db.add<DumbMaterial>("Al");
}

//dump material to use in tests

#include <plask/material/db.h>

struct DumbMaterial: public plask::Material {
    virtual std::string name() const { return "Dumb"; }
    virtual plask::Material::Kind kind() const { return plask::Material::SEMICONDUCTOR; }
};

inline plask::Material* construct_dumb_material(const plask::Material::Composition& composition, plask::Material::DopingAmountType doping_amount_type, double doping_amount) {
    return new DumbMaterial();
}

inline void initDumbMaterialDb(plask::MaterialsDB& db) {
    db.add<DumbMaterial>("Al");
}

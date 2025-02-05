//dump material to use in tests

#include <plask/material/db.hpp>

struct DumbMaterial: public plask::Material {
    std::string name() const override { return "Dumb"; }
    plask::Material::Kind kind() const override { return plask::Material::SEMICONDUCTOR; }
};

inline void initDumbMaterialDb(plask::MaterialsDB& db = plask::MaterialsDB::getDefault()) {
    db.add<DumbMaterial>("Al");
}

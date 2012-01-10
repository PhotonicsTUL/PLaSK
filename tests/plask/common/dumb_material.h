//dump material to use in tests

#include <plask/material/material.h>

struct DumbMaterial: public plask::Material {
    virtual std::string getName() const { return "Dumb"; }
};

inline plask::Material* construct_dumb_material(const std::vector<double>& composition, plask::MaterialsDB::DOPANT_AMOUNT_TYPE dopant_amount_type, double dopant_amount) {
    return new DumbMaterial();
}

inline void initDumbMaterialDb(plask::MaterialsDB& db) {
    db.add("Dumb", &construct_dumb_material);
}

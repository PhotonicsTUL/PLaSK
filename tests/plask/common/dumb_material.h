//dump material to use in tests

#include <plask/material/material.h>

struct DumbMaterial: public plask::Material {
    virtual std::string name() const { return "Dumb"; }
};

inline plask::shared_ptr<plask::Material> construct_dumb_material(const std::string& name, const std::vector<double>& composition, plask::MaterialsDB::DOPANT_AMOUNT_TYPE dopant_amount_type, double dopant_amount) {
    return plask::shared_ptr<plask::Material> { new DumbMaterial() };
}

inline void initDumbMaterialDb(plask::MaterialsDB& db) {
    db.add("Dumb", &construct_dumb_material);
}

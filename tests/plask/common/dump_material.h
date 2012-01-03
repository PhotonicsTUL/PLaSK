//dump material to use in tests

#include <plask/material/material.h>

struct DumpMaterial: public plask::Material {
    virtual std::string getName() const { return "Dump"; }
};

inline plask::Material* construct_dump_material(const std::vector<double>& composition, plask::MaterialsDB::DOPANT_AMOUNT_TYPE dopant_amount_type, double dopant_amount) {
    return new DumpMaterial();
}

inline void initDumpMaterialDb(plask::MaterialsDB& db) {
    db.add("Dump", &construct_dump_material);
}

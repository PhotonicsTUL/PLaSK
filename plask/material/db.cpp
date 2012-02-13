#include "db.h"

#include "../utils/string.h"

namespace plask {

std::string dbKey(const Material::Composition &composition, const std::string& dopant_name) {
    std::string db_key;
    for (auto c: composition) db_key += c.first;
    if (!dopant_name.empty()) {
        db_key += ':';
        db_key += dopant_name;
    }
    return db_key;
}

shared_ptr<Material> MaterialsDB::get(const Material::Composition &composition, const std::string& dopant_name, Material::DOPING_AMOUNT_TYPE doping_amount_type, double doping_amount) const {
    auto it = constructors.find(dbKey(composition, dopant_name));
    if (it == constructors.end()) throw NoSuchMaterial(composition, dopant_name);
    return (*it->second)(composition, doping_amount_type, doping_amount);
}

shared_ptr<Material> plask::MaterialsDB::get(const std::string& parsed_name_with_donor, const std::vector<double>& composition,
                                               Material::DOPING_AMOUNT_TYPE doping_amount_type, double doping_amount) const {
    std::string name, dopant;
    std::tie(name, dopant) = splitString2(parsed_name_with_donor, ':');
    std::vector<std::string> elements = Material::parseElementsNames(name);
    if (composition.size() > elements.size())
        throw plask::Exception("To long composition vector (longer than number of elements in \"%1%\").", parsed_name_with_donor);
    Material::Composition comp;
    for (std::size_t i = 0; i < composition.size(); ++i) comp[elements[i]] = composition[i];
    for (std::size_t i = composition.size(); i < elements.size(); ++i) comp[elements[i]] = std::numeric_limits<double>::quiet_NaN();
    return get(Material::completeComposition(comp), dopant, doping_amount_type, doping_amount);
}

shared_ptr< Material > MaterialsDB::get(const std::string& name_with_components, const std::string& doping_descr) const {
    Material::Composition composition = Material::parseComposition(name_with_components);
    std::string dopant_name;
    double doping_amount = 0.0;
    Material::DOPING_AMOUNT_TYPE doping_amount_type = Material::NO_DOPING;
    if (!doping_descr.empty())
        Material::parseDopant(doping_descr.data(), doping_descr.data() + doping_descr.size(), dopant_name, doping_amount_type, doping_amount);
    return get(composition, dopant_name, doping_amount_type, doping_amount);
}

shared_ptr< Material > MaterialsDB::get(const std::string& full_name) const {
    auto pair = splitString2(full_name, ':');
    return get(std::get<0>(pair), std::get<1>(pair));
}

void MaterialsDB::add(std::vector<std::string> elemenNames, const std::string &dopant, const MaterialsDB::MaterialConstructor *constructor) {
    std::sort(elemenNames.begin(), elemenNames.end());
    std::string dbKey;
    for (auto n: elemenNames) dbKey += n;
    if (!dopant.empty()) {
        dbKey += ':';
        dbKey += dopant;
    }
    constructors[dbKey] = std::unique_ptr<const MaterialConstructor>(constructor);   
}

void MaterialsDB::add(const std::string& full_name, const MaterialConstructor* constructor) {
    std::string name, dopant;
    std::tie(name, dopant) = splitString2(full_name, ':');
    add(Material::parseElementsNames(name), dopant, constructor);
}

void MaterialsDB::add(const std::string& name, plask::MaterialsDB::construct_material_f* constructor) {
    add(name, new FunctionBasedMaterialConstructor(constructor));
}

}   // namespace plask

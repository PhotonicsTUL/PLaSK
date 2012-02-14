#include "db.h"

#include "../utils/string.h"

namespace plask {

//Append to name dopant, if it is not empty
void appendDopant(std::string& name, const std::string& dopant_name) {
    if (!dopant_name.empty()) {
        name += ':';
        name += dopant_name;
    }
}

std::string dbKey(const Material::Composition &composition, const std::string& dopant_name) {
    std::string db_key;
    for (auto c: composition) db_key += c.first;
    appendDopant(db_key, dopant_name);
    return db_key;
}

std::string dbKey(std::vector<std::string> elNames, const std::string& dopant_name) {
    std::string db_key;
    std::sort(elNames.begin(), elNames.end());
    for (std::string& c: elNames) db_key += c;
    appendDopant(db_key, dopant_name);
    return db_key;
}

std::string dbKey(const std::string& name, const std::string& dopant_name) {
    return dbKey(Material::parseElementsNames(name), dopant_name);
}

/*std::string dbKey(std::vector<std::string> elemenNames, const std::string& dopant_name = "") {
    std::string result;
    std::vector<std::string>::iterator grBegin = elemenNames.begin(); 
    if (grBegin == elemenNames.end()) return "";    //exception??
    int grNr = elementGroup(*grBegin);
    for (std::vector<std::string>::iterator grEnd = grBegin + 1; grEnd != elemenNames.end(); ++grEnd) {
        int endNr = elementGroup(*grEnd);
        if (grNr != endNr) {
            std::sort(grBegin, grEnd);
            for (auto s_iter = grBegin; s_iter != grEnd; ++s_iter) result += *s_iter;
            grNr = endNr;
            grBegin = grEnd;
        }
    }
    std::sort(grBegin, elemenNames.end());
    for (auto s_iter = grBegin; s_iter != elemenNames.end(); ++s_iter) result += *s_iter;
    appendDopant(result, dopant_name);
    return result;
}*/

void MaterialsDB::ensureCompositionIsNotEmpty(const Material::Composition &composition) {
    if (composition.empty()) throw MaterialParseException("Unknown composition.");
}

shared_ptr<Material> MaterialsDB::get(const std::string& dbKey, const Material::Composition& composition, const std::string& dopant_name, Material::DOPING_AMOUNT_TYPE doping_amount_type, double doping_amount) const {
    auto it = constructors.find(dbKey);
    if (it == constructors.end()) {
        if (composition.empty()) throw NoSuchMaterial(dbKey, dopant_name);
        throw NoSuchMaterial(composition, dopant_name);
    }
    return (*it->second)(composition, doping_amount_type, doping_amount);
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
    std::string dopant_name;
    double doping_amount = 0.0;
    Material::DOPING_AMOUNT_TYPE doping_amount_type = Material::NO_DOPING;
    if (!doping_descr.empty())
        Material::parseDopant(doping_descr, dopant_name, doping_amount_type, doping_amount);
    if (name_with_components.find('(') == std::string::npos) {  //simple case, without parsing composition
        return get(name_with_components, Material::Composition(), dopant_name, doping_amount_type, doping_amount);
    } else  //parse composition:
        return get(Material::parseComposition(name_with_components), dopant_name, doping_amount_type, doping_amount);
}

shared_ptr< Material > MaterialsDB::get(const std::string& full_name) const {
    auto pair = splitString2(full_name, ':');
    return get(std::get<0>(pair), std::get<1>(pair));
}

void MaterialsDB::add(const std::string& elemenNames, const std::string &dopant, const MaterialsDB::MaterialConstructor *constructor) {
    constructors[dbKey(elemenNames, dopant)] = std::unique_ptr<const MaterialConstructor>(constructor);
}

void MaterialsDB::addSimple(const MaterialConstructor* constructor) {
    constructors[constructor->materialName] = std::unique_ptr<const MaterialConstructor>(constructor);
}

void MaterialsDB::add(const MaterialConstructor* constructor) {
    std::string name, dopant;
    std::tie(name, dopant) = splitString2(constructor->materialName, ':');
    add(name, dopant, constructor);
}

}   // namespace plask

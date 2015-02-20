#include "db.h"

#include "../utils/string.h"
#include "../utils/dynlib/manager.h"
#include "../log/log.h"

#include <boost/filesystem.hpp>

namespace plask {

void checkCompositionSimilarity(const Material::Composition& material1composition, const Material::Composition& material2composition) {
    for (auto& p1: material1composition) {
        auto p2 = material2composition.find(p1.first);
        if (p2 == material2composition.end())
            throw MaterialParseException("Materials compositions are different: %1% if missing from one of the materials", p1.first);
        if (std::isnan(p1.second) != std::isnan(p2->second))
            throw MaterialParseException("Amounts must be defined for the same elements, which is not true in case of '%1%' element", p1.first);
    }
}

/*const MaterialsDB *MaterialsDB::getFromSource(const MaterialsSource &materialsSource) {
    const MaterialsDB::Source* src = materialsSource.target<const MaterialsDB::Source>();
    return src ? &src->materialsDB : nullptr;
}*/

MaterialsDB::MixedCompositionOnlyFactory::MixedCompositionOnlyFactory(shared_ptr<const MaterialConstructor> constructor, const Material::Composition& material1composition, const Material::Composition& material2composition)
    : MaterialsDB::MixedCompositionFactory::MixedCompositionFactory(constructor), material1composition(material1composition), material2composition(material2composition) {
    //check if compositions are fine and simillar:
    checkCompositionSimilarity(material1composition, material2composition);
    checkCompositionSimilarity(material2composition, material1composition);
    Material::completeComposition(material1composition);
    Material::completeComposition(material2composition);
}

Material::Composition plask::MaterialsDB::MixedCompositionOnlyFactory::mixedComposition(double m1_weight) const {
    Material::Composition result = material1composition;
    for (auto& p1: result) {
        if (!std::isnan(p1.second)) {
            auto p2 = material2composition.find(p1.first);
            p1.second = p1.second * m1_weight + p2->second * (1.0 - m1_weight);
        }
    }
    return result;
}

//if part_value is not empty append separator and part_value to name
std::string& appendPart(std::string& name, const std::string& part_value, char separator) {
    if (!part_value.empty()) {
        name += separator;
        name += part_value;
    }
    return name;
}

//Append to name dopant, if it is not empty, return name
std::string& appendDopant(std::string& name, const std::string& dopant_name) {
    return appendPart(name, dopant_name, ':');
}

std::string& appendLabel(std::string& name, const std::string& label) {
    return appendPart(name, label, '_');
}

std::string& appendLabelDopant(std::string& name, const std::string& label, const std::string& dopant_name) {
    return  appendDopant(appendLabel(name, label), dopant_name);
}

std::string complexDbKey(const Material::Composition &composition, const std::string& label, const std::string& dopant_name) {
    std::string db_key;
    for (auto c: composition) db_key += c.first;
    return appendLabelDopant(db_key, label, dopant_name);
}

std::string complexDbKey(std::vector<std::string> elNames, const std::string& label, const std::string& dopant_name) {
    std::string db_key;
    std::sort(elNames.begin(), elNames.end());
    for (std::string& c: elNames) db_key += c;
    return appendLabelDopant(db_key, label, dopant_name);
}

std::string complexDbKey(const std::string& name, const std::string& label, const std::string& dopant_name) {
    return complexDbKey(Material::parseObjectsNames(name), label, dopant_name);
}

std::string complexDbKey(const std::string& fullComplexName) {
    auto fullname_dopant = splitString2(fullComplexName, ':');
    auto name_label = splitString2(fullname_dopant.first, '_');
    return complexDbKey(name_label.first, name_label.second, fullname_dopant.second);
}

std::string dbKey(const Material::Parameters& parameters) {
    std::string res;
    if (parameters.isSimple())
        res = parameters.name;
    else
        for (auto c: parameters.composition) res += c.first;
    return appendLabelDopant(res, parameters.label, parameters.dopantName);
}


MaterialsDB& MaterialsDB::getDefault() {
    static MaterialsDB defaultDb;
    return defaultDb;
}

void MaterialsDB::loadToDefault(const std::string &fileName_mainpart) {
    //DynamicLibraries::defaultLoad(plaskMaterialsPath() + fileName_mainpart + DynamicLibrary::DEFAULT_EXTENSION);
    DynamicLibrary(plaskMaterialsPath() + fileName_mainpart + DynamicLibrary::DEFAULT_EXTENSION, DynamicLibrary::DONT_CLOSE);
}

void MaterialsDB::loadAllToDefault(const std::string& dir) {
    if (boost::filesystem::exists(dir) && boost::filesystem::is_directory(dir)) {
        boost::filesystem::directory_iterator iter(dir);
        boost::filesystem::directory_iterator end;
        while (iter != end) {
            boost::filesystem::path p = iter->path();
            if (boost::filesystem::is_regular_file(p))
                DynamicLibraries::defaultLoad(p.string(), DynamicLibrary::DONT_CLOSE);
            ++iter;
        }
    } else {
        writelog(LOG_WARNING, "MaterialsDB: '%1%' does not exist or is not a directory. Cannot load default materials", dir);
    }
}

void MaterialsDB::ensureCompositionIsNotEmpty(const Material::Composition &composition) {
    if (composition.empty()) throw MaterialParseException("Unknown material composition");
}

MaterialsDB::ProxyMaterialConstructor::ProxyMaterialConstructor():
    MaterialsDB::MaterialConstructor(""), material(new EmptyMaterial)
{}

MaterialsDB::ProxyMaterialConstructor::ProxyMaterialConstructor(const std::string& name, const MaterialsDB& db):
    MaterialsDB::MaterialConstructor(name)
{
    if (name.empty()) {
        material = make_shared<EmptyMaterial>();
    } else {
        Material::Parameters p(name, true);
        if (p.hasDopant())  // base material has defined dopant
            material = db.get(p);
        else  // (dopant name without ammount still can be defined here)
            constructor = db.getConstructor(p, true);
    }
}

MaterialsDB::ProxyMaterialConstructor::ProxyMaterialConstructor(const shared_ptr<Material>& material):
    MaterialsDB::MaterialConstructor(material->name()), material(material)
{}

shared_ptr<const MaterialsDB::MaterialConstructor> MaterialsDB::getConstructor(const std::string& db_Key, const Material::Composition& composition, const std::string& dopant_name, bool allow_complex_without_composition) const {
    auto it = constructors.find(db_Key);
    if (it == constructors.end()) {
        if (composition.empty()) {
            // check if material is complex, but user forgot to provide composition:
            std::string complex_DbKey;
            try { complex_DbKey = complexDbKey(db_Key); } catch (std::exception& e) {}
            auto c = constructors.find(complex_DbKey);
            if (c != constructors.end()) { //material is complex
                if (allow_complex_without_composition)
                    return c->second;
                else
                    throw MaterialParseException(format("Material composition is required for %1%", db_Key));
            } else
                throw NoSuchMaterial(db_Key);
        }
        throw NoSuchMaterial(composition, dopant_name);
    }
    return it->second;
}

shared_ptr<Material> MaterialsDB::get(const std::string& db_Key, const Material::Composition& composition, const std::string& dopant_name, Material::DopingAmountType doping_amount_type, double doping_amount) const {
    return (*getConstructor(db_Key, composition, dopant_name))(composition, doping_amount_type, doping_amount);
}

shared_ptr<const MaterialsDB::MaterialConstructor> MaterialsDB::getConstructor(const Material::Composition& composition, const std::string& label, const std::string& dopant_name) const {
    return getConstructor(complexDbKey(composition, label, dopant_name), composition, dopant_name);
}

shared_ptr<const MaterialsDB::MaterialConstructor> MaterialsDB::getConstructor(const Material::Parameters &material, bool allow_complex_without_composition) const
{
    return getConstructor(dbKey(material), material.composition, material.dopantName, allow_complex_without_composition);
}

shared_ptr<Material> MaterialsDB::get(const Material::Composition &composition, const std::string& label, const std::string& dopant_name, Material::DopingAmountType doping_amount_type, double doping_amount) const {
    return get(complexDbKey(composition, label, dopant_name), composition, dopant_name, doping_amount_type, doping_amount);
}

/*shared_ptr<Material> MaterialsDB::get(const std::string& parsed_name_with_dopant, const std::vector<double>& composition,
                                               Material::DopingAmountType doping_amount_type, double doping_amount) const {
    std::string name, dopant;
    std::tie(name, dopant) = splitString2(parsed_name_with_dopant, ':');
    if (composition.empty())
        return get(parsed_name_with_dopant, Material::Composition(), dopant, doping_amount_type, doping_amount);
    std::vector<std::string> objects = Material::parseObjectsNames(name);
    if (composition.size() > objects.size())
        throw plask::Exception("Too long material composition vector (longer than number of objects in '%1%')", parsed_name_with_dopant);
    Material::Composition comp;
    for (std::size_t i = 0; i < composition.size(); ++i) comp[objects[i]] = composition[i];
    for (std::size_t i = composition.size(); i < objects.size(); ++i) comp[objects[i]] = std::numeric_limits<double>::quiet_NaN();
    return get(Material::completeComposition(comp), dopant, doping_amount_type, doping_amount);
}*/

shared_ptr<Material> MaterialsDB::get(const std::string& name_with_dopant, Material::DopingAmountType doping_amount_type, double doping_amount) const {
    Material::Parameters p(name_with_dopant, true);
    if (p.hasDopantName()) {
        p.dopantAmount = doping_amount;
        p.dopantAmountType = doping_amount_type;
    }
    return get(p);
}

shared_ptr<const MaterialsDB::MaterialConstructor> MaterialsDB::getConstructor(const std::string& name_without_composition) const {
    auto it = constructors.find(name_without_composition);  // try get as simple
    if (it != constructors.end()) return it->second;
    it = constructors.find(complexDbKey(name_without_composition)); // try get as complex
    if (it != constructors.end()) return it->second;
    throw NoSuchMaterial(name_without_composition);
}

shared_ptr<Material> MaterialsDB::get(const Material::Parameters &m) const {
    return (*getConstructor(m))(m.composition, m.dopantAmountType, m.dopantAmount);
}

shared_ptr< Material > MaterialsDB::get(const std::string& full_name) const {
    return get(Material::Parameters(full_name));
}

shared_ptr<MaterialsDB::MixedCompositionFactory> MaterialsDB::getFactory(const std::string& material1_fullname, const std::string& material2_fullname) const {
    Material::Parameters m1(material1_fullname), m2(material2_fullname);
    if (m1.dopantName != m2.dopantName)
        throw MaterialParseException("Cannot mix materials with different doping: '%1%' and '%2%'", material1_fullname, material2_fullname);
    if (m1.dopantAmountType != m2.dopantAmountType)
        throw MaterialParseException("Cannot mix materials for which doping is given in different formats: '%1%' and '%2%'", material1_fullname, material2_fullname);
    if ((m1.label != m2.label) || (m1.isSimple() != m2.isSimple()))
        throw MaterialParseException("Cannot mix different materials: '%1%' and '%2%'", material1_fullname, material2_fullname);

    if (m1.isSimple()) {  // simple material, without parsing composition, still dopants can be mixed
        if (m1.name != m2.name)
            throw MaterialParseException("Cannot mix different materials: '%1%' and '%2%'", material1_fullname, material2_fullname);

        if (!m1.hasDopant()) //??
            throw MaterialParseException("%1%: only complex or doped materials with different doping concentrations can be mixed", material1_fullname);

        return shared_ptr<MaterialsDB::MixedCompositionFactory>(
                    new MixedDopantFactory(getConstructor(m1), m1.dopantAmountType, m1.dopantAmount, m2.dopantAmount)
                    );
    }

    //complex materials:
    if (m1.hasDopant()) //both dopped
        return shared_ptr<MaterialsDB::MixedCompositionFactory>(
                    new MixedCompositionAndDopantFactory(getConstructor(m1),
                                                    m1.composition, m2.composition,
                                                    m1.dopantAmountType, m1.dopantAmount, m2.dopantAmount)
                    );

    //both undopped
    return shared_ptr<MaterialsDB::MixedCompositionFactory>(
            new MixedCompositionOnlyFactory(getConstructor(m1), m1.composition, m2.composition)
    );

    /*std::string m1comp, m1dop, m2comp, m2dop;
    std::tie(m1comp, m1dop) = splitString2(material1_fullname, ':');
    std::tie(m2comp, m2dop) = splitString2(material2_fullname, ':');
    std::string m1_dop_name, m2_dop_name;
    Material::DopingAmountType m1_dop_type = Material::NO_DOPING, m2_dop_type = Material::NO_DOPING;
    double m1_dop_am = 0.0, m2_dop_am = 0.0;
    if (m1dop != "") Material::parseDopant(m1dop, m1_dop_name, m1_dop_type, m1_dop_am);
    if (m2dop != "") Material::parseDopant(m2dop, m2_dop_name, m2_dop_type, m2_dop_am);
    if (m1_dop_name != m2_dop_name)
        throw MaterialParseException("Cannot mix materials with different doping: '%1%' and '%2%'", material1_fullname, material2_fullname);
    if (m1_dop_type != m2_dop_type)
        throw MaterialParseException("Cannot mix materials for which doping is given in different formats: '%1%' and '%2%'", material1_fullname, material2_fullname);
    return getFactory(m1comp, m2comp, m1_dop_name, m1_dop_type, m1_dop_am, m2_dop_am);*/
}

/*void MaterialsDB::addSimple(const MaterialConstructor* constructor) {
    constructors[constructor->materialName] = shared_ptr<const MaterialConstructor>(constructor);
}*/

void MaterialsDB::addSimple(shared_ptr<MaterialConstructor> constructor) {
    constructors[constructor->materialName] = constructor;
}

/*void MaterialsDB::addComplex(const MaterialConstructor* constructor) {
    constructors[dbKey(constructor->materialName)] = shared_ptr<const MaterialConstructor>(constructor);
}*/

void MaterialsDB::addComplex(shared_ptr<MaterialConstructor> constructor) {
    constructors[complexDbKey(constructor->materialName)] = constructor;
}

void MaterialsDB::removeSimple(const std::string& name) {
    constructors.erase(name);
}

void MaterialsDB::removeComplex(const std::string& name) {
    constructors.erase(complexDbKey(name));
}

bool MaterialsDB::isSimple(const std::string &material_name) const { return getConstructor(material_name)->isSimple(); }

}  // namespace plask

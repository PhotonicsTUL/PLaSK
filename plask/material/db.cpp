#include "db.h"

#include "../utils/string.h"
#include "../utils/dynlib/manager.h"

#include <boost/filesystem.hpp>

namespace plask {

void checkCompositionSimilarity(const Material::Composition& material1composition, const Material::Composition& material2composition) {
    for (auto& p1: material1composition) {
        auto p2 = material2composition.find(p1.first);
        if (p2 == material2composition.end())
            throw MaterialParseException("materials compositions are different: in first there is \"%1%\" element which is missing in second.", p1.first);
        if (std::isnan(p1.second) != std::isnan(p2->second))
            throw MaterialParseException("amounts must be defined for the same elements, which is not true in case of \"%1%\" element.", p1.first);
    }
}

MaterialsDB::MixedCompositionOnlyFactory::MixedCompositionOnlyFactory(const shared_ptr<const MaterialConstructor>& constructor, const Material::Composition& material1composition, const Material::Composition& material2composition)
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

std::string dbKey(const std::string& fullComplexName) {
    auto name_dopant = splitString2(fullComplexName, ':');
    return dbKey(name_dopant.first, name_dopant.second);
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

MaterialsDB& MaterialsDB::getDefault() {
    static MaterialsDB defaultDb;
    return defaultDb;
}

void MaterialsDB::loadToDefault(const std::string &fileName_mainpart) {
    DynamicLibraries::defaultLoad(plaskMaterialsPath() + fileName_mainpart + DynamicLibrary::DEFAULT_EXTENSION);
}

void MaterialsDB::loadAllToDefault(const std::string& dir) {
    boost::filesystem::directory_iterator iter(dir);
    boost::filesystem::directory_iterator end;
    while (iter != end) {
        boost::filesystem::path p = iter->path();
        if (boost::filesystem::is_regular_file(p))
            DynamicLibraries::defaultLoad(p.string());
        ++iter;
    }
}

void MaterialsDB::ensureCompositionIsNotEmpty(const Material::Composition &composition) {
    if (composition.empty()) throw MaterialParseException("unknown composition.");
}

shared_ptr<const MaterialsDB::MaterialConstructor> MaterialsDB::getConstructor(const std::string& db_Key, const Material::Composition& composition, const std::string& dopant_name) const {
    auto it = constructors.find(db_Key);
    if (it == constructors.end()) {
        if (composition.empty()) {
            //check if material is complex, but user forget to provide composition:
            std::string complexDbKey;
            try { complexDbKey = dbKey(db_Key); } catch (std::exception& e) {}
            if (constructors.find(complexDbKey) != constructors.end())  //material is complex
                throw MaterialParseException(format("composition is required to get \"%1%\" material.", db_Key));
            throw NoSuchMaterial(db_Key);
        }
        throw NoSuchMaterial(composition, dopant_name);
    }
    return it->second;
}

shared_ptr<Material> MaterialsDB::get(const std::string& db_Key, const Material::Composition& composition, const std::string& dopant_name, Material::DopingAmountType doping_amount_type, double doping_amount) const {
    return (*getConstructor(db_Key, composition, dopant_name))(composition, doping_amount_type, doping_amount);
}

shared_ptr<const MaterialsDB::MaterialConstructor> MaterialsDB::getConstructor(const Material::Composition& composition, const std::string& dopant_name) const {
    return getConstructor(dbKey(composition, dopant_name), composition, dopant_name);
}

shared_ptr<Material> MaterialsDB::get(const Material::Composition &composition, const std::string& dopant_name, Material::DopingAmountType doping_amount_type, double doping_amount) const {
    return get(dbKey(composition, dopant_name), composition, dopant_name, doping_amount_type, doping_amount);
}

shared_ptr<Material> MaterialsDB::get(const std::string& parsed_name_with_dopant, const std::vector<double>& composition,
                                               Material::DopingAmountType doping_amount_type, double doping_amount) const {
    std::string name, dopant;
    std::tie(name, dopant) = splitString2(parsed_name_with_dopant, ':');
    if (composition.empty())
        return get(parsed_name_with_dopant, Material::Composition(), dopant, doping_amount_type, doping_amount);
    std::vector<std::string> elements = Material::parseElementsNames(name);
    if (composition.size() > elements.size())
        throw plask::Exception("Too long composition vector (longer than number of elements in \"%1%\")", parsed_name_with_dopant);
    Material::Composition comp;
    for (std::size_t i = 0; i < composition.size(); ++i) comp[elements[i]] = composition[i];
    for (std::size_t i = composition.size(); i < elements.size(); ++i) comp[elements[i]] = std::numeric_limits<double>::quiet_NaN();
    return get(Material::completeComposition(comp), dopant, doping_amount_type, doping_amount);
}

shared_ptr<Material> MaterialsDB::get(const std::string& name_with_dopant, Material::DopingAmountType doping_amount_type, double doping_amount) const {
    std::string name_with_components, dopant_name;
    std::tie(name_with_components, dopant_name) = splitString2(name_with_dopant, ':');
    if (name_with_components.find('(') == std::string::npos) {  //simple case, without parsing composition
        std::string dbKey = name_with_components;
        appendDopant(dbKey, dopant_name);
        return get(dbKey, Material::Composition(), dopant_name, doping_amount_type, doping_amount);
    } else // parse composition:
        return get(Material::parseComposition(name_with_components), dopant_name, doping_amount_type, doping_amount);
}

shared_ptr< Material > MaterialsDB::get(const std::string& name_with_components, const std::string& doping_descr) const {
    std::string dopant_name;
    double doping_amount = 0.0;
    Material::DopingAmountType doping_amount_type = Material::NO_DOPING;
    if (!doping_descr.empty())
        Material::parseDopant(doping_descr, dopant_name, doping_amount_type, doping_amount);
    if (name_with_components.find('(') == std::string::npos) {  //simple case, without parsing composition
        std::string dbKey = name_with_components;
        appendDopant(dbKey, dopant_name);
        return get(dbKey, Material::Composition(), dopant_name, doping_amount_type, doping_amount);
    } else  //parse composition:
        return get(Material::parseComposition(name_with_components), dopant_name, doping_amount_type, doping_amount);
}

shared_ptr< Material > MaterialsDB::get(const std::string& full_name) const {
    auto pair = splitString2(full_name, ':');
    return get(pair.first, pair.second);
}

MaterialsDB::MixedCompositionFactory* MaterialsDB::getFactory(const std::string& material1name_with_components, const std::string& material2name_with_components,
                                                 const std::string& dopant_name, Material::DopingAmountType dopAmountType, double m1DopAmount, double m2DopAmount)
{
    if (material1name_with_components.find('(') == std::string::npos) {  //simple material, without parsing composition, stil dopants can be mixed
        if (material1name_with_components != material2name_with_components)
            throw MaterialCantBeMixedException(material1name_with_components, material2name_with_components, dopant_name);
        if (dopAmountType == Material::NO_DOPING || m1DopAmount == m2DopAmount)
            throw MaterialParseException("%1%: only complex or doping materials with different amount of dopant can be mixed.", material1name_with_components);
        std::string dbKey = material1name_with_components;
        appendDopant(dbKey, dopant_name);
        return new MixedDopantFactory(getConstructor(dbKey, Material::Composition(), dopant_name), dopAmountType, m1DopAmount, m2DopAmount);
    }
    //complex materials:
    if (material2name_with_components.find('(') == std::string::npos)   //mix complex with simple?
        throw MaterialCantBeMixedException(material1name_with_components, material2name_with_components, dopant_name);

    return getFactory(Material::parseComposition(material1name_with_components),
                      Material::parseComposition(material2name_with_components),
                      dopant_name, dopAmountType, m1DopAmount, m2DopAmount);
}

MaterialsDB::MixedCompositionFactory* MaterialsDB::getFactory(const Material::Composition& material1composition, const Material::Composition& material2composition) {
    return new MixedCompositionOnlyFactory(getConstructor(material1composition), material1composition, material2composition);
}

MaterialsDB::MixedCompositionFactory* MaterialsDB::getFactory(const Material::Composition& material1composition, const Material::Composition& material2composition,
                                 const std::string& dopant_name, Material::DopingAmountType dopAmountType, double m1DopAmount, double m2DopAmount)
{
    if (dopAmountType == Material::NO_DOPING)
        getFactory(material1composition, material2composition);
    return new MixedCompositionAndDopantFactory(getConstructor(material1composition, dopant_name),
                                                material1composition, material2composition,
                                                dopAmountType, m1DopAmount, m2DopAmount);
}

MaterialsDB::MixedCompositionFactory* MaterialsDB::getFactory(const std::string& material1_fullname, const std::string& material2_fullname) {
    std::string m1comp, m1dop, m2comp, m2Dop;
    std::tie(m1comp, m1dop) = splitString2(material1_fullname, ':');
    std::tie(m2comp, m2Dop) = splitString2(material2_fullname, ':');
    std::string m1_dop_name, m2_dop_name;
    Material::DopingAmountType m1_dop_type = Material::NO_DOPING, m2_dop_type = Material::NO_DOPING;
    double m1_dop_am = 0.0, m2_dop_am = 0.0;
    Material::parseDopant(m1dop, m1_dop_name, m1_dop_type, m1_dop_am);
    Material::parseDopant(m2Dop, m2_dop_name, m2_dop_type, m2_dop_am);
    if (m1_dop_name != m2_dop_name)
        throw MaterialParseException("can't mix materials with different doping: \"%1%\" and \"%2%\"", material1_fullname, material2_fullname);
    if (m1_dop_type != m2_dop_type)
        throw MaterialParseException("can't mix materials for which amounts of dopings are given in different format: \"%1%\" and \"%2%\".", material1_fullname, material2_fullname);
    return getFactory(m1comp, m2comp, m1_dop_name, m1_dop_type, m1_dop_am, m2_dop_am);
}

void MaterialsDB::addSimple(const MaterialConstructor* constructor) {
    constructors[constructor->materialName] = shared_ptr<const MaterialConstructor>(constructor);
}

void MaterialsDB::addSimple(const shared_ptr<const MaterialConstructor>& constructor) {
    constructors[constructor->materialName] = constructor;
}

void MaterialsDB::addComplex(const MaterialConstructor* constructor) {
    constructors[dbKey(constructor->materialName)] = shared_ptr<const MaterialConstructor>(constructor);
}

void MaterialsDB::addComplex(const shared_ptr<const MaterialConstructor>& constructor) {
    constructors[dbKey(constructor->materialName)] = constructor;
}

void MaterialsDB::removeSimple(const std::string& name) {
    constructors.erase(name);
}

void MaterialsDB::removeComplex(const std::string& name) {
    constructors.erase(dbKey(name));
}


}  // namespace plask

/*
 * This file is part of PLaSK (https://plask.app) by Photonics Group at TUL
 * Copyright (c) 2022 Lodz University of Technology
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 */
#include "db.hpp"
#include "mixed.hpp"
#include "const_material.hpp"

#include "../utils/string.hpp"
#include "../utils/dynlib/manager.hpp"
#include "../log/log.hpp"

#include <boost/filesystem.hpp>

namespace plask {

void checkCompositionSimilarity(const Material::Composition& material1composition, const Material::Composition& material2composition) {
    for (auto& p1: material1composition) {
        auto p2 = material2composition.find(p1.first);
        if (p2 == material2composition.end())
            throw MaterialParseException("materials compositions are different: {0} if missing from one of the materials", p1.first);
        if (std::isnan(p1.second) != std::isnan(p2->second))
            throw MaterialParseException("amounts must be defined for the same elements, which is not true in case of '{0}' element", p1.first);
    }
}

/*const MaterialsDB *MaterialsDB::getFromSource(const MaterialsDB &materialsDB) {
    const MaterialsDB::Source* src = materialsDB.target<const MaterialsDB::Source>();
    return src ? &src->materialsDB : nullptr;
}*/

MaterialsDB::MixedCompositionOnlyFactory::MixedCompositionOnlyFactory(shared_ptr<const MaterialConstructor> constructor, const Material::Composition& material1composition, const Material::Composition& material2composition, double shape)
    : MaterialsDB::MixedCompositionFactory::MixedCompositionFactory(constructor), material1composition(material1composition), material2composition(material2composition), shape(shape) {
    //check if compositions are fine and similar:
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
            p1.second = p1.second * pow(m1_weight, shape) + p2->second * (1.0 - pow(m1_weight, shape));
        }
    }
    return result;
}

// if part_value is not empty append separator and part_value to name
std::string& appendPart(std::string& name, const std::string& part_value, char separator) {
    if (!part_value.empty()) {
        name += separator;
        name += part_value;
    }
    return name;
}

// Append to name dopant, if it is not empty, return name
std::string& appendDopant(std::string& name, const std::string& dopant_name) {
    return appendPart(name, dopant_name, ':');
}

std::string& appendLabel(std::string& name, const std::string& label) {
    return appendPart(name, label, '_');
}

std::string& appendLabelDopant(std::string& name, const std::string& label, const std::string& dopant_name) {
    return  appendDopant(appendLabel(name, label), dopant_name);
}

std::string alloyDbKey(const Material::Composition &composition, const std::string& label, const std::string& dopant_name) {
    std::string db_key;
    for (auto c: composition) db_key += c.first;
    return appendLabelDopant(db_key, label, dopant_name);
}

std::string alloyDbKey(std::vector<std::string> elNames, const std::string& label, const std::string& dopant_name) {
    std::string db_key;
    std::sort(elNames.begin(), elNames.end());
    for (std::string& c: elNames) db_key += c;
    return appendLabelDopant(db_key, label, dopant_name);
}

std::string alloyDbKey(const std::string& name, const std::string& label, const std::string& dopant_name) {
    return alloyDbKey(Material::parseObjectsNames(name), label, dopant_name);
}

std::string alloyDbKey(const std::string& fullAlloyName) {
    auto fullname_dopant = splitString2(fullAlloyName, ':');
    auto name_label = splitString2(fullname_dopant.first, '_');
    return alloyDbKey(name_label.first, name_label.second, fullname_dopant.second);
}

std::string dbKey(const Material::Parameters& parameters) {
    std::string res;
    if (!parameters.isAlloy())
        res = parameters.name;
    else
        for (auto c: parameters.composition) res += c.first;
    return appendLabelDopant(res, parameters.label, parameters.dopant);
}


MaterialsDB& MaterialsDB::getDefault() {
    static MaterialsDB defaultDb;
    return defaultDb;
}

static void loadLibrary(const std::string& file_name) {
    static std::map<void*, MaterialsDB> libraryCache;
    void* key;
    {
        MaterialsDB::TemporaryClearDefault temporary;
        key = DynamicLibraries::defaultLoad(file_name, DynamicLibrary::DONT_CLOSE).getHandle();
        if (libraryCache.find(key) == libraryCache.end()) libraryCache[key] = MaterialsDB::getDefault();
    }
    MaterialsDB::getDefault().update(libraryCache[key]);
}

void MaterialsDB::loadToDefault(const std::string &fileName_mainpart) {
    loadLibrary(boost::filesystem::absolute(fileName_mainpart + DynamicLibrary::DEFAULT_EXTENSION, boost::filesystem::current_path()).string<std::string>());
}

void MaterialsDB::loadAllToDefault(const std::string& dir) {
    if (boost::filesystem::exists(dir) && boost::filesystem::is_directory(dir)) {
        boost::filesystem::directory_iterator iter(dir);
        boost::filesystem::directory_iterator end;
        while (iter != end) {
            boost::filesystem::path p = iter->path();
            if (boost::filesystem::is_regular_file(p) && p.extension() == DynamicLibrary::DEFAULT_EXTENSION)
                loadLibrary(p.string());
            ++iter;
        }
    } else {
        writelog(LOG_WARNING, "MaterialsDB: '{0}' does not exist or is not a directory. Cannot load default materials", dir);
    }
}

void MaterialsDB::ensureCompositionIsNotEmpty(const Material::Composition &composition) {
    if (composition.empty()) throw MaterialParseException("unknown material composition");
}

MaterialsDB::ProxyMaterialConstructor::ProxyMaterialConstructor():
    MaterialsDB::MaterialConstructor(""), material(new GenericMaterial)
{}

MaterialsDB::ProxyMaterialConstructor::ProxyMaterialConstructor(const std::string& name, const MaterialsDB& db):
    MaterialsDB::MaterialConstructor(name)
{
    if (name.empty()) {
        material = plask::make_shared<GenericMaterial>();
    } else {
        try {
            material = db.get(name);
        } catch (plask::MaterialParseException&) {
            Material::Parameters p(name, true);
            constructor = db.getConstructor(p, true);
            composition = p.composition;
            doping = p.doping;
        }
    }
    assert(material || constructor);
}

MaterialsDB::ProxyMaterialConstructor::ProxyMaterialConstructor(const shared_ptr<Material>& material):
    MaterialsDB::MaterialConstructor(material->name()), material(material)
{}

shared_ptr<Material> MaterialsDB::ProxyMaterialConstructor::operator()(const Material::Composition& comp, double dop) const {
    if (material) {
        return material;
    }
    if (!isnan(doping)) dop = doping;
    if (composition.empty()) {
        return (*constructor)(comp, dop);
    } else {
        return (*constructor)(composition, dop);
    }
}

bool MaterialsDB::ProxyMaterialConstructor::isAlloy() const {
    if (material || !composition.empty() || !constructor) return false;
    return constructor->isAlloy();
}


shared_ptr<const MaterialsDB::MaterialConstructor> MaterialsDB::getConstructor(const std::string& db_Key, const Material::Composition& composition, bool allow_alloy_without_composition) const {
    auto it = constructors.find(db_Key);
    if (it == constructors.end()) {
        if (composition.empty()) {
            // check if material is alloy, but user forgot to provide composition:
            std::string alloy_DbKey;
            try { alloy_DbKey = alloyDbKey(db_Key); } catch (std::exception&) {}
            auto c = constructors.find(alloy_DbKey);
            if (c != constructors.end()) { //material is alloy
                if (allow_alloy_without_composition)
                    return c->second;
                else
                    throw MaterialParseException(format("material composition required for {0}", db_Key));
            } else
                throw NoSuchMaterial(db_Key);
        }
        // throw NoSuchMaterial(composition, dopant_name);
        throw NoSuchMaterial(db_Key + " (alloy)");
    }
    return it->second;
}

shared_ptr<Material> MaterialsDB::get(const std::string& db_Key, const Material::Composition& composition, double doping) const {
    return (*getConstructor(db_Key, composition))(composition, doping);
}

shared_ptr<const MaterialsDB::MaterialConstructor> MaterialsDB::getConstructor(const Material::Composition& composition, const std::string& label, const std::string& dopant_name) const {
    return getConstructor(alloyDbKey(composition, label, dopant_name), composition);
}

shared_ptr<const MaterialsDB::MaterialConstructor> MaterialsDB::getConstructor(const Material::Parameters &material, bool allow_alloy_without_composition) const
{
    return getConstructor(dbKey(material), material.composition, allow_alloy_without_composition);
}

shared_ptr<Material> MaterialsDB::get(const Material::Composition &composition, const std::string& label, const std::string& dopant_name, double doping) const {
    return get(alloyDbKey(composition, label, dopant_name), composition, doping);
}

/*shared_ptr<Material> MaterialsDB::get(const std::string& parsed_name_with_dopant, const std::vector<double>& composition, double doping) const {
    std::string name, dopant;
    std::tie(name, dopant) = splitString2(parsed_name_with_dopant, ':');
    if (composition.empty())
        return get(parsed_name_with_dopant, Material::Composition(), dopant, doping);
    std::vector<std::string> objects = Material::parseObjectsNames(name);
    if (composition.size() > objects.size())
        throw plask::Exception("too long material composition vector (longer than number of objects in '{0}')", parsed_name_with_dopant);
    Material::Composition comp;
    for (std::size_t i = 0; i < composition.size(); ++i) comp[objects[i]] = composition[i];
    for (std::size_t i = composition.size(); i < objects.size(); ++i) comp[objects[i]] = std::numeric_limits<double>::quiet_NaN();
    return get(Material::completeComposition(comp), dopant, doping);
}*/

shared_ptr<Material> MaterialsDB::get(const std::string& name_with_dopant, double doping) const {
    Material::Parameters p(name_with_dopant, true);
    if (p.hasDopantName()) {
        p.doping = doping;
    }
    return get(p);
}

shared_ptr<const MaterialsDB::MaterialConstructor> MaterialsDB::getConstructor(const std::string& name_without_composition) const {
    auto it = constructors.find(name_without_composition);  // try get as simple
    if (it != constructors.end()) return it->second;
    it = constructors.find(alloyDbKey(name_without_composition)); // try get as alloy
    if (it != constructors.end()) return it->second;
    throw NoSuchMaterial(name_without_composition);
}

shared_ptr<Material> MaterialsDB::get(const Material::Parameters &m) const {
    return (*getConstructor(m))(m.composition, m.doping);
}

shared_ptr< Material > MaterialsDB::get(const std::string& full_name) const {
    if (full_name.size() != 0 && full_name.find('[') != std::string::npos && full_name[full_name.size()-1] == ']')
        return plask::make_shared<ConstMaterial>(full_name);
    else
        return get(Material::Parameters(full_name));
}

shared_ptr<MaterialsDB::MixedCompositionFactory> MaterialsDB::getFactory(const std::string& material1_fullname, const std::string& material2_fullname, double shape) const {
    Material::Parameters m1(material1_fullname), m2(material2_fullname);
    if (m1.dopant != m2.dopant)
        throw MaterialParseException("cannot mix materials with different doping: '{0}' and '{1}'", material1_fullname, material2_fullname);
    if ((m1.label != m2.label) || (m1.isAlloy() != m2.isAlloy()))
        throw MaterialParseException("cannot mix different materials: '{0}' and '{1}'", material1_fullname, material2_fullname);

    if (!m1.isAlloy()) {  // simple material, without parsing composition, still dopants can be mixed
        if (m1.name != m2.name)
            throw MaterialParseException("cannot mix different materials: '{0}' and '{1}'", material1_fullname, material2_fullname);

        if (!m1.hasDoping()) //??
            throw MaterialParseException("{0}: only alloy or doped materials with different doping concentrations can be mixed", material1_fullname);

        return shared_ptr<MaterialsDB::MixedCompositionFactory>(
                    new MixedDopantFactory(getConstructor(m1), m1.doping, m2.doping, shape)
                    );
    }

    //alloy materials:
    if (m1.hasDoping()) //both dopped
        return shared_ptr<MaterialsDB::MixedCompositionFactory>(
                    new MixedCompositionAndDopantFactory(getConstructor(m1),
                                                    m1.composition, m2.composition,
                                                    m1.doping, m2.doping, shape)
                    );

    //both undopped
    return shared_ptr<MaterialsDB::MixedCompositionFactory>(
            new MixedCompositionOnlyFactory(getConstructor(m1), m1.composition, m2.composition, shape)
    );
}

/*void MaterialsDB::addSimple(const MaterialConstructor* constructor) {
    constructors[constructor->materialName] = shared_ptr<const MaterialConstructor>(constructor);
}*/

void MaterialsDB::addSimple(shared_ptr<MaterialConstructor> constructor) {
    constructors[constructor->materialName] = constructor;
}

/*void MaterialsDB::addAlloy(const MaterialConstructor* constructor) {
    constructors[dbKey(constructor->materialName)] = shared_ptr<const MaterialConstructor>(constructor);
}*/

void MaterialsDB::addAlloy(shared_ptr<MaterialConstructor> constructor) {
    constructors[alloyDbKey(constructor->materialName)] = constructor;
}

void MaterialsDB::remove(const std::string& name) {
    auto it = constructors.find(name);  // try get as simple
    if (it != constructors.end()) {
        constructors.erase(it);
        return;
    }
    it = constructors.find(alloyDbKey(name)); // try get as alloy
    if (it != constructors.end()) {
        constructors.erase(it);
        return;
    }
    throw NoSuchMaterial(name);
}


bool MaterialsDB::isAlloy(const std::string &material_name) const { return getConstructor(material_name)->isAlloy(); }

}  // namespace plask

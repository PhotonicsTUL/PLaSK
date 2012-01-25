#include "material.h"

#include "../utils/string.h"
#include <boost/lexical_cast.hpp>

#include <cmath>

namespace plask {

double Material::A(double T) const { throwNotImplemented("A(double T)"); assert(0); }

double Material::absp(double wl, double T) const { throwNotImplemented("absp(double wl, double T)"); assert(0); }

double Material::B(double T) const { throwNotImplemented("B(double T)"); assert(0); }

double Material::C(double T) const { throwNotImplemented("C(double T)"); assert(0); }

double Material::CBO(double T, char point) const { throwNotImplemented("CBO(double T, char point)"); assert(0); }

double Material::chi(double T, char point) const { throwNotImplemented("chi(double T, char point)"); assert(0); }

double Material::cond(double T) const { throwNotImplemented("cond(double T)"); assert(0); }
double Material::cond_l(double T) const { throwNotImplemented("cond_l(double T)"); assert(0); }
double Material::cond_v(double T) const { throwNotImplemented("cond_v(double T)"); assert(0); }

double Material::D(double T) const { throwNotImplemented("D(double T)"); assert(0); }

double Material::dens(double T) const { throwNotImplemented("dens(double T)"); assert(0); }

double Material::Dso(double T) const { throwNotImplemented("Dso(double T)"); assert(0); }

double Material::EactA(double T) const { throwNotImplemented("EactA(double T)"); assert(0); }
double Material::EactD(double T) const { throwNotImplemented("EactD(double T)"); assert(0); }

double Material::Eg(double T, char point) const { throwNotImplemented("Eg(double T, char point)"); assert(0); }

double Material::eps(double T) const { throwNotImplemented("eps(double T)"); assert(0); }

double Material::lattC(double T, char x) const { throwNotImplemented("lattC(double T, char x)"); assert(0); }

double Material::Me(double T, char point) const { throwNotImplemented("Me(double T, char point)"); assert(0); }
double Material::Me_l(double T, char point) const { throwNotImplemented("Me_l(double T, char point)"); assert(0); }
double Material::Me_v(double T, char point) const { throwNotImplemented("Me_v(double T, char point)"); assert(0); }

double Material::Mh(double T, char EqType) const { throwNotImplemented("Mh(double T, char EqType)"); assert(0); }
double Material::Mh_l(double T, char point) const { throwNotImplemented("Mh_l(double T, char point)"); assert(0); }
double Material::Mh_v(double T, char point) const { throwNotImplemented("Mh_v(double T, char point)"); assert(0); }

double Material::Mhh(double T, char point) const { throwNotImplemented("Mhh(double T, char point)"); assert(0); }
double Material::Mhh_l(double T, char point) const { throwNotImplemented("Mhh_l(double T, char point)"); assert(0); }
double Material::Mhh_v(double T, char point) const { throwNotImplemented("Mhh_v(double T, char point)"); assert(0); }

double Material::Mlh(double T, char point) const { throwNotImplemented("B(double T)"); assert(0); }
double Material::Mlh_l(double T, char point) const { throwNotImplemented("B(double T)"); assert(0); }
double Material::Mlh_v(double T, char point) const { throwNotImplemented("B(double T)"); assert(0); }

double Material::mob(double T) const { throwNotImplemented("mob(double T)"); assert(0); }

double Material::Mso(double T) const { throwNotImplemented("Mso(double T)"); assert(0); }

double Material::Nc(double T, char point) const { throwNotImplemented("Nc(double T, char point)"); assert(0); }
double Material::Nc(double T) const { throwNotImplemented("Nc(double T)"); assert(0); }

double Material::Nf(double T) const { throwNotImplemented("Nf(double T)"); assert(0); }

double Material::Ni(double T) const { throwNotImplemented("Ni(double T)"); assert(0); }

double Material::nr(double wl, double T) const { throwNotImplemented("nr(double wl, double T)"); assert(0); }

dcomplex Material::Nr(double wl, double T) const { throwNotImplemented("Nr(double wl, double T)"); assert(0); }

double Material::res(double T) const { throwNotImplemented("res(double T)"); assert(0); }
double Material::res_l(double T) const { throwNotImplemented("res_l(double T)"); assert(0); }
double Material::res_v(double T) const { throwNotImplemented("res_v(double T)"); assert(0); }

double Material::specHeat(double T) const { throwNotImplemented("specHeat(double T)"); assert(0); }

double Material::thermCond(double T, double thickness) const { throwNotImplemented("thermCond(double T, double thickness)"); assert(0); }
double Material::thermCond_l(double T, double thickness) const { throwNotImplemented("thermCond_l(double T, double thickness)"); assert(0); }
double Material::thermCond_v(double T, double thickness) const { throwNotImplemented("thermCond_v(double T, double thickness)"); assert(0); }

double Material::VBO(double T) const { throwNotImplemented("VBO(double T)"); assert(0); }

void Material::throwNotImplemented(const std::string& method_name) const {
    throw MaterialMethodNotImplemented(name(), method_name);
};

inline void fillGroupMaterialCompositionAmounts(std::vector<double>::iterator begin, std::vector<double>::iterator end) {
    auto no_info = end;
    double sum = 0.0;
    unsigned n = 0;
    for (auto i = begin; i != end; ++i) {
        if (std::isnan(*i)) {
            if (no_info != end)
                throw plask::MaterialParseException("More than one element in group have no information about composition amount");
            else
                no_info = i;
        } else {
            sum += *i;
            ++n;
        }
    }
    if (n > 0 && sum - 1.0 > SMALL*n)
        throw plask::MaterialParseException("Sum of composition ammounts in group exceeds 1");
    if (no_info != end) {
        *no_info = 1.0 - sum;
    } else {
        if (!is_zero(sum - 1.0))
             throw plask::MaterialParseException("Sum of composition ammounts in group diffrent from 1");
    }
}

inline void fillMaterialCompositionAmountsI(std::vector< double >& composition, unsigned int pattern) {
    auto end = composition.end();
    while (pattern != 0) {
        unsigned group_size = pattern % 10;     // last group size
        if (end - composition.begin() < group_size)
            throw plask::CriticalException("Wrong material composition pattern");
        auto begin = end - group_size;
        fillGroupMaterialCompositionAmounts(begin, end);
        end = begin;
        pattern /= 10;
    }
    if (end != composition.begin())
        throw plask::CriticalException("Wrong material composition pattern");
}

std::vector< double > Material::completeComposition(const std::vector< double >& composition, unsigned int pattern) {
    std::vector<double> result = composition;
    fillMaterialCompositionAmountsI(result, pattern);
    return result;
}


shared_ptr< Material > plask::MaterialsDB::get(const std::string& parsed_name_with_donor, const std::vector< double >& composition,
                                               DOPING_AMOUNT_TYPE doping_amount_type, double doping_amount) const
{
    auto it = constructors.find(parsed_name_with_donor);
    if (it == constructors.end()) throw NoSuchMaterial(parsed_name_with_donor);
    return (*it->second)(composition, doping_amount_type, doping_amount);
}

const char* getElementEnd(const char* begin, const char* end) {
    if (!('A' <= *begin && *begin <= 'Z')) return begin;
    do { ++begin; } while (begin != end && 'a' <= *begin && *begin <= 'z');
    return begin;
}

const char* getAmountEnd(const char* begin, const char* end) {
    if (*begin != '(') return begin;
    do { ++begin; } while (begin != end && *begin != ')');
    return begin;
}

double toDouble(const std::string& s) {
    try {
        return boost::lexical_cast<double>(s);
    } catch (std::exception& e) {
        throw MaterialParseException(e.what());
    }
}

void parseNameWithComposition(const char* begin, const char* end, std::vector<std::string>& components, std::vector<double>& components_amounts) {
    while (begin != end) {
        const char* comp_end = getElementEnd(begin, end);
        if (comp_end == begin)
            throw MaterialParseException(std::string("Expected element but found character: ") + *begin);
        components.push_back(std::string(begin, comp_end));
        const char* amount_end = getAmountEnd(comp_end, end);
        if (amount_end == comp_end) {       //no amount info for this element
            components_amounts.push_back(std::numeric_limits<double>::quiet_NaN());
            begin = amount_end;
        } else {
            if (amount_end == end)
                throw MaterialParseException("Unexpected end of input while reading amount of element. Couldn't find ')'");
            components_amounts.push_back(toDouble(std::string(comp_end+1, amount_end)));
            begin = amount_end+1;   //skip also ')', begin now points to 1 character after ')'
        }
    }
}

void parseDopant(const char* begin, const char* end, std::string& dopant_elem_name, MaterialsDB::DOPING_AMOUNT_TYPE& doping_amount_type, double& doping_amount) {
    const char* name_end = getElementEnd(begin, end);
    if (name_end == begin)
         throw MaterialParseException("No dopant name");
    dopant_elem_name.assign(begin, name_end);
    if (*name_end == '=') {
        if (name_end+1 == end) throw MaterialParseException("Unexpected end of input while reading dopants concentation");
        doping_amount_type = MaterialsDB::DOPANT_CONCENTRATION;
        doping_amount = toDouble(std::string(name_end+1, end));
        return;
    } else if (name_end+1 == end) { // there might be some reason to specify material with dopant but undoped (can be caught in material constructor)
        doping_amount_type = MaterialsDB::NO_DOPING;
        doping_amount = 0.;
        return;
    }
    if (!isspace(*name_end))
        throw MaterialParseException("Expected space or '=' but found '%1%' instead", *name_end);
    do {  ++name_end; } while (name_end != end && isspace(*name_end));   //skip whites
    auto p = splitString2(std::string(name_end, end), '=');
    //TODO check std::get<0>(p) if is p/n compatibile with dopant_elem_name
    doping_amount_type = MaterialsDB::CARRIER_CONCENTRATION;
    doping_amount = toDouble(std::get<1>(p));
}

shared_ptr< Material > MaterialsDB::get(const std::string& name_with_components, const std::string& doping_descr) const {

    std::vector<std::string> components;
    std::vector<double> components_amounts;
    parseNameWithComposition(name_with_components.data(), name_with_components.data() + name_with_components.size(), components, components_amounts);

    std::string parsed_name_with_dopant;
    for (std::string& c: components) parsed_name_with_dopant += c;

    double doping_amount = 0.0;
    DOPING_AMOUNT_TYPE doping_amount_type = NO_DOPING;
    if (!doping_descr.empty()) {
        std::string dopant_name;
        parseDopant(doping_descr.data(), doping_descr.data() + doping_descr.size(), dopant_name, doping_amount_type, doping_amount);
        parsed_name_with_dopant += ':';
        parsed_name_with_dopant += dopant_name;
    }

    return get(parsed_name_with_dopant, components_amounts, doping_amount_type, doping_amount);
}

shared_ptr< Material > MaterialsDB::get(const std::string& full_name) const {
    auto pair = splitString2(full_name, ':');
    return get(std::get<0>(pair), std::get<1>(pair));
}

void MaterialsDB::add(const std::string& name, const MaterialConstructor* constructor) {
    constructors[name] = std::unique_ptr<const MaterialConstructor>(constructor);
}

void MaterialsDB::add(const std::string& name, plask::MaterialsDB::construct_material_f* constructor) {
    add(name, new FunctionBasedMaterialConstructor(constructor));
}


}       // namespace plask

#include "material.h"

#include <boost/lexical_cast.hpp>
#include "../utils/stl.h"
#include "../utils/string.h"

#include <cmath>
#include <set>

namespace plask {

inline std::pair<std::string, int> el_g(const std::string& g, int p) { return std::pair<std::string, int>(g, p); }

int elementGroup(const std::string& elementName) {
    static const std::map<std::string, int> elementGroups =
        { el_g("Al", 3), el_g("Ga", 3), el_g("In", 3),
          el_g("N", 5), el_g("P", 5), el_g("As", 5) };
    return map_find(elementGroups, elementName, 0);
}


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

dcomplex Material::Nr(double wl, double T) const { return dcomplex(nr(wl,T), -7.95774715459e-09*absp(wl,T)*wl); }

double Material::res(double T) const { throwNotImplemented("res(double T)"); assert(0); }
double Material::res_l(double T) const { throwNotImplemented("res_l(double T)"); assert(0); }
double Material::res_v(double T) const { throwNotImplemented("res_v(double T)"); assert(0); }

double Material::specHeat(double T) const { throwNotImplemented("specHeat(double T)"); assert(0); }

double Material::condT(double T) const { throwNotImplemented("condT(double T)"); assert(0); }
double Material::condT(double T, double thickness) const { throwNotImplemented("condT(double T, double t)"); assert(0); }
double Material::condT_l(double T, double thickness) const { throwNotImplemented("condT_l(double T, double thickness)"); assert(0); }
double Material::condT_v(double T, double thickness) const { throwNotImplemented("condT_v(double T, double thickness)"); assert(0); }

double Material::VBO(double T) const { throwNotImplemented("VBO(double T)"); assert(0); }

void Material::throwNotImplemented(const std::string& method_name) const {
    throw MaterialMethodNotImplemented(name(), method_name);
}

template <typename NameValuePairIter>
inline void fillGroupMaterialCompositionAmounts(NameValuePairIter begin, NameValuePairIter end, int group_nr) {
    auto no_info = end;
    double sum = 0.0;
    unsigned n = 0;
    for (auto i = begin; i != end; ++i) {
        if (std::isnan(i->second)) {
            if (no_info != end)
                throw plask::MaterialParseException("More than one element in group (%1% in periodic table) have no information about composition amount.", group_nr);
            else
                no_info = i;
        } else {
            sum += i->second;
            ++n;
        }
    }
    if (n > 0 && sum - 1.0 > SMALL*n)
        throw plask::MaterialParseException("Sum of composition ammounts in group (%1% in periodic table) exceeds 1.", group_nr);
    if (no_info != end) {
        no_info->second = 1.0 - sum;
    } else {
        if (!is_zero(sum - 1.0))
             throw plask::MaterialParseException("Sum of composition ammounts in group (%1% in periodic table) diffrent from 1.", group_nr);
    }
}

Material::Composition Material::completeComposition(const Composition &composition) {
    std::map<int, std::vector< std::pair<std::string, double> > > by_group;
    for (auto c: composition) {
        int group = elementGroup(c.first);
        if (group == 0) throw plask::MaterialParseException("Wrong element name \"%1%\".", c.first);
        by_group[group].push_back(c);
    }
    Material::Composition result;
    for (auto g: by_group) {
        fillGroupMaterialCompositionAmounts(g.second.begin(), g.second.end(), g.first);
        result.insert(g.second.begin(), g.second.end());
    }
    return result;
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

std::pair<std::string, double> Material::getFirstCompositionElement(const char*& begin, const char* end) {
    std::pair<std::string, double> result;
    const char* comp_end = getElementEnd(begin, end);
    if (comp_end == begin)
        throw MaterialParseException(std::string("Expected element but found character: ") + *begin);
    result.first = std::string(begin, comp_end);
    const char* amount_end = getAmountEnd(comp_end, end);
    if (amount_end == comp_end) {       //no amount info for this element
        result.second = std::numeric_limits<double>::quiet_NaN();
        begin = amount_end;
    } else {
        if (amount_end == end)
            throw MaterialParseException("Unexpected end of input while reading amount of element. Couldn't find ')'");
        result.second = toDouble(std::string(comp_end+1, amount_end));
        begin = amount_end+1;   //skip also ')', begin now points to 1 character after ')'
    }
    return result;
}


Material::Composition Material::parseComposition(const char* begin, const char* end) {
    const char* fullname = begin;   //for excpetions only
    Material::Composition result;
    std::set<int> groups;
    int prev_g = -1;
    while (begin != end) {
        auto c = getFirstCompositionElement(begin, end);
        int g = elementGroup(c.first);
        if (g != prev_g) {
            if (!groups.insert(g).second)
                throw MaterialParseException("Incorrect elements order in \"%1%\".", fullname);
            prev_g = g;
        }
        result.insert(c);
    }
    return result;
}

Material::Composition Material::parseComposition(const std::string& str) {
    const char* c = str.data();
    return parseComposition(c, c + str.size());
}

void Material::parseDopant(const char* begin, const char* end, std::string& dopant_elem_name, Material::DOPING_AMOUNT_TYPE& doping_amount_type, double& doping_amount) {
    const char* name_end = getElementEnd(begin, end);
    if (name_end == begin)
         throw MaterialParseException("No dopant name");
    dopant_elem_name.assign(begin, name_end);
    if (*name_end == '=') {
        if (name_end+1 == end) throw MaterialParseException("Unexpected end of input while reading dopants concentation");
        doping_amount_type = Material::DOPANT_CONCENTRATION;
        doping_amount = toDouble(std::string(name_end+1, end));
        return;
    } else if (name_end+1 == end) { // there might be some reason to specify material with dopant but undoped (can be caught in material constructor)
        doping_amount_type = Material::NO_DOPING;
        doping_amount = 0.;
        return;
    }
    if (!isspace(*name_end))
        throw MaterialParseException("Expected space or '=' but found '%1%' instead", *name_end);
    do {  ++name_end; } while (name_end != end && isspace(*name_end));   //skip whites
    auto p = splitString2(std::string(name_end, end), '=');
    //TODO check std::get<0>(p) if is p/n compatibile with dopant_elem_name
    doping_amount_type = Material::CARRIER_CONCENTRATION;
    doping_amount = toDouble(std::get<1>(p));
}

void Material::parseDopant(const std::string &dopant, std::string &dopant_elem_name, Material::DOPING_AMOUNT_TYPE &doping_amount_type, double &doping_amount) {
    const char* c = dopant.data();
    parseDopant(c, c + dopant.size(), dopant_elem_name, doping_amount_type, doping_amount);
}

std::vector<std::string> Material::parseElementsNames(const char *begin, const char *end) {
    const char* full_name = begin;  //store for error msg. only
    std::vector<std::string> elemenNames;
    do {
        const char* new_begin = getElementEnd(begin, end);
        if (new_begin == begin) throw MaterialParseException("Ill-formated name \"%1%\".", std::string(full_name, end));
        elemenNames.push_back(std::string(begin, new_begin));
        begin = new_begin;
    } while (begin != end);
    return elemenNames;
}

std::vector<std::string> Material::parseElementsNames(const std::string &allNames) {
    const char* c = allNames.c_str();
    return parseElementsNames(c, c + allNames.size());
}


//------------ MixedMaterial -------------------------

void MixedMaterial::normalizeWeights() {
    double sum = 0;
    for (auto& p: materials) sum += std::get<1>(p);
    for (auto& p: materials) std::get<1>(p) /= sum;
}

MixedMaterial & MixedMaterial::add(const shared_ptr<plask::Material> &material, double weight) {
    materials.push_back(std::tuple <shared_ptr<Material>, double>(material, weight));
    return *this;
}

std::string MixedMaterial::name() const {
    std::string result = "[mixed material consists of: ";
    for (auto i = materials.begin(); i != materials.end(); ++i) {
        if (i != materials.begin()) result += " + ";
        result += boost::lexical_cast<std::string>(std::get<1>(*i));
        result += '*';
        result += std::get<0>(*i)->name();
    }
    result += ']';
    return result;
}

}   // namespace plask

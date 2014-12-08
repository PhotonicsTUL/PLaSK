#include "material.h"

#include <boost/lexical_cast.hpp>
#include "../utils/stl.h"
#include "../utils/string.h"
#include "../log/log.h"
#include "db.h"

#include <cmath>
#include <set>

namespace plask {

inline std::pair<std::string, int> el_g(const std::string& g, int p) { return std::pair<std::string, int>(g, p); }

int objectGroup(const std::string& objectName) {
    static const std::map<std::string, int> objectGroups =
        { el_g("Al", 3), el_g("Ga", 3), el_g("In", 3),
          el_g("N", 5), el_g("P", 5), el_g("As", 5), el_g("Sb", 5), el_g("Bi", 5) };
    return map_find(objectGroups, objectName, 0);
}

Material::StringBuilder& Material::StringBuilder::operator()(const std::string& objectName, double ammount) {
    str << objectName;
    str << '(';
    str << ammount;
    str << ')';
    return *this;
}

std::string Material::StringBuilder::dopant(const std::string& dopantName, double dopantConcentration) {
    str << ':';
    str << dopantName;
    str << '=';
    str << dopantConcentration;
    return str.str();
}

std::string Material::StringBuilder::dopant(const std::string& dopantName, char n_or_p, double carrierConcentration) {
    str << ':';
    str << dopantName;
    str << ' ';
    str << n_or_p;
    str << '=';
    str << carrierConcentration;
    return str.str();
}

std::string Material::str() const {
    return name();
}

bool Material::isSimple() const {
    return MaterialsDB::isSimpleMaterialName(str());
}

double Material::A(double T) const { throwNotImplemented("A(double T)"); return 0; }

double Material::absp(double wl, double T) const { return 0.; }

double Material::B(double T) const { throwNotImplemented("B(double T)"); return 0; }

double Material::C(double T) const { throwNotImplemented("C(double T)"); return 0; }

double Material::CB(double T, double e, char point) const {
    if (e == 0.)
        return VB(T, 0., point) + Eg(T, 0., point);
    return max(VB(T, e, point, 'H'), VB(T, e, point, 'L')) + Eg(T, e, point);
}

double Material::chi(double T, double e, char point) const { throwNotImplemented("chi(double T, double e, char point)"); return 0; }

Tensor2<double> Material::cond(double T) const { throwNotImplemented("cond(double T)"); return 0.; }

Material::ConductivityType Material::condtype() const { return CONDUCTIVITY_UNDETERMINED; }

double Material::D(double T) const {
    // Use Einstein relation here
    double mu;
    try { mu = mob(T).c00; }
    catch(plask::NotImplemented) { throwNotImplemented("D(double T)"); }
    return mu * T * 8.6173423e-5;  // D = µ kB T / e
}

double Material::dens(double T) const { throwNotImplemented("dens(double T)"); return 0; }

double Material::Dso(double T, double e) const { throwNotImplemented("Dso(double T, double e)"); return 0; }

double Material::EactA(double T) const { throwNotImplemented("EactA(double T)"); return 0; }
double Material::EactD(double T) const { throwNotImplemented("EactD(double T)"); return 0; }

double Material::Eg(double T, double e, char point) const { throwNotImplemented("Eg(double T, double e, char point)"); return 0; }

double Material::eps(double T) const { throwNotImplemented("eps(double T)"); return 0; }

double Material::lattC(double T, char x) const { throwNotImplemented("lattC(double T, char x)"); return 0; }

Tensor2<double> Material::Me(double T, double e, char point) const { throwNotImplemented("Me(double T, double e, char point)"); return 0.; }
Tensor2<double> Material::Mh(double T, double e) const { throwNotImplemented("Mh(double T, double e)"); return 0.; }
Tensor2<double> Material::Mhh(double T, double e) const { throwNotImplemented("Mhh(double T, double e)"); return 0.; }
Tensor2<double> Material::Mlh(double T, double e) const { throwNotImplemented("Mlh(double T, double e)"); return 0.; }

double Material::ac(double T) const { throwNotImplemented("ac(double T)"); return 0; }
double Material::av(double T) const { throwNotImplemented("av(double T)"); return 0; }
double Material::b(double T) const { throwNotImplemented("b(double T)"); return 0; }
double Material::d(double T) const { throwNotImplemented("d(double T)"); return 0; }
double Material::c11(double T) const { throwNotImplemented("c11(double T)"); return 0; }
double Material::c12(double T) const { throwNotImplemented("c12(double T)"); return 0; }
double Material::c44(double T) const { throwNotImplemented("c44(double T)"); return 0; }

Tensor2<double> Material::mob(double T) const { throwNotImplemented("mob(double T)"); return 0.; }

double Material::Mso(double T, double e) const { throwNotImplemented("Mso(double T, double e)"); return 0; }

double Material::Nc(double T, double e, char point) const { throwNotImplemented("Nc(double T, double e, char point)"); return 0; }

double Material::Nv(double T, double e, char point) const { throwNotImplemented("Nv(double T, double e, char point)"); return 0; }

double Material::Nf(double T) const { throwNotImplemented("Nf(double T)"); return 0; }

double Material::Ni(double T) const { throwNotImplemented("Ni(double T)"); return 0; }

double Material::nr(double wl, double T, double n) const { throwNotImplemented("nr(double wl, double T, double n)"); return 0; }

dcomplex Material::Nr(double wl, double T, double n) const { return dcomplex(nr(wl,T), -7.95774715459e-09*absp(wl,T)*wl); }

Tensor3<dcomplex> Material::NR(double wl, double T, double n) const {
    return Nr(wl, T);
}

bool Material::operator ==(const Material &other) const {
    return typeid(*this) == typeid(other) && this->isEqual(other);
}

double Material::cp(double T) const { throwNotImplemented("cp(double T)"); return 0; }

Tensor2<double> Material::thermk(double T, double h) const { throwNotImplemented("thermk(double T)"); return 0.; }

double Material::VB(double T, double e, char point, char hole) const { throwNotImplemented("VB(double T, double e, char point, char hole)"); return 0; }

void Material::throwNotImplemented(const std::string& method_name) const {
    throw MaterialMethodNotImplemented(name(), method_name);
}

void Material::throwNotApplicable(const std::string& method_name) const {
    throw MaterialMethodNotApplicable(name(), method_name);
}

template <typename NameValuePairIter>
inline void fillGroupMaterialCompositionAmounts(NameValuePairIter begin, NameValuePairIter end, int group_nr) {
    static const char* const ROMANS[] = { "I", "II", "III", "IV", "V", "VI", "VII" };
    assert(0 < group_nr && group_nr < 8);
    auto no_info = end;
    double sum = 0.0;
    unsigned n = 0;
    for (auto i = begin; i != end; ++i) {
        if (std::isnan(i->second)) {
            if (no_info != end)
                throw plask::MaterialParseException("Incomplete material composition for group %1% elements", ROMANS[group_nr-1]);
            else
                no_info = i;
        } else {
            sum += i->second;
            ++n;
        }
    }
    if (n > 0 && sum - 1.0 > SMALL*n)
        throw plask::MaterialParseException("Total material composition for group %1% elements exceeds 1", ROMANS[group_nr-1]);
    if (no_info != end) {
        no_info->second = 1.0 - sum;
    } else {
        if (!is_zero(sum - 1.0))
             throw plask::MaterialParseException("Total material composition for group %1% elements (%2%) differs from 1", ROMANS[group_nr-1], sum);
    }
}

Material::Composition Material::completeComposition(const Composition &composition) {
    std::map<int, std::vector< std::pair<std::string, double> > > by_group;
    for (auto c: composition) {
        int group = objectGroup(c.first);
        if (group == 0) throw plask::MaterialParseException("Wrong object name \"%1%\"", c.first);
        by_group[group].push_back(c);
    }
    Material::Composition result;
    for (auto g: by_group) {
        fillGroupMaterialCompositionAmounts(g.second.begin(), g.second.end(), g.first);
        result.insert(g.second.begin(), g.second.end());
    }
    return result;
}

const char* getObjectEnd(const char* begin, const char* end) {
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

std::pair<std::string, double> Material::firstCompositionObject(const char*& begin, const char* end) {
    std::pair<std::string, double> result;
    const char* comp_end = getObjectEnd(begin, end);
    if (comp_end == begin)
        throw MaterialParseException(std::string("Expected element but found character: ") + *begin);
    result.first = std::string(begin, comp_end);
    const char* amount_end = getAmountEnd(comp_end, end);
    if (amount_end == comp_end) {       //no amount info for this object
        result.second = std::numeric_limits<double>::quiet_NaN();
        begin = amount_end;
    } else {
        if (amount_end == end)
            throw MaterialParseException("Unexpected end of input while reading element amount. Couldn't find ')'");
        result.second = toDouble(std::string(comp_end+1, amount_end));
        begin = amount_end+1;   //skip also ')', begin now points to 1 character after ')'
    }
    return result;
}


Material::Composition Material::parseComposition(const char* begin, const char* end) {
    const char* fullname = begin;   // for exceptions only
    Material::Composition result;
    std::set<int> groups;
    int prev_g = -1;
    while (begin != end) {
        auto c = firstCompositionObject(begin, end);
        int g = objectGroup(c.first);
        if (g != prev_g) {
            if (!groups.insert(g).second)
                throw MaterialParseException("Incorrect elements order in \"%1%\"", fullname);
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

void Material::parseDopant(const char* begin, const char* end, std::string& dopant_elem_name, Material::DopingAmountType& doping_amount_type, double& doping_amount) {
    const char* name_end = getObjectEnd(begin, end);
    if (name_end == begin)
         throw MaterialParseException("No dopant name");
    dopant_elem_name.assign(begin, name_end);
    if (*name_end == '=') {
        if (name_end+1 == end) throw MaterialParseException("Unexpected end of input while reading doping concentation");
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
    //TODO check p.first if is p/n compatibile with dopant_elem_name
    doping_amount_type = Material::CARRIER_CONCENTRATION;
    doping_amount = toDouble(p.second);
}

void Material::parseDopant(const std::string &dopant, std::string &dopant_elem_name, Material::DopingAmountType &doping_amount_type, double &doping_amount) {
    const char* c = dopant.data();
    parseDopant(c, c + dopant.size(), dopant_elem_name, doping_amount_type, doping_amount);
}

std::vector<std::string> Material::parseObjectsNames(const char *begin, const char *end) {
    const char* full_name = begin;  //store for error msg. only
    std::vector<std::string> elemenNames;
    do {
        const char* new_begin = getObjectEnd(begin, end);
        if (new_begin == begin) throw MaterialParseException("Ill-formated name \"%1%\"", std::string(full_name, end));
        elemenNames.push_back(std::string(begin, new_begin));
        begin = new_begin;
    } while (begin != end);
    return elemenNames;
}

std::vector<std::string> Material::parseObjectsNames(const std::string &allNames) {
    const char* c = allNames.c_str();
    return parseObjectsNames(c, c + allNames.size());
}

std::string Material::dopantName() const {
    std::string::size_type p = this->name().rfind(':');
    return p == std::string::npos ? "" : this->name().substr(p+1);
}

std::string Material::nameWithoutDopant() const {
    return this->name().substr(0, this->name().rfind(':'));
}

//------------ Different material kinds -------------------------

std::string Semiconductor::name() const { return NAME; }
Material::Kind Semiconductor::kind() const { return Material::SEMICONDUCTOR; }
static MaterialsDB::Register<Semiconductor> materialDB_register_Semiconductor;

std::string Metal::name() const { return NAME; }
Material::Kind Metal::kind() const { return Material::METAL; }
static MaterialsDB::Register<Metal> materialDB_register_Metal;

std::string Oxide::name() const { return NAME; }
Material::Kind Oxide::kind() const { return Material::OXIDE; }
static MaterialsDB::Register<Metal> materialDB_register_Oxide;

std::string Dielectric::name() const { return NAME; }
Material::Kind Dielectric::kind() const { return Material::DIELECTRIC; }
static MaterialsDB::Register<Dielectric> materialDB_register_Dielectric;

std::string LiquidCrystal::name() const { return NAME; }
Material::Kind LiquidCrystal::kind() const { return Material::LIQUID_CRYSTAL; }
static MaterialsDB::Register<LiquidCrystal> materialDB_register_LiquidCrystal;

//------------ Metals -------------------------

double Metal::eps(double T) const {
    return 1.;
}

//------------ MixedMaterial -------------------------

Material::Kind MixedMaterial::kind() const { return Material::MIXED; }

void MixedMaterial::normalizeWeights() {
    double sum = 0;
    for (auto& p: materials) sum += std::get<1>(p);
    for (auto& p: materials) std::get<1>(p) /= sum;
}

MixedMaterial & MixedMaterial::add(const shared_ptr<plask::Material> &material, double weight) {
    materials.push_back(std::pair<shared_ptr<Material>,double>(material, weight));
    return *this;
}

std::string MixedMaterial::name() const {
    std::string result = "[";
    for (auto i = materials.begin(); i != materials.end(); ++i) {
        if (i != materials.begin()) result += '+';
        result += boost::lexical_cast<std::string>(std::get<1>(*i));
        result += '*';
        result += std::get<0>(*i)->name();
    }
    result += ']';
    return result;
}

double MixedMaterial::A(double T) const {
    return avg([&](const Material& m) { return m.A(T); });
}

double MixedMaterial::absp(double wl, double T) const {
    return avg([&](const Material& m) { return m.absp(wl, T); });
}

double MixedMaterial::B(double T) const {
    return avg([&](const Material& m) { return m.B(T); });
}

double MixedMaterial::C(double T) const {
    return avg([&](const Material& m) { return m.C(T); });
}

double MixedMaterial::CB(double T, double e, char point) const {
    return avg([&](const Material& m) { return m.CB(T, e, point); });
}

double MixedMaterial::chi(double T, double e, char point) const {
    return avg([&](const Material& m) { return m.chi(T, e, point); });
}

Tensor2<double> MixedMaterial::cond(double T) const {
    return avg_pairs([&](const Material& m) { return m.cond(T); });
}

Material::ConductivityType MixedMaterial::condtype() const {
    if (materials.size() == 0) return CONDUCTIVITY_UNDETERMINED;
    Material::ConductivityType result = materials[0].first->condtype();
    for(auto mat = materials.begin()+1; mat != materials.end(); ++mat)
        if (mat->first->condtype() != result) return CONDUCTIVITY_UNDETERMINED;
    return result;
}

double MixedMaterial::D(double T) const {
    return avg([&](const Material& m) { return m.D(T); });
}

double MixedMaterial::dens(double T) const {
    return avg([&](const Material& m) { return m.dens(T); });
}

double MixedMaterial::Dso(double T, double e) const {
    return avg([&](const Material& m) { return m.Dso(T); });
}

double MixedMaterial::EactA(double T) const {
    return avg([&](const Material& m) { return m.EactA(T); });
}
double MixedMaterial::EactD(double T) const {
    return avg([&](const Material& m) { return m.EactD(T); });
}

double MixedMaterial::Eg(double T, double e, char point) const {
    return avg([&](const Material& m) { return m.Eg(T, e, point); });
}

double MixedMaterial::eps(double T) const {
    return avg([&](const Material& m) { return m.eps(T); });
}

double MixedMaterial::lattC(double T, char x) const {
    return avg([&](const Material& m) { return m.lattC(T, x); });
}

Tensor2<double> MixedMaterial::Me(double T, double e, char point) const {
    return avg_pairs([&](const Material& m) { return m.Me(T, e, point); });
}

Tensor2<double> MixedMaterial::Mh(double T, double e) const {
    return avg_pairs([&](const Material& m) { return m.Mh(T, e); });
}

Tensor2<double> MixedMaterial::Mhh(double T, double e) const {
    return avg_pairs([&](const Material& m) { return m.Mhh(T, e); });
}

Tensor2<double> MixedMaterial::Mlh(double T, double e) const  {
    return avg_pairs([&](const Material& m) { return m.Mlh(T, e); });
}

double MixedMaterial::ac(double T) const {
    return avg([&](const Material& m) { return m.ac(T); });
}

double MixedMaterial::av(double T) const {
    return avg([&](const Material& m) { return m.av(T); });
}

double MixedMaterial::b(double T) const {
    return avg([&](const Material& m) { return m.b(T); });
}

double MixedMaterial::d(double T) const {
    return avg([&](const Material& m) { return m.d(T); });
}

double MixedMaterial::c11(double T) const {
    return avg([&](const Material& m) { return m.c11(T); });
}

double MixedMaterial::c12(double T) const {
    return avg([&](const Material& m) { return m.c12(T); });
}

double MixedMaterial::c44(double T) const {
    return avg([&](const Material& m) { return m.c44(T); });
}

Tensor2<double> MixedMaterial::mob(double T) const {
    return avg_pairs([&](const Material& m) { return m.mob(T); });
}

double MixedMaterial::Mso(double T, double e) const {
    return avg([&](const Material& m) { return m.Mso(T); });
}

double MixedMaterial::Nc(double T, double e, char point) const {
    return avg([&](const Material& m) { return m.Nc(T, e, point); });
}

double MixedMaterial::Nv(double T, double e, char point) const {
    return avg([&](const Material& m) { return m.Nv(T, e, point); });
}

double MixedMaterial::Nf(double T) const {
    return avg([&](const Material& m) { return m.Nf(T); });
}

double MixedMaterial::Ni(double T) const {
    return avg([&](const Material& m) { return m.Ni(T); });
}

double MixedMaterial::nr(double wl, double T, double n) const {
    return avg([&](const Material& m) { return m.nr(wl, T, n); });
}

dcomplex MixedMaterial::Nr(double wl, double T, double n) const {
    return avg([&](const Material& m) { return m.Nr(wl, T, n); });
}

Tensor3<dcomplex> MixedMaterial::NR(double wl, double T, double n) const {
    Tensor3<dcomplex> result;
    result.c00 = avg([&](const Material& m) { return m.NR(wl, T, n).c00; });
    result.c11 = avg([&](const Material& m) { return m.NR(wl, T, n).c11; });
    result.c22 = avg([&](const Material& m) { return m.NR(wl, T, n).c22; });
    result.c01 = avg([&](const Material& m) { return m.NR(wl, T, n).c01; });
    result.c11 = avg([&](const Material& m) { return m.NR(wl, T, n).c11; });
    return result;
}

double MixedMaterial::cp(double T) const {
    return avg([&](const Material& m) { return m.cp(T); });
}

Tensor2<double> MixedMaterial::thermk(double T, double h) const {
    return avg_pairs([&](const Material& m) { return m.thermk(T, h); });
}

double MixedMaterial::VB(double T, double e, char point, char hole) const  {
    return avg([&](const Material& m) { return m.VB(T, e, point); });
}

bool Material::isEqual(const Material &other) const {
    return this->str() == other.str();
}

}   // namespace plask

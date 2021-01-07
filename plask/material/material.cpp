#include "material.hpp"

#include <boost/lexical_cast.hpp>
#include "../utils/stl.hpp"
#include "../utils/string.hpp"
#include "../log/log.hpp"
#include "db.hpp"

#include <cmath>
#include <set>

namespace plask {

// inline std::pair<std::string, int> el_g(const std::string& g, int p) { return std::pair<std::string, int>(g, p); }

int objectGroup(const std::string& objectName) {
    static const std::map<std::string, int> objectGroups =
        { {"Be", 2}, {"Mg", 2}, {"Ca", 2}, {"Sr", 2}, {"Ba", 2},
          {"B", 3}, {"Al", 3}, {"Ga", 3}, {"In", 3}, {"Tl", 3},
          {"C", 4}, {"Si", 4}, {"Ge", 4}, {"Sn", 4}, {"Pb", 4},
          {"N", 5}, {"P", 5}, {"As", 5}, {"Sb", 5}, {"Bi", 5},
          {"O", 6}, {"S", 6}, {"Se", 6}, {"Te", 6}
        };
    return map_find(objectGroups, objectName, 0);
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
                throw plask::MaterialParseException("Incomplete material composition for group {0} elements", ROMANS[group_nr-1]);
            else
                no_info = i;
        } else {
            sum += i->second;
            ++n;
        }
    }
    if (n > 0 && sum - 1.0 > SMALL*n)
        throw plask::MaterialParseException("Total material composition for group {0} elements exceeds 1", ROMANS[group_nr-1]);
    if (no_info != end) {
        no_info->second = 1.0 - sum;
    } else {
        if (!is_zero(sum - 1.0))
             throw plask::MaterialParseException("Total material composition for group {0} elements ({1}) differs from 1", ROMANS[group_nr-1], sum);
    }
}


Material::StringBuilder& Material::StringBuilder::operator()(const std::string& objectName, double ammount) {
    str << objectName;
    str << '(';
    str << ammount;
    str << ')';
    return *this;
}

std::string Material::StringBuilder::dopant(const std::string& dopant, double dopantConcentration) {
    str << ':';
    str << dopant;
    str << '=';
    str << dopantConcentration;
    return str.str();
}

void Material::Parameters::parse(const std::string &full_material_str, bool allow_dopant_without_amount) {
    std::string dope;
    std::tie(name, dope) = splitString2(full_material_str, ':');
    std::tie(name, label) = splitString2(name, '_');
    if (!dope.empty())
        Material::parseDopant(dope, dopant, doping, allow_dopant_without_amount);
    else
        this->clearDoping();
    if (isSimpleMaterialName(name))
        composition.clear();
    else
        composition = Material::parseComposition(name);
}

std::string Material::Parameters::str() const {
    std::string result;
    if (isAlloy()) {
        std::map<int, std::vector<std::pair<std::string, double>>> by_group;
        for (auto c: composition) {
            int group = objectGroup(c.first);
            if (group == 0) throw plask::MaterialParseException("Wrong object name \"{0}\"", c.first);
            by_group[group].push_back(c);
        }
        for (auto g: by_group) {
            fillGroupMaterialCompositionAmounts(g.second.begin(), g.second.end(), g.first);
            if (g.second.begin() != g.second.end()) {
                auto last = g.second.end() - 1;
                for (auto el = g.second.begin(); el != last; ++el)
                    result += format("{}({})", el->first, el->second);
                result += last->first;
            }
        }
    } else {
        result = name;
    }
    if (label != "") result += "_" + label;
    if (dopant != "") result += ":" + dopant + "=" + plask::str(doping);
    return result;
}

Material::Composition Material::Parameters::completeComposition() const {
    return Material::completeComposition(composition);
}

void Material::Parameters::setDoping(const std::string& dopant, double doping) {
    this->dopant = dopant;
    this->doping = doping;
}

std::string Material::str() const {
    return name();
}

bool Material::isAlloy() const {
    return !isSimpleMaterialName(str());
}

Material::Composition Material::composition() const {
    return Composition();
}

double Material::doping() const {
    return 0;
}

double Material::A(double /*T*/) const { throwNotImplemented("A(double T)"); }

double Material::absp(double /*lam*/, double /*T*/) const { return 0.; }

double Material::B(double /*T*/) const { throwNotImplemented("B(double T)"); }

double Material::C(double /*T*/) const { throwNotImplemented("C(double T)"); }

double Material::CB(double T, double e, char point) const {
    if (e == 0.)
        return VB(T, 0., point) + Eg(T, 0., point);
    else
        return max(VB(T, e, point, 'H'), VB(T, e, point, 'L')) + Eg(T, e, point);
}

double Material::chi(double /*T*/, double /*e*/, char /*point*/) const { throwNotImplemented("chi(double T, double e, char point)"); }

Tensor2<double> Material::cond(double /*T*/) const { throwNotImplemented("cond(double T)"); }

Material::ConductivityType Material::condtype() const { return CONDUCTIVITY_UNDETERMINED; }

double Material::D(double T) const {
    // Use Einstein relation here
    double mu;
    try { mu = mob(T).c00; }
    catch(plask::NotImplemented&) { throwNotImplemented("D(double T)"); }
    return mu * T * 8.6173423e-5;  // D = µ kB T / e
}

double Material::dens(double /*T*/) const { throwNotImplemented("dens(double T)"); }

double Material::Dso(double /*T*/, double /*e*/) const { throwNotImplemented("Dso(double T, double e)"); }

double Material::EactA(double /*T*/) const { throwNotImplemented("EactA(double T)"); }
double Material::EactD(double /*T*/) const { throwNotImplemented("EactD(double T)"); }

double Material::Eg(double /*T*/, double /*e*/, char /*point*/) const { throwNotImplemented("Eg(double T, double e, char point)"); }

double Material::eps(double /*T*/) const { throwNotImplemented("eps(double T)"); }

double Material::lattC(double /*T*/, char /*x*/) const { throwNotImplemented("lattC(double T, char x)"); }

Tensor2<double> Material::Me(double /*T*/, double /*e*/, char /*point*/) const { throwNotImplemented("Me(double T, double e, char point)"); }
Tensor2<double> Material::Mh(double /*T*/, double /*e*/) const { throwNotImplemented("Mh(double T, double e)"); }
Tensor2<double> Material::Mhh(double /*T*/, double /*e*/) const { throwNotImplemented("Mhh(double T, double e)"); }
Tensor2<double> Material::Mlh(double /*T*/, double /*e*/) const { throwNotImplemented("Mlh(double T, double e)"); }

double Material::y1() const { throwNotImplemented("y1()"); }
double Material::y2() const { throwNotImplemented("y2()"); }
double Material::y3() const { throwNotImplemented("y3()"); }

double Material::ac(double /*T*/) const { throwNotImplemented("ac(double T)"); }
double Material::av(double /*T*/) const { throwNotImplemented("av(double T)"); }
double Material::b(double /*T*/) const { throwNotImplemented("b(double T)"); }
double Material::d(double /*T*/) const { throwNotImplemented("d(double T)"); }
double Material::c11(double /*T*/) const { throwNotImplemented("c11(double T)"); }
double Material::c12(double /*T*/) const { throwNotImplemented("c12(double T)"); }
double Material::c44(double /*T*/) const { throwNotImplemented("c44(double T)"); }

Tensor2<double> Material::mob(double /*T*/) const { throwNotImplemented("mob(double T)"); }

double Material::Mso(double /*T*/, double /*e*/) const { throwNotImplemented("Mso(double T, double e)"); }

double Material::Nf(double /*T*/) const { throwNotImplemented("Nf(double T)"); }

double Material::Ni(double /*T*/) const { throwNotImplemented("Ni(double T)"); }

double Material::nr(double /*lam*/, double /*T*/, double /*n*/) const { throwNotImplemented("nr(double lam, double T, double n)"); }

dcomplex Material::Nr(double lam, double T, double n) const { return dcomplex(nr(lam,T, n), -7.95774715459e-09*absp(lam,T)*lam); }

Tensor3<dcomplex> Material::NR(double lam, double T, double n) const {
    return Nr(lam, T, n);
}

bool Material::operator ==(const Material &other) const {
    return typeid(*this) == typeid(other) && this->isEqual(other);
}

double Material::cp(double /*T*/) const { throwNotImplemented("cp(double T)"); }

Tensor2<double> Material::thermk(double /*T*/, double /*h*/) const { throwNotImplemented("thermk(double T, double h)"); }

double Material::VB(double /*T*/, double /*e*/, char /*point*/, char /*hole*/) const { throwNotImplemented("VB(double T, double e, char point, char hole)"); }


Tensor2<double> Material::mobe(double /*T*/) const { throwNotImplemented("mobe(double T)"); }

Tensor2<double> Material::mobh(double /*T*/) const { throwNotImplemented("mobh(double T)"); }

double Material::taue(double /*T*/) const { throwNotImplemented("taue(double T)"); }

double Material::tauh(double /*T*/) const { throwNotImplemented("tauh(double T)"); }

double Material::Ce(double /*T*/) const { throwNotImplemented("Ce(double T)"); }

double Material::Ch(double /*T*/) const { throwNotImplemented("Ch(double T)"); }

double Material::e13(double /*T*/) const { throwNotImplemented("e13(double T)"); }

double Material::e15(double /*T*/) const { throwNotImplemented("e15(double T)"); }

double Material::e33(double /*T*/) const { throwNotImplemented("e33(double T)"); }

double Material::c13(double /*T*/) const { throwNotImplemented("c13(double T)"); }

double Material::c33(double /*T*/) const { throwNotImplemented("c33(double T)"); }

double Material::Psp(double /*T*/) const { throwNotImplemented("Psp(double T)"); }

double Material::Na() const { throwNotImplemented("Na()"); }

double Material::Nd() const { throwNotImplemented("Nd()"); }






[[noreturn]] void Material::throwNotImplemented(const std::string& method_name) const {
    throw MaterialMethodNotImplemented(name(), method_name);
}

Material::Composition Material::completeComposition(const Composition &composition) {
    std::map<int, std::vector<std::pair<std::string, double>>> by_group;
    for (auto c: composition) {
        int group = objectGroup(c.first);
        if (group == 0) throw plask::MaterialParseException("Wrong object name \"{0}\"", c.first);
        by_group[group].push_back(c);
    }
    Material::Composition result;
    for (auto g: by_group) {
        fillGroupMaterialCompositionAmounts(g.second.begin(), g.second.end(), g.first);
        result.insert(g.second.begin(), g.second.end());
    }
    return result;
}

Material::Composition Material::minimalComposition(const Composition &composition) {
    std::map<int, std::vector<std::pair<std::string, double>>> by_group;
    for (auto c: composition) {
        int group = objectGroup(c.first);
        if (group == 0) throw plask::MaterialParseException("Wrong object name \"{0}\"", c.first);
        by_group[group].push_back(c);
    }
    Material::Composition result;
    for (auto g: by_group) {
        fillGroupMaterialCompositionAmounts(g.second.begin(), g.second.end(), g.first);
        if (g.second.begin() != g.second.end()) (g.second.end()-1)->second = NAN;
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
                throw MaterialParseException("Incorrect elements order in \"{0}\"", fullname);
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

void Material::parseDopant(const char* begin, const char* end, std::string& dopant_elem_name, double& doping, bool allow_dopant_without_amount) {
    const char* name_end = getObjectEnd(begin, end);
    if (name_end == begin)
         throw MaterialParseException("No dopant name");
    dopant_elem_name.assign(begin, name_end);
    if (name_end == end) {
        if (!allow_dopant_without_amount)
            throw MaterialParseException("Unexpected end of input while reading doping concentration");
        // there might be some reason to specify material with dopant but undoped (can be caught in material constructor)
        doping = NAN;
        return;
    }
    if (*name_end == '=') {
        if (name_end+1 == end) throw MaterialParseException("Unexpected end of input while reading doping concentration");
        doping = toDouble(std::string(name_end+1, end));
        return;
    }
    throw MaterialParseException("Expected '=' but found '{0}' instead", *name_end);
}

void Material::parseDopant(const std::string &dopant, std::string &dopant_elem_name, double &doping, bool allow_dopant_without_amount) {
    const char* c = dopant.data();
    parseDopant(c, c + dopant.size(), dopant_elem_name, doping, allow_dopant_without_amount);
}

std::vector<std::string> Material::parseObjectsNames(const char *begin, const char *end) {
    const char* full_name = begin;  //store for error msg. only
    std::vector<std::string> elemenNames;
    do {
        const char* new_begin = getObjectEnd(begin, end);
        if (new_begin == begin) throw MaterialParseException("Ill-formatted name \"{0}\"", std::string(full_name, end));
        elemenNames.push_back(std::string(begin, new_begin));
        begin = new_begin;
    } while (begin != end);
    return elemenNames;
}

std::vector<std::string> Material::parseObjectsNames(const std::string &allNames) {
    const char* c = allNames.c_str();
    return parseObjectsNames(c, c + allNames.size());
}

std::string Material::dopant() const {
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

double Metal::eps(double /*T*/) const {
    return 1.;
}

}   // namespace plask

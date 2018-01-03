#include "mixed.h"

namespace plask {

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

double MixedMaterial::absp(double lam, double T) const {
    return avg([&](const Material& m) { return m.absp(lam, T); });
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

double MixedMaterial::Nf(double T) const {
    return avg([&](const Material& m) { return m.Nf(T); });
}

double MixedMaterial::Ni(double T) const {
    return avg([&](const Material& m) { return m.Ni(T); });
}

double MixedMaterial::nr(double lam, double T, double n) const {
    return avg([&](const Material& m) { return m.nr(lam, T, n); });
}

dcomplex MixedMaterial::Nr(double lam, double T, double n) const {
    return avg([&](const Material& m) { return m.Nr(lam, T, n); });
}

Tensor3<dcomplex> MixedMaterial::NR(double lam, double T, double n) const {
    Tensor3<dcomplex> result;
    result.c00 = avg([&](const Material& m) { return m.NR(lam, T, n).c00; });
    result.c11 = avg([&](const Material& m) { return m.NR(lam, T, n).c11; });
    result.c22 = avg([&](const Material& m) { return m.NR(lam, T, n).c22; });
    result.c01 = avg([&](const Material& m) { return m.NR(lam, T, n).c01; });
    result.c11 = avg([&](const Material& m) { return m.NR(lam, T, n).c11; });
    return result;
}

double MixedMaterial::cp(double T) const {
    return avg([&](const Material& m) { return m.cp(T); });
}

Tensor2<double> MixedMaterial::thermk(double T, double h) const {
    return avg_pairs([&](const Material& m) { return m.thermk(T, h); });
}

double MixedMaterial::VB(double T, double e, char point, char hole) const  {
    return avg([&](const Material& m) { return m.VB(T, e, point, hole); });
}


Tensor2<double> MixedMaterial::mobe(double T) const {
    return avg_pairs([&](const Material& m) { return m.mobe(T); });
}

Tensor2<double> MixedMaterial::mobh(double T) const {
    return avg_pairs([&](const Material& m) { return m.mobh(T); });
}

double MixedMaterial::taue(double T) const {
    return avg([&](const Material& m) { return m.taue(T); });
}

double MixedMaterial::tauh(double T) const {
    return avg([&](const Material& m) { return m.tauh(T); });
}

double MixedMaterial::Ce(double T) const {
    return avg([&](const Material& m) { return m.Ce(T); });
}

double MixedMaterial::Ch(double T) const {
    return avg([&](const Material& m) { return m.Ch(T); });
}

double MixedMaterial::e13(double T) const {
    return avg([&](const Material& m) { return m.e13(T); });
}

double MixedMaterial::e15(double T) const {
    return avg([&](const Material& m) { return m.e15(T); });
}

double MixedMaterial::e33(double T) const {
    return avg([&](const Material& m) { return m.e33(T); });
}

double MixedMaterial::c13(double T) const {
    return avg([&](const Material& m) { return m.c13(T); });
}

double MixedMaterial::c33(double T) const {
    return avg([&](const Material& m) { return m.c33(T); });
}

double MixedMaterial::Psp(double T) const {
    return avg([&](const Material& m) { return m.Psp(T); });
}

double MixedMaterial::Na() const {
    return avg([&](const Material& m) { return m.Na(); });
}

double MixedMaterial::Nd() const {
    return avg([&](const Material& m) { return m.Nd(); });
}


bool Material::isEqual(const Material &other) const {
    return this->str() == other.str();
}

} // namespace plask

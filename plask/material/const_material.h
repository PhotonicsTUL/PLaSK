#ifndef PLASK__CONST_MATERIAL_H
#define PLASK__CONST_MATERIAL_H

/**
 * \file
 * Here is a definition of a material with constant parameters specified in its name
 */

#include <map>
#include "material.h"

namespace plask {

class PLASK_API ConstMaterial: public MaterialWithBase {

    MaterialCache cache;

public:

    ConstMaterial(const std::string& full_name);

    ConstMaterial(const shared_ptr<Material>& base, const std::map<std::string, double>& items);

    bool isEqual(const Material& other) const override {
        const ConstMaterial& cother = static_cast<const ConstMaterial&>(other);
        return 
            ((!base && !cother.base) || (base && cother.base && *base == *cother.base)) &&
            (cache == cother.cache);
    }

    std::string name() const override {
        if (base) return base->name();
        else return "";
    }

    Material::Kind kind() const override {
        if (base) return base->kind();
        else return Material::GENERIC; 
    }

    Material::ConductivityType condtype() const override {
        if (base) return base->condtype();
        else return Material::CONDUCTIVITY_UNDETERMINED;
    }

    std::string str() const override;

    double lattC(double T, char x) const override;
    double Eg(double T, double e=0., char point='*') const override;
    double CB(double T, double e=0., char point='*') const override;
    double VB(double T, double e=0., char point='*', char hole='H') const override;
    double Dso(double T, double e=0.) const override;
    double Mso(double T, double e=0.) const override;
    Tensor2<double> Me(double T, double e=0., char point='*') const override;
    Tensor2<double> Mhh(double T, double e=0.) const override;
    Tensor2<double> Mlh(double T, double e=0.) const override;
    Tensor2<double> Mh(double T, double e=0.) const override;
    double y1() const override;
    double y2() const override;
    double y3() const override;
    double ac(double T) const override;
    double av(double T) const override;
    double b(double T) const override;
    double d(double T) const override;
    double c11(double T) const override;
    double c12(double T) const override;
    double c44(double T) const override;
    double eps(double T) const override;
    double chi(double T, double e=0., char point='*') const override;
    double Ni(double T) const override;
    double Nf(double T) const override;
    double EactD(double T) const override;
    double EactA(double T) const override;
    Tensor2<double> mob(double T) const override;
    Tensor2<double> cond(double T) const override;
    double A(double T) const override;
    double B(double T) const override;
    double C(double T) const override;
    double D(double T) const override;
    Tensor2<double> thermk(double T, double h=INFINITY) const override;
    double dens(double T) const override;
    double cp(double T) const override;
    double nr(double lam, double T, double n = 0) const override;
    double absp(double lam, double T) const override;
    dcomplex Nr(double lam, double T, double n = 0) const override;
    Tensor3<dcomplex> NR(double lam, double T, double n = 0) const override;
    Tensor2<double> mobe(double T) const override;
    Tensor2<double> mobh(double T) const override;
    double taue(double T) const override;
    double tauh(double T) const override;
    double Ce(double T) const override;
    double Ch(double T) const override;
    double e13(double T) const override;
    double e15(double T) const override;
    double e33(double T) const override;
    double c13(double T) const override;
    double c33(double T) const override;
    double Psp(double T) const override;
    double Na() const override;
    double Nd() const override;
};

}   // namespace plask

#endif // PLASK__CONST_MATERIAL_H
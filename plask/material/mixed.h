#ifndef PLASK__MATERIAL_SPECIAL_H
#define PLASK__MATERIAL_SPECIAL_H

/** @file
This file contains classes for some special materials.
*/

#include "material.h"

namespace plask {

/**
 * Material which consist of several real materials.
 * It calculate averages for all properties.
 *
 * Example:
 * @code
 * MixedMaterial m;
 * // mat1, mat2, mat3 are materials, 2.0, 5.0, 3.0 weights for it:
 * m.add(mat1, 2.0).add(mat2, 5.0).add(mat3, 3.0).normalizeWeights();
 * double avg_VB = m.VB(300);
 * @endcode
 */
struct PLASK_API MixedMaterial: public Material {

    /** Vector of materials and its weights. */
    std::vector < std::pair<shared_ptr<Material>,double> > materials;

    /**
      Delegate all constructor to materials vector.
      */
    template<typename ...Args>
    MixedMaterial(Args&&... params)
    : materials(std::forward<Args>(params)...) {
    }

    /**
     * Scale weights in materials vector, making sum of this weights equal to 1.
     */
    void normalizeWeights();

    /**
     * Add material with weight to materials vector.
     * @param material material to add
     * @param weight weight
     */
    MixedMaterial& add(const shared_ptr<Material>& material, double weight);

    virtual ~MixedMaterial() {}

    //Material methods implementation:
    std::string name() const override;

    Kind kind() const override;

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

    double ac(double T) const override;

    double av(double T) const override;

    double b(double T) const override;

    double d(double T) const override;

    double c11(double T) const override;

    double c12(double T) const override;

    double c44(double T) const override;

    double eps(double T) const override;

    double chi(double T, double e=0., char point='*') const override;

    double Ni(double T=0.) const override;

    double Nf(double T=0.) const override;

    double EactD(double T) const override;

    double EactA(double T) const override;

    Tensor2<double> mob(double T) const override;

    Tensor2<double> cond(double T) const override;

    ConductivityType condtype() const override;

    double A(double T) const override;

    double B(double T) const override;

    double C(double T) const override;

    double D(double T) const override;

    Tensor2<double> thermk(double T, double h) const override;

    double dens(double T) const override;

    double cp(double T) const override;

    double nr(double lam, double T, double n = 0.0) const override;

    double absp(double lam, double T) const override;

    dcomplex Nr(double lam, double T, double n = 0.0) const override;

    Tensor3<dcomplex> NR(double lam, double T, double n = 0.0) const override;

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

private:

    /**
     * Calulate weighted sums of materials (from materials vector) properties.
     * @param f functore which calculate property value for given material
     * @return calculated sum, with the same type which return functor
     * @tparam Functor type of functor which can take const Material& argument, and return something which can be multiple by scalar, added, and assigned
     */
    template <typename Functor>
    auto avg(Functor f) const -> typename std::remove_cv<decltype(f(*((const Material*)0)))>::type {
        typename std::remove_cv<decltype(f(*((const Material*)0)))>::type w_sum = 0.;
        for (auto& p: materials) {
            w_sum += std::get<1>(p) * f(*std::get<0>(p));
        }
        return w_sum;
    }

    /**
     * Calulate weighted sums of materials (from materials vector) properties.
     * @param f functore which calculate property value for given material
     * @return calculated sum, with the same type which return functor
     * @tparam Functor type of functor which can take const Material& argument, and return something which can be multiple by scalar, added, and assigned
     */
    template <typename Functor>
    auto avg_pairs(Functor f) const -> Tensor2<double> {
        Tensor2<double> w_sum(0., 0.);
        for (auto& p: materials) {
            Tensor2<double> m = f(*std::get<0>(p));
            w_sum.c00 += std::get<1>(p) * m.c00;    //std::get<1>(p) is weight of current material
            w_sum.c11 += std::get<1>(p) * m.c11;
        }
        return w_sum;
    }

};

} // namespace plask

#endif // PLASK__MATERIAL_SPECIAL_H

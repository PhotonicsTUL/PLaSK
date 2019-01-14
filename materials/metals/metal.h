#ifndef PLASK__METALS_METAL_H
#define PLASK__METALS_METAL_H

/** @\file
This file contains some common base for all metals
*/

#include <plask/material/material.h>

namespace plask { namespace materials {

struct LorentzDrudeMetal: public Metal {
  protected:

    static constexpr double eh = 2.*PI * phys::qe / phys::h_J;
    const std::vector<double> f, G, w;
    const double wp;

    dcomplex opt_eps(double lam) const;

    LorentzDrudeMetal(double wp,
                      std::initializer_list<double> f,
                      std::initializer_list<double> G,
                      std::initializer_list<double> w) :
        wp(wp), f(f), G(G), w(w) {
        assert(f.size() == G.size());
        assert(G.size() == w.size());
    }

  public:

    dcomplex Nr(double lam, double /*T*/, double n = 0) const override {
        return sqrt(opt_eps(lam));
    }

    double nr(double lam, double /*T*/, double n = 0) const override {
        //sqrt((sqrt(epsRe*epsRe + epsIm * epsIm) + epsRe) / 2.)
        return sqrt(opt_eps(lam)).real();
    }

    double absp(double lam, double /*T*/) const override {
        return - 4e7*M_PI / lam * sqrt(opt_eps(lam)).imag();
    }
};


struct BrendelBormannMetal: public Metal {
  protected:

    static constexpr double eh = 2.*PI * phys::qe / phys::h_J;

    const std::vector<double> f, G, w, s;
    const double wp;

    dcomplex opt_eps(double lam) const;

    BrendelBormannMetal(double wp,
                        std::initializer_list<double> f,
                        std::initializer_list<double> G,
                        std::initializer_list<double> w,
                        std::initializer_list<double> s) :
        wp(wp), f(f), G(G), w(w), s(s) {
        assert(f.size() == G.size());
        assert(G.size() == w.size());
        assert(w.size() == s.size());
    }

  public:

    dcomplex Nr(double lam, double /*T*/, double n = 0) const override {
        return sqrt(opt_eps(lam));
    }

    double nr(double lam, double /*T*/, double n = 0) const override {
        //sqrt((sqrt(epsRe*epsRe + epsIm * epsIm) + epsRe) / 2.)
        return sqrt(opt_eps(lam)).real();
    }

    double absp(double lam, double /*T*/) const override {
        return - 4e7*PI / lam * sqrt(opt_eps(lam)).imag();
    }
};


}} // namespace plask::materials

#endif //PLASK__METALS_METAL_H

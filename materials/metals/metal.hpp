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
#ifndef PLASK__METALS_METAL_H
#define PLASK__METALS_METAL_H

/** @\file
This file contains some common base for all metals
*/

#include "plask/material/material.hpp"

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
        f(f), G(G), w(w), wp(wp) {
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
        f(f), G(G), w(w), s(s), wp(wp) {
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

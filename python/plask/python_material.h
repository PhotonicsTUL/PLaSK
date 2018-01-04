#ifndef PLASK__PLASK_PYTHON_MATERIAL_H
#define PLASK__PLASK_PYTHON_MATERIAL_H

#include <plask/material/material.h>

namespace plask { namespace python {

/**
 * Cache used to hold constant material parameters for efficient access
 */
struct MaterialCache {
    plask::optional<double> lattC;
    plask::optional<double> Eg;
    plask::optional<double> CB;
    plask::optional<double> VB;
    plask::optional<double> Dso;
    plask::optional<double> Mso;
    plask::optional<Tensor2<double>> Me;
    plask::optional<Tensor2<double>> Mhh;
    plask::optional<Tensor2<double>> Mlh;
    plask::optional<Tensor2<double>> Mh;
    plask::optional<double> ac;
    plask::optional<double> av;
    plask::optional<double> b;
    plask::optional<double> d;
    plask::optional<double> c11;
    plask::optional<double> c12;
    plask::optional<double> c44;
    plask::optional<double> eps;
    plask::optional<double> chi;
    plask::optional<double> Na;
    plask::optional<double> Nd;
    plask::optional<double> Ni;
    plask::optional<double> Nf;
    plask::optional<double> EactD;
    plask::optional<double> EactA;
    plask::optional<Tensor2<double>> mob;
    plask::optional<Tensor2<double>> cond;
    plask::optional<double> A;
    plask::optional<double> B;
    plask::optional<double> C;
    plask::optional<double> D;
    plask::optional<Tensor2<double>> thermk;
    plask::optional<double> dens;
    plask::optional<double> cp;
    plask::optional<double> nr;
    plask::optional<double> absp;
    plask::optional<dcomplex> Nr;
    plask::optional<Tensor3<dcomplex>> NR;

    plask::optional<Tensor2<double>> mobe;
    plask::optional<Tensor2<double>> mobh;
    plask::optional<double> taue;
    plask::optional<double> tauh;
    plask::optional<double> Ce;
    plask::optional<double> Ch;
    plask::optional<double> e13;
    plask::optional<double> e15;
    plask::optional<double> e33;
    plask::optional<double> c13;
    plask::optional<double> c33;
    plask::optional<double> Psp;
};

/**
 * Material with another one as base
 */
struct MaterialWithBase: public Material {
    shared_ptr<Material> base;

    MaterialWithBase() = default;
    MaterialWithBase(const shared_ptr<Material>& base): base(base) {}
    MaterialWithBase(Material* base): base(base) {}
};

}} // namespace plask::python

#endif // PLASK__PLASK_PYTHON_MATERIAL_H

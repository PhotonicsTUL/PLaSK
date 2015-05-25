#ifndef PLASK__PLASK_PYTHON_MATERIAL_H
#define PLASK__PLASK_PYTHON_MATERIAL_H

#include <plask/material/material.h>

namespace plask { namespace python {

/**
 * Cache used to hold constant material parameters for efficient access
 */
struct MaterialCache {
    boost::optional<double> lattC;
    boost::optional<double> Eg;
    boost::optional<double> CB;
    boost::optional<double> VB;
    boost::optional<double> Dso;
    boost::optional<double> Mso;
    boost::optional<Tensor2<double>> Me;
    boost::optional<Tensor2<double>> Mhh;
    boost::optional<Tensor2<double>> Mlh;
    boost::optional<Tensor2<double>> Mh;
    boost::optional<double> ac;
    boost::optional<double> av;
    boost::optional<double> b;
    boost::optional<double> d;
    boost::optional<double> c11;
    boost::optional<double> c12;
    boost::optional<double> c44;
    boost::optional<double> eps;
    boost::optional<double> chi;
    boost::optional<double> Na;
    boost::optional<double> Nc;
    boost::optional<double> Nd;
    boost::optional<double> Nv;
    boost::optional<double> Ni;
    boost::optional<double> Nf;
    boost::optional<double> EactD;
    boost::optional<double> EactA;
    boost::optional<Tensor2<double>> mob;
    boost::optional<Tensor2<double>> cond;
    boost::optional<double> A;
    boost::optional<double> B;
    boost::optional<double> C;
    boost::optional<double> D;
    boost::optional<Tensor2<double>> thermk;
    boost::optional<double> dens;
    boost::optional<double> cp;
    boost::optional<double> nr;
    boost::optional<double> absp;
    boost::optional<dcomplex> Nr;
    boost::optional<Tensor3<dcomplex>> NR;

    boost::optional<Tensor2<double>> mobe;
    boost::optional<Tensor2<double>> mobh;
    boost::optional<double> Ae;
    boost::optional<double> Ah;
    boost::optional<double> Ce;
    boost::optional<double> Ch;
    boost::optional<double> e13;
    boost::optional<double> e15;
    boost::optional<double> e33;
    boost::optional<double> c13;
    boost::optional<double> c33;
    boost::optional<double> Psp;
};


}} // namespace plask::python

#endif // PLASK__PLASK_PYTHON_MATERIAL_H

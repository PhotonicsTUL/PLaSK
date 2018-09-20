#ifndef PLASK__MESH_HYMAN_H
#define PLASK__MESH_HYMAN_H

#include "../math.h"
#include "rectangular2d.h"
#include "rectilinear3d.h"
#include "rectangular3d.h"
#include "equilateral3d.h"
#include "interpolation.h"

namespace plask {

template <typename T> struct Hyman {
    static void filter(T& data, const T& a, const T& b) {
        T lim = 3 * min(abs(a), abs(b));
        if (data > lim) data = lim;
        else if (data < -lim) data = -lim;
    }
};

template <> struct Hyman<dcomplex> {
    static void filter(dcomplex& data, const dcomplex& a, const dcomplex& b) {
        double re = data.real(), im = data.imag();
        Hyman<double>::filter(re, real(a), real(b));
        Hyman<double>::filter(im, imag(a), imag(b));
        data = dcomplex(re,im);
    }
};

template <typename T> struct Hyman<Vec<2,T>> {
    static void filter(Vec<2,T>& data, const Vec<2,T>& a, const Vec<2,T>& b) {
        Hyman<T>::filter(data.c0, a.c0, b.c0);
        Hyman<T>::filter(data.c1, a.c1, b.c1);
    }
};

template <typename T> struct Hyman<Vec<3,T>> {
    static void filter(Vec<3,T>& data, const Vec<3,T>& a, const Vec<3,T>& b) {
        Hyman<T>::filter(data.c0, a.c0, b.c0);
        Hyman<T>::filter(data.c1, a.c1, b.c1);
        Hyman<T>::filter(data.c2, a.c2, b.c2);
    }
};

template <typename T> struct Hyman<Tensor2<T>> {
    static void filter(Tensor2<T>& data, const Tensor2<T>& a, const Tensor2<T>& b) {
        Hyman<T>::filter(data.c00, a.c00, b.c00);
        Hyman<T>::filter(data.c11, a.c11, b.c11);
    }
};

template <typename T> struct Hyman<Tensor3<T>> {
    static void filter(Tensor3<T>& data, const Tensor3<T>& a, const Tensor3<T>& b) {
        Hyman<T>::filter(data.c00, a.c00, b.c00);
        Hyman<T>::filter(data.c11, a.c11, b.c11);
        Hyman<T>::filter(data.c22, a.c22, b.c22);
        Hyman<T>::filter(data.c01, a.c01, b.c01);
    }
};


} // namespace plask

#endif // PLASK__MESH_HYMAN_H

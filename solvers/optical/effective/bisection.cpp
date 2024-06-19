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
#include "bisection.hpp"
#include <plask/parallel.hpp>

namespace plask { namespace optical { namespace effective {

#define CALL_FUN(result, re, im) \
    if (error) continue; \
    try { \
        result = fun(dcomplex(re, im)); \
    } catch (...) { \
        PLASK_PRAGMA(omp critical) \
        error = std::current_exception();\
    }

Contour::Contour(const Solver* solver, const std::function<dcomplex(dcomplex)>& fun, dcomplex corner0, dcomplex corner1, size_t ren, size_t imn):
    solver(solver), fun(fun), re0(real(corner0)), im0(imag(corner0)), re1(real(corner1)), im1(imag(corner1))
{
    bottom.reset(ren+1);
    right.reset(imn+1);
    top.reset(ren+1);
    left.reset(imn+1);

    double dr = (re1 - re0) / double(ren);
    double di = (im1 - im0) / double(imn);

    std::exception_ptr error;
    PLASK_OMP_PARALLEL
    {
        #pragma omp for nowait
        for (int i = 0; i < int(ren); ++i) {
            CALL_FUN(bottom[i], re0+i*dr, im0)
        }
        #pragma omp for nowait
        for (int i = 0; i < int(imn); ++i) {
            CALL_FUN(right[i], re1, im0+i*di)
        }
        #pragma omp for nowait
        for (int i = 1; i <= int(ren); ++i) {
            CALL_FUN(top[i], re0+i*dr, im1)
        }
        #pragma omp for
        for (int i = 1; i <= int(imn); ++i) {
            CALL_FUN(left[i], re0, im0+i*di)
        }
    }
    if (error) std::rethrow_exception(error);
    // Wrap values
    bottom[ren] = right[0];
    right[imn] = top[ren];
    top[0] = left[imn];
    left[0] = bottom[0];
}

namespace detail {
    // Inform that there is zero very close to the boundary
    static void contourLogZero(size_t i, size_t n, const Solver* solver, double r0, double i0, double r1, double i1) {
        const double f = double(2*std::ptrdiff_t(i)-1) / double(2*std::ptrdiff_t(n)-2);
        const double re = r0 + f*(r1-r0), im = i0 + f*(i1-i0);
        solver->writelog(LOG_WARNING, "Zero at contour in {0} (possibly not counted)", str(dcomplex(re,im)));
    }
}

int Contour::crossings(const DataVector<dcomplex>& line, double r0, double i0, double r1, double i1) const
{
    int wind = 0;
    for (size_t i = 1; i < line.size(); ++i) {
        if (real(line[i-1]) < 0. && real(line[i]) < 0.) {
            if (imag(line[i-1]) >= 0. && imag(line[i]) < 0.) {
                if (real(line[i-1]) >= 0. || real(line[i]) >= 0.) detail::contourLogZero(i, line.size(), solver, r0, i0, r1, i1);
                ++wind;
            } else if (imag(line[i-1]) < 0. && imag(line[i]) >= 0.) {
                if (real(line[i-1]) >= 0. || real(line[i]) >= 0.) detail::contourLogZero(i, line.size(), solver, r0, i0, r1, i1);
                --wind;
            }
        }
    }
    return wind;
}

#ifdef NDEBUG
std::pair<Contour,Contour> Contour::divide(double /*reps*/, double ieps) const
#else
std::pair<Contour,Contour> Contour::divide(double reps, double ieps) const
#endif
{
    Contour contoura(solver, fun), contourb(solver, fun);

    assert(re1-re0 > reps || im1-im0 > ieps);

    if (bottom.size() > right.size() || im1-im0 <= ieps) { // divide real axis
        double re = 0.5 * (re0 + re1);
        contoura.re0 = re0; contoura.im0 = im0; contoura.re1 = re; contoura.im1 = im1;
        contourb.re0 = re; contourb.im0 = im0; contourb.re1 = re1; contourb.im1 = im1;

        size_t n = (bottom.size()-1) / 2; // because of the > in the condition, bottom.size() is always >= 3

        size_t imn = right.size() - 1;
        DataVector<dcomplex> middle(imn+1);
        middle[0] = bottom[n]; middle[imn] = top[n];
        double di = (im1 - im0) / double(imn);

        std::exception_ptr error;
        PLASK_OMP_PARALLEL_FOR
        for (plask::openmp_size_t i = 1; i < imn; ++i) {
            if (error) continue;
            try {
                middle[i] = fun(dcomplex(re, im0+double(i)*di));
            } catch (...) {
                error = std::current_exception();
            }
        }
        if (error) std::rethrow_exception(error);

        contoura.left = left;
        contoura.right = middle;
        contoura.bottom = DataVector<dcomplex>(const_cast<dcomplex*>(&bottom[0]), n+1);
        contoura.top = DataVector<dcomplex>(const_cast<dcomplex*>(&top[0]), n+1);

        contourb.left = middle;
        contourb.right = right;
        contourb.bottom = DataVector<dcomplex>(const_cast<dcomplex*>(&bottom[n]), n+1);
        contourb.top = DataVector<dcomplex>(const_cast<dcomplex*>(&top[n]), n+1);

        return std::make_pair(std::move(contoura), std::move(contourb));

    } else {    // divide imaginary axis
        double im = 0.5 * (im0 + im1);
        contoura.re0 = re0; contoura.im0 = im0; contoura.re1 = re1; contoura.im1 = im;
        contourb.re0 = re0; contourb.im0 = im; contourb.re1 = re1; contourb.im1 = im1;

        size_t ren = bottom.size() - 1;
        DataVector<dcomplex> middle(ren+1);
        if (right.size() <= 2) {
            assert(ren == 1); // real axis also has only one segment
            middle[0] = fun(dcomplex(re0, im));
            middle[1] = fun(dcomplex(re1, im));
            contoura.left = DataVector<dcomplex>({left[0], middle[0]});
            contoura.right = DataVector<dcomplex>({right[0], middle[1]});
            contourb.left = DataVector<dcomplex>({middle[0], left[1]});
            contourb.right = DataVector<dcomplex>({middle[1], right[1]});
        } else {
            size_t n = (right.size()-1) / 2; // no less than 1
            middle[0] = left[n]; middle[ren] = right[n];
            double dr = (re1 - re0) / double(ren);

            std::exception_ptr error;
            PLASK_OMP_PARALLEL_FOR
            for (plask::openmp_size_t i = 1; i < ren; ++i) {
                if (error) continue;
                try {
                    middle[i] = fun(dcomplex(re0+double(i)*dr, im));
                } catch (...) {
                    error = std::current_exception();
                }
            }

            contoura.left = DataVector<dcomplex>(const_cast<dcomplex*>(&left[0]), n+1);
            contoura.right = DataVector<dcomplex>(const_cast<dcomplex*>(&right[0]), n+1);
            contourb.left = DataVector<dcomplex>(const_cast<dcomplex*>(&left[n]), n+1);
            contourb.right = DataVector<dcomplex>(const_cast<dcomplex*>(&right[n]), n+1);
        }

        contoura.bottom = bottom;
        contoura.top = middle;

        contourb.bottom = middle;
        contourb.top = top;

        return std::make_pair(std::move(contoura), std::move(contourb));

    }
}

namespace detail {

    struct ContourBisect {
        double reps, ieps;
        std::vector<std::pair<dcomplex,dcomplex>>& results;
        ContourBisect(double reps, double ieps, std::vector<std::pair<dcomplex,dcomplex>>& results): reps(reps), ieps(ieps), results(results) {}

        int operator()(const Contour& contour) {
            int wind = contour.winding();
            if (wind == 0)
                return 0;
            if (contour.re1-contour.re0 <= reps && contour.im1-contour.im0 <= ieps) {
                for(int i = 0; i != abs(wind); ++i)
                    results.push_back(std::make_pair(dcomplex(contour.re0, contour.im0), dcomplex(contour.re1, contour.im1)));
                return wind;
            }
            auto contours = contour.divide(reps, ieps);
            std::size_t w1 = (*this)(contours.first);
            std::size_t w2 = (*this)(contours.second);
            if (int(w1 + w2) < wind)
                contour.solver->writelog(LOG_WARNING, "Lost zero between {0} and {1}", str(dcomplex(contour.re0, contour.im0)), str(dcomplex(contour.re1, contour.im1)));
            else if (int(w1 + w2) > wind)
                contour.solver->writelog(LOG_WARNING, "New zero between {0} and {1}", str(dcomplex(contour.re0, contour.im0)), str(dcomplex(contour.re1, contour.im1)));
            return wind;
        }
    };
}

std::vector<std::pair<dcomplex,dcomplex>> findZeros(const Solver* solver, const std::function<dcomplex(dcomplex)>& fun,
                                                    dcomplex corner0, dcomplex corner1, size_t resteps, size_t imsteps, dcomplex eps)
{
    // Find first power of 2 not smaller than range/precision
    size_t Nr = 1, Ni = 1;
    for(; resteps > Nr; Nr <<= 1);
    for(; imsteps > Ni; Ni <<= 1);

    double reps = real(eps), ieps = imag(eps);

    std::vector<std::pair<dcomplex,dcomplex>> results;
    detail::ContourBisect bisection(reps, ieps, results);
    Contour contour(solver, fun, corner0, corner1, Nr, Ni);
    int zeros = abs(contour.winding());
    solver->writelog(LOG_DETAIL, "Looking for {4} zero{5} between {0} and {1} with {2}/{3} real/imaginary intervals",
                     str(corner0), str(corner1), Nr, Ni, zeros, (zeros!=1)?"s":"");
    bisection(contour);
    return results;
}


}}} // namespace plask::optical::effective

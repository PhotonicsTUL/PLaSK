#include "bisection.h"

namespace plask { namespace solvers { namespace effective {


Contour::Contour(const Solver* solver, const std::function<dcomplex(dcomplex)>& fun, dcomplex corner0, dcomplex corner1, size_t ren, size_t imn):
    solver(solver), fun(fun), re0(real(corner0)), im0(imag(corner0)), re1(real(corner1)), im1(imag(corner1))
{
    bottom.reset(ren+1);
    right.reset(imn+1);
    top.reset(ren+1);
    left.reset(imn+1);

    double dr = (re1 - re0) / ren;
    double di = (im1 - im0) / imn;

    #pragma omp parallel
    {
        #pragma for nowait
        for (size_t i = 0; i < ren; ++i) {
            bottom[i] = fun(dcomplex(re0+i*dr, im0));
        }
        #pragma for nowait
        for (size_t i = 0; i < imn; ++i) {
            right[i] = fun(dcomplex(re1, im0+i*di));
        }
        #pragma for nowait
        for (size_t i = 1; i <= ren; ++i) {
            top[i] = fun(dcomplex(re0+i*dr, im1));
        }
        #pragma for
        for (size_t i = 1; i <= imn; ++i) {
            left[i] = fun(dcomplex(re0, im0+i*di));
        }
    }
    // Wrap values
    bottom[ren] = right[0];
    right[imn] = top[ren];
    top[0] = left[imn];
    left[0] = bottom[0];
}

namespace detail {
    // Inform that there is zero very close to the boundary
    static void contourLogZero(size_t i, size_t n, const Solver* solver, double r0, double i0, double r1, double i1) {
        double f = (2.*i-1.) / (2.*n-2.);
        double re = r0 + f*(r1-r0), im = i0 + f*(i1-i0);
        solver->writelog(LOG_WARNING, "Zero at contour in %1% (possibly not counted)", str(dcomplex(re,im)));
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


std::pair<Contour,Contour> Contour::divide(double reps, double ieps) const
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
        double di = (im1 - im0) / imn;
        #pragma omp parallel for
        for (size_t i = 1; i < imn; ++i)
            middle[i] = fun(dcomplex(re, im0+i*di));

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
            double dr = (re1 - re0) / ren;
            #pragma omp parallel for
            for (size_t i = 1; i < ren; ++i)
                middle[i] = fun(dcomplex(re0+i*dr, im));
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
        std::vector<dcomplex>& results;
        ContourBisect(double reps, double ieps, std::vector<dcomplex>& results): reps(reps), ieps(ieps), results(results) {}

        int operator()(const Contour& contour) {
            int wind = contour.winding();
            if (wind == 0)
                return 0;
            if (contour.re1-contour.re0 <= reps && contour.im1-contour.im0 <= ieps) {
                for(int i = 0; i != abs(wind); ++i)
                    results.push_back(0.5 * dcomplex(contour.re0+contour.re1, contour.im0+contour.im1));
                return wind;
            }
            auto contours = contour.divide(reps, ieps);
            size_t w1 = (*this)(contours.first);
            size_t w2 = (*this)(contours.second);
            if (w1 + w2 != wind) {
                contour.solver->writelog(LOG_WARNING, "Lost zero between %1% and %2%",
                                         str(dcomplex(contour.re0, contour.im0)), str(dcomplex(contour.re1, contour.im1)));
            }
            return wind;
        }
    };
}

std::vector<dcomplex> findZeros(const Solver* solver, const std::function<dcomplex(dcomplex)>& fun,
                                dcomplex corner0, dcomplex corner1, size_t resteps, size_t imsteps, dcomplex eps)
{
    // Find first power of 2 not smaller than range/precision
    size_t Nr = 1, Ni = 1;
    for(; resteps > Nr; Nr <<= 1);
    for(; imsteps > Ni; Ni <<= 1);

    std::vector<dcomplex> results;
    detail::ContourBisect bisection(real(eps), imag(eps), results);
    Contour contour(solver, fun, corner0, corner1, Nr, Ni);
    int zeros = abs(contour.winding());
    solver->writelog(LOG_DETAIL, "Looking for %5% zero%6% between %1% and %2% with %3%/%4% real/imaginary intervals",
                     str(corner0), str(corner1), Nr, Ni, zeros, (zeros!=1)?"s":"");
    bisection(contour);
    return results;
}


}}} // namespace plask::solvers::effective

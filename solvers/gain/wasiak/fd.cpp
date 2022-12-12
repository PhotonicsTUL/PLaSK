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
#include <limits>
#include <cmath>

#include "fd.hpp"

namespace plask {


static double clenshaw(std::size_t n, const double* coeffs, double x, double left, double right) {
    double b1  = 0.;
    double b2 = 0.;

    double t  = (2.*x - left - right) / (right - left);
    double t2 = 2. * t;

    for(int k = n-1; k >= 1; k--) {
        double b0 = coeffs[k] + t2 * b1 - b2;
        b2 = b1; b1 = b0;
    }

    return 0.5 * coeffs[0] + t * b1 - b2 ;
}

// F_{1/2}(t);  -1 < x < 1
static const double fd_half_coeffs_1_1[] = {
  1.7177138871306189157,
  0.6192579515822668460,
  0.0932802275119206269,
  0.0047094853246636182,
 -0.0004243667967864481,
 -0.0000452569787686193,
  5.2426509519168e-6,
  6.387648249080e-7,
 -8.05777004848e-8,
 -1.04290272415e-8,
  1.3769478010e-9,
  1.847190359e-10,
 -2.51061890e-11,
 -3.4497818e-12,
  4.784373e-13,
  6.68828e-14,
 -9.4147e-15,
 -1.3333e-15,
  1.898e-16,
  2.72e-17,
 -3.9e-18,
 -6.e-19,
  1.e-19
};

// F_{1/2}(3/2(t+1) + 1);  1 < x < 4
static const double fd_half_coeffs_1_4[] = {
  7.651013792074984027,
  2.475545606866155737,
  0.218335982672476128,
 -0.007730591500584980,
 -0.000217443383867318,
  0.000147663980681359,
 -0.000021586361321527,
  8.07712735394e-7,
  3.28858050706e-7,
 -7.9474330632e-8,
  6.940207234e-9,
  6.75594681e-10,
 -3.10200490e-10,
  4.2677233e-11,
 -2.1696e-14,
 -1.170245e-12,
  2.34757e-13,
 -1.4139e-14,
 -3.864e-15,
  1.202e-15
};

// F_{1/2}(3(t+1) + 4);  4 < x < 10
static const double fd_half_coeffs_4_10[] = {
  29.584339348839816528,
  8.808344283250615592,
  0.503771641883577308,
 -0.021540694914550443,
  0.002143341709406890,
 -0.000257365680646579,
  0.000027933539372803,
 -1.678525030167e-6,
 -2.78100117693e-7,
  1.35218065147e-7,
 -3.3740425009e-8,
  6.474834942e-9,
 -1.009678978e-9,
  1.20057555e-10,
 -6.636314e-12,
 -1.710566e-12,
  7.75069e-13,
 -1.97973e-13,
  3.9414e-14,
 -6.374e-15,
  7.77e-16,
 -4.0e-17,
 -1.4e-17
};

// F_{1/2}(x) / x^(3/2);  10 < x < 30
static const double fd_half_coeffs_10_30[] = {
  1.5116909434145508537,
 -0.0036043405371630468,
  0.0014207743256393359,
 -0.0005045399052400260,
  0.0001690758006957347,
 -0.0000546305872688307,
  0.0000172223228484571,
 -5.3352603788706e-6,
  1.6315287543662e-6,
 -4.939021084898e-7,
  1.482515450316e-7,
 -4.41552276226e-8,
  1.30503160961e-8,
 -3.8262599802e-9,
  1.1123226976e-9,
 -3.204765534e-10,
  9.14870489e-11,
 -2.58778946e-11,
  7.2550731e-12,
 -2.0172226e-12,
  5.566891e-13,
 -1.526247e-13,
  4.16121e-14,
 -1.12933e-14,
  3.0537e-15,
 -8.234e-16,
  2.215e-16,
 -5.95e-17,
  1.59e-17,
 -4.0e-18
};

#define WITH_SIZE(arr) sizeof(arr)/sizeof(double), arr

static const double eta_table[] = {
    0.50000000000000000000000000000, // eta(0)
    0.693147180559945309417,         // eta(1)
    0.82246703342411321823620758332,
    0.90154267736969571404980362113,
    0.94703282949724591757650323447,
    0.97211977044690930593565514355,
    0.98555109129743510409843924448,
    0.99259381992283028267042571313,
    0.99623300185264789922728926008,
    0.99809429754160533076778303185,
    0.99903950759827156563922184570,
    0.99951714349806075414409417483,
    0.99975768514385819085317967871,
    0.99987854276326511549217499282,
    0.99993917034597971817095419226,
    0.99996955121309923808263293263,
    0.99998476421490610644168277496,
    0.99999237829204101197693787224,
    0.99999618786961011347968922641,
    0.99999809350817167510685649297,
    0.99999904661158152211505084256,
    0.99999952325821554281631666433,
    0.99999976161323082254789720494,
    0.99999988080131843950322382485,
    0.99999994039889239462836140314,
    0.99999997019885696283441513311,
    0.99999998509923199656878766181,
    0.99999999254955048496351585274,
    0.99999999627475340010872752767,
    0.99999999813736941811218674656,
    0.99999999906868228145397862728,
    0.99999999953434033145421751469,
    0.99999999976716989595149082282,
    0.99999999988358485804603047265,
    0.99999999994179239904531592388,
    0.99999999997089618952980952258,
    0.99999999998544809143388476396,
    0.99999999999272404460658475006,
    0.99999999999636202193316875550,
    0.99999999999818101084320873555,
    0.99999999999909050538047887809,
    0.99999999999954525267653087357,
    0.99999999999977262633369589773,
    0.99999999999988631316532476488,
    0.99999999999994315658215465336,
    0.99999999999997157829090808339,
    0.99999999999998578914539762720,
    0.99999999999999289457268000875,
    0.99999999999999644728633373609,
    0.99999999999999822364316477861,
    0.99999999999999911182158169283,
    0.99999999999999955591079061426,
    0.99999999999999977795539522974,
    0.99999999999999988897769758908,
    0.99999999999999994448884878594,
    0.99999999999999997224442439010,
    0.99999999999999998612221219410,
    0.99999999999999999306110609673,
    0.99999999999999999653055304826,
    0.99999999999999999826527652409,
    0.99999999999999999913263826204,
    0.99999999999999999956631913101,
    0.99999999999999999978315956551,
    0.99999999999999999989157978275,
    0.99999999999999999994578989138,
    0.99999999999999999997289494569,
    0.99999999999999999998644747284,
    0.99999999999999999999322373642,
    0.99999999999999999999661186821,
    0.99999999999999999999830593411,
    0.99999999999999999999915296705,
    0.99999999999999999999957648353,
    0.99999999999999999999978824176,
    0.99999999999999999999989412088,
    0.99999999999999999999994706044,
    0.99999999999999999999997353022,
    0.99999999999999999999998676511,
    0.99999999999999999999999338256,
    0.99999999999999999999999669128,
    0.99999999999999999999999834564,
    0.99999999999999999999999917282,
    0.99999999999999999999999958641,
    0.99999999999999999999999979320,
    0.99999999999999999999999989660,
    0.99999999999999999999999994830,
    0.99999999999999999999999997415,
    0.99999999999999999999999998708,
    0.99999999999999999999999999354,
    0.99999999999999999999999999677,
    0.99999999999999999999999999838,
    0.99999999999999999999999999919,
    0.99999999999999999999999999960,
    0.99999999999999999999999999980,
    0.99999999999999999999999999990,
    0.99999999999999999999999999995,
    0.99999999999999999999999999997,
    0.99999999999999999999999999999,
    0.99999999999999999999999999999,
    1.00000000000000000000000000000,
    1.00000000000000000000000000000,
    1.00000000000000000000000000000,
};

double fermiDiracHalf(double x)
{
    if(x < -1.) {
        // Series [Goano]
        double ex   = exp(x);
        double term = ex;
        double sum  = term;
        for(int n = 2; n < 100 ; n++) {
            double rat = (n - 1.) / n;
            term *= -ex * rat * sqrt(rat);
            sum  += term;
            if(fabs(term/sum) < std::numeric_limits<double>::epsilon()) break;
        }
        return sum;

    } else if(x < 1.) {
        return clenshaw(WITH_SIZE(fd_half_coeffs_1_1), x, -1., 1.);

    } else if(x < 4.) {
        return clenshaw(WITH_SIZE(fd_half_coeffs_1_4), x, 1., 4.);

    } else if(x < 10.) {
        return clenshaw(WITH_SIZE(fd_half_coeffs_4_10), x, 4., 10.);

    } else if(x < 30.) {
        double val = clenshaw(WITH_SIZE(fd_half_coeffs_10_30), x, 10., 30.);
        return x * sqrt(x) * val;

    } else {
        // Asymptotic expansion [Goano]
        const int itmax = 200;
        const double lg = 0.28468287047291918; // gammaln(ord + 2);
        double seqn = 0.5;
        double xm2  = 1. / x / x;
        double xgam = 1.;
        double add = std::numeric_limits<double>::max();

        for(int n = 1; n <= itmax; n++) {
            double add_previous = add;
            const double eta = (n <= 50)? eta_table[2*n]: 1.;
            xgam = xgam * xm2 * (1.5 - (2*n-2)) * (1.5 - (2*n-1));
            add  = eta * xgam;
            if(fabs(add) > fabs(add_previous)) throw "Divergent series";
            if(fabs(add/seqn) < std::numeric_limits<double>::epsilon()) break;
            seqn += add;
        }
        return 2.0 * seqn * exp(1.5 * log(x) - lg);
    }
}


} // namespace plask

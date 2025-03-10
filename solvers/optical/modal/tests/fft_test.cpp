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
#undef _GLIBCXX_DEBUG

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "FFT test"
#include <boost/test/unit_test.hpp>

#if !defined(_WIN32) && !defined(__WIN32__) && !defined(WIN32)
namespace boost { namespace unit_test { namespace ut_detail {
std::string normalize_test_case_name(const_string name) {
    return ( name[0] == '&' ? std::string(name.begin()+1, name.size()-1) : std::string(name.begin(), name.size() ));
}
}}}
#endif

#define CHECK_CLOSE_COLLECTION(aa, bb, tolerance) { \
    using std::distance; \
    using std::begin; \
    using std::end; \
    auto a = begin(aa), ae = end(aa); \
    auto b = begin(bb); \
    BOOST_REQUIRE_EQUAL(distance(a, ae), distance(b, end(bb))); \
    double total_error = 0.;\
    for(; a != ae; ++a, ++b) total_error += abs2(*a - *b); \
    BOOST_CHECK_SMALL(total_error, double(distance(begin(aa), ae)) * tolerance); \
}

#include "../fourier/fft.hpp"
using namespace plask;
using namespace plask::optical::modal;

BOOST_AUTO_TEST_SUITE(fft)

BOOST_AUTO_TEST_CASE(FullFFT)
{
    // Test forward transform
    DataVector<dcomplex> source = {
        -1.0,    0.000000000000,
        -1.0,    0.055363321799,
        -1.0,    0.103806228374,
        -1.0,    0.145328719723,
         1.0,    0.179930795848,
         1.0,    0.207612456747,
         1.0,    0.228373702422,
         1.0,    0.242214532872,
         1.0,    0.249134948097,
         1.0,    0.249134948097,
         1.0,    0.242214532872,
         1.0,    0.228373702422,
         1.0,    0.207612456747,
         1.0,    0.179930795848,
         0.0,    0.145328719723,
         0.0,    0.103806228374,
         0.0,    0.055363321799
    };
    DataVector<dcomplex> results = {
         0.352941176471 + 0.000000000000 * I,     0.166089965398,
        -0.491274029836 + 0.113535329482 * I,    -0.051241253516,
        -0.157995760054 + 0.145143943954 * I,    -0.013257953397,
         0.083506361227 + 0.088790058872 * I,    -0.006242913701,
         0.087829599463 + 0.012803414761 * I,    -0.003811927951,
        -0.049728610250 - 0.014017774315 * I,    -0.002716737190,
        -0.127560224051 + 0.011613720606 * I,    -0.002159074523,
        -0.062551102267 + 0.036882532225 * I,    -0.001870162991,
         0.041303177531 + 0.020525883206 * I,    -0.001744959431,
         0.041303177531 - 0.020525883206 * I,    -0.001744959431,
        -0.062551102267 - 0.036882532225 * I,    -0.001870162991,
        -0.127560224051 - 0.011613720606 * I,    -0.002159074523,
        -0.049728610250 + 0.014017774315 * I,    -0.002716737190,
         0.087829599463 - 0.012803414761 * I,    -0.003811927951,
         0.083506361227 - 0.088790058872 * I,    -0.006242913701,
        -0.157995760054 - 0.145143943954 * I,    -0.013257953397,
        -0.491274029836 - 0.113535329482 * I,    -0.051241253516
    };
    DataVector<dcomplex> data = source.copy();

    FFT::Forward1D(2, 17, FFT::SYMMETRY_NONE).execute(data.data());
    CHECK_CLOSE_COLLECTION(data, results, 1e-16)

    FFT::Backward1D(2, 17, FFT::SYMMETRY_NONE).execute(data.data());
    CHECK_CLOSE_COLLECTION(data, source, 1e-16)
}

BOOST_AUTO_TEST_CASE(EvenFTT)
{
    // Test symmetric forward transform
    DataVector<dcomplex> source = {
        -1. + 1.96157056081 * I,    0.03027344,
        -1. + 1.66293922461 * I,    0.08496094,
        -1. + 1.11114046604 * I,    0.13183594,
         1. + 0.39018064403 * I,    0.17089844,
         1. - 0.39018064403 * I,    0.20214844,
         1. - 1.11114046604 * I,    0.22558594,
         1. - 1.66293922461 * I,    0.24121094,
         1. - 1.96157056081 * I,    0.24902344
    };
    DataVector<dcomplex> results = {
           0.250000000000000000,       0.1669921875000000000,
          -0.591956281431344490 + I,  -0.0503306486148838860,
          -0.230969883127821680,      -0.0123215704292927740,
           0.086101497529203347,      -0.0052613656320620386,
           0.176776695296636890,      -0.0027621358640099515,
           0.057531181341875015,      -0.0015695539354374734,
          -0.095670858091272459,      -0.0008756670491561827,
          -0.117747425324767870,      -0.00039611189655973031
    };
    DataVector<dcomplex> data = source.copy();

    FFT::Forward1D(2, 8, FFT::SYMMETRY_EVEN_2).execute(data.data());
    CHECK_CLOSE_COLLECTION(data, results, 1e-16)
    // for (int i = 0; i != 16; i += 2) {
    //     std::cerr << data[i].real() << " " << results[i].real() << " " << (results[i].real()/data[i].real()) << "\n";
    // }

    FFT::Backward1D(2, 8, FFT::SYMMETRY_EVEN_2).execute(data.data());
    CHECK_CLOSE_COLLECTION(data, source, 1e-16)
}


BOOST_AUTO_TEST_CASE(FTT2D) {

    DataVector<dcomplex> source = {
        0.45620484 + 0.41708854 * I,  0.94044409 + 0.84322012 * I,  0.48811984 + 0.15906212 * I,  0.65106851 + 0.83611012 * I,
        0.95936019 + 0.89303437 * I,  0.11658367 + 0.30349193 * I,  0.80063473 + 0.28900787 * I,  0.70242832 + 0.50029993 * I,
        0.99968088 + 0.66600515 * I,  0.04235006 + 0.80731292 * I,  0.21483692 + 0.12231301 * I,  0.49802439 + 0.76753812 * I,
        0.87514767 + 0.69301587 * I,  0.97164344 + 0.19984670 * I,  0.30878141 + 0.46510830 * I,  0.17120057 + 0.88271407 * I,
        0.19539467 + 0.26338963 * I,  0.67873011 + 0.83122420 * I,  0.36754428 + 0.52678457 * I,  0.23674332 + 0.93697281 * I,
        0.49070677 + 0.13989021 * I,  0.89710110 + 0.96517860 * I,  0.81690291 + 0.46331055 * I,  0.51356861 + 0.85189615 * I,
        0.97889703 + 0.82403223 * I,  0.27653356 + 0.16416582 * I,  0.55839489 + 0.97132576 * I,  0.16205838 + 0.29809225 * I,
        0.59524709 + 0.75016220 * I,  0.11830271 + 0.11512313 * I,  0.26292468 + 0.28149780 * I,  0.05072159 + 0.40484505 * I,
        0.96499110 + 0.70463706 * I,  0.30206322 + 0.48698087 * I,  0.60382174 + 0.66574441 * I,  0.06752426 + 0.44374348 * I,
        0.72586154 + 0.08601000 * I,  0.30160112 + 0.53539594 * I,  0.61004256 + 0.35241993 * I,  0.12331822 + 0.40778225 * I,
        0.07472216 + 0.46306215 * I,  0.83688102 + 0.17849970 * I,  0.91469491 + 0.76788978 * I,  0.30968558 + 0.56394684 * I,
        0.82965080 + 0.77057865 * I,  0.71096797 + 0.86578195 * I,  0.54261026 + 0.95414125 * I,  0.91996870 + 0.31419490 * I,
        0.73878501 + 0.22389432 * I,  0.47969572 + 0.35863825 * I,  0.89688296 + 0.79395957 * I,  0.55132401 + 0.14091447 * I,
        0.80541572 + 0.88300110 * I,  0.28858964 + 0.49966569 * I,  0.49161168 + 0.97692278 * I,  0.65358160 + 0.50904435 * I,
        0.54899159 + 0.18659225 * I,  0.78882520 + 0.42276475 * I,  0.33364548 + 0.47599538 * I,  0.80441527 + 0.43937997 * I,
        0.88078556 + 0.56652283 * I,  0.68987376 + 0.32151232 * I,  0.35475892 + 0.90500881 * I,  0.93712591 + 0.31717507 * I
    };

    DataVector<dcomplex> results = {
        0.55435929 + 0.53460721 * I, -0.00469486 - 0.01261273 * I,  0.02871543 - 0.02698444 * I, -0.02133275 - 0.00774750 * I,
        0.06082980 + 0.01856181 * I, -0.03933173 - 0.02561188 * I,  0.05108565 + 0.00699772 * I, -0.00992242 - 0.01862250 * I,
        0.04439237 + 0.05233980 * I,  0.03316517 - 0.01574809 * I,  0.00609536 + 0.04184079 * I,  0.03650046 - 0.08239704 * I,
       -0.10934701 - 0.06528122 * I, -0.02407978 - 0.05248308 * I,  0.07601414 - 0.02281543 * I, -0.00336507 + 0.01836583 * I,
        0.02733193 - 0.03625005 * I,  0.03861884 + 0.03675802 * I,  0.00871392 + 0.04523171 * I,  0.02418076 - 0.01661509 * I,
       -0.00399539 + 0.00654292 * I, -0.02710212 + 0.03902478 * I,  0.03929127 + 0.03543257 * I, -0.03135683 + 0.02586604 * I,
       -0.00236280 + 0.04008876 * I,  0.01283329 - 0.05343289 * I,  0.01009425 - 0.00098367 * I, -0.06298394 - 0.04417410 * I,
        0.04081262 - 0.04419021 * I, -0.05695920 - 0.04061685 * I, -0.01645388 - 0.03460166 * I,  0.03700757 - 0.00730647 * I,
        0.00553590 + 0.00566517 * I,  0.02222765 + 0.00762365 * I, -0.01439696 - 0.00806300 * I, -0.02184168 - 0.02555778 * I,
        0.03004255 - 0.06894938 * I, -0.01333101 + 0.02871952 * I, -0.04908167 - 0.01046693 * I,  0.00998071 + 0.00469345 * I,
        0.01836471 - 0.01272118 * I,  0.02171606 - 0.02634274 * I, -0.06533582 + 0.04108809 * I, -0.02380615 + 0.01344260 * I,
       -0.05871008 + 0.08888313 * I, -0.00257139 + 0.04419675 * I, -0.03383656 + 0.04886292 * I, -0.03251913 - 0.00727067 * I,
       -0.03634787 - 0.00877052 * I,  0.01811706 - 0.02934202 * I,  0.01352709 + 0.02220216 * I, -0.01199163 + 0.00193924 * I,
        0.06337336 - 0.00553163 * I, -0.00458471 + 0.01311777 * I, -0.00237988 + 0.01496717 * I,  0.00635827 + 0.01403733 * I,
        0.02808199 - 0.04479481 * I, -0.06637419 + 0.02241185 * I,  0.00472578 + 0.03048360 * I,  0.03465185 - 0.02375195 * I,
        0.01371852 - 0.02065157 * I, -0.06592436 + 0.02348020 * I, -0.02507549 + 0.03232162 * I, -0.02286242 - 0.04201658 * I
    };

    DataVector<dcomplex> data = source.copy();

    FFT::Forward2D(1, 8, 8, FFT::SYMMETRY_NONE, FFT::SYMMETRY_NONE).execute(data.data());
    CHECK_CLOSE_COLLECTION(data, results, 1e-16)
}


BOOST_AUTO_TEST_SUITE_END()

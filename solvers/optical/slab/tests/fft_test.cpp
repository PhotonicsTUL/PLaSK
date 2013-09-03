#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "FFT test"
#include <boost/test/unit_test.hpp>

#define CHECK_CLOSE_COLLECTION(aa, bb, tolerance) { \
    using std::distance; \
    using std::begin; \
    using std::end; \
    auto a = begin(aa), ae = end(aa); \
    auto b = begin(bb); \
    BOOST_REQUIRE_EQUAL(distance(a, ae), distance(b, end(bb))); \
    for(; a != ae; ++a, ++b) { \
        BOOST_CHECK_SMALL(abs2(*a-*b), tolerance); \
    } \
}

#include "../fft.h"
using namespace plask;
using namespace plask::solvers::slab;


BOOST_AUTO_TEST_SUITE(fft)

FFT fourier;

BOOST_AUTO_TEST_CASE(ForwardAsymmetric)
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

    fourier.forward(2, 17, data.data(), FFT::SYMMETRY_NONE);
    CHECK_CLOSE_COLLECTION(data, results, 1e-16)

    fourier.backward(2, 17, data.data(), FFT::SYMMETRY_NONE);
    CHECK_CLOSE_COLLECTION(data, source, 1e-16)
}

BOOST_AUTO_TEST_CASE(ForwardSymmetric)
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

    fourier.forward(2, 8, data.data(), FFT::SYMMETRY_EVEN);
    CHECK_CLOSE_COLLECTION(data, results, 1e-16)

    fourier.backward(2, 8, data.data(), FFT::SYMMETRY_EVEN);
    CHECK_CLOSE_COLLECTION(data, source, 1e-16)
}

BOOST_AUTO_TEST_SUITE_END()

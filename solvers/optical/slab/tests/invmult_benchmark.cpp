#include <plask/plask.hpp>
#include "../matrices.h"
#include "../fourier/toeplitz.h"

using namespace plask;
using namespace plask::optical::slab;


const int n = 1000;


const int N = 2*n + 1;
const int nN = 4*n + 1;
const double b = 2*PI;

const dcomplex e(3., 0.1);

int main() {
    DataVector<dcomplex> eps(nN);

    for (int k = -N+1; k < N; ++k) {
        size_t j = (k>=0)? k : k + nN;
        dcomplex ff = (j)? (dcomplex(0., 0.5/PI/k) * (exp(dcomplex(0., -b*k*0.25)) - exp(dcomplex(0., +b*k*0.15)))) : 0.5;
        eps[j] = e * ff;
        // eps[j] = dcomplex(k, 0.);
    }
    eps[0] += 1.;

//     for(size_t j = 0; j < nN; ++j)
//         std::cerr << str(eps[j], "{:9.6f}{:+09.6f}j") << " ";
//     std::cerr << "\n\n";

    cmatrix T(N, N);

//     dcomplex* p = T.data();
//     for (size_t j = 0; j  < N; ++j, p += N+1) {
//         std::copy_n(eps.data(), N-j, p);                    // below diagonal
//         std::copy_n(eps.data()+nN-j, j, T.data()+N*j);    // above diagonal
//     }
    for (int i = -n; i <= n; ++i) {
        for (int j = -n; j <= n; ++j) {
            int ij = i-j; if (ij < 0) ij += nN;
            T((i>=0)?i:i+N, (j>=0)?j:j+N) = eps[ij];
            // T(i+n, j+n) = eps[ij];
        }
    }

    cmatrix X(N, N, 0.);
    for (size_t i = 0; i < N; ++i) {
        X(i, i) = 1.;
    }

//     std::cerr << "T = matrix([\n";
//     for (size_t i = 0; i < N; ++i) {
//         std::cerr << "    [ ";
//         for(size_t j = 0; j < N; ++j) {
//             std::cerr << str(T(i,j), "{:9.6f}{:+09.6f}j") << ", ";
//         }
//         std::cerr << "],\n";
//     }
//     std::cerr << "])\n\n";

    invmult(T, X);
//     ToeplitzLevinson(eps, X);

    // std::cerr << "X = matrix([\n";
    // for (size_t i = 0; i < N; ++i) {
    //     std::cerr << "    [ ";
    //     for(size_t j = 0; j < N; ++j) {
    //         std::cerr << str(X(i,j), "{:9.6f}{:+09.6f}j") << ", ";
    //     }
    //     std::cerr << "],\n";
    // }
    // std::cerr << "])\n";

}

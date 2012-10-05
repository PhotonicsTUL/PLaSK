#include "functions.h"

namespace plask { namespace phys {

double Varshni(double Eg0K, double alpha, double beta, double T) {
    return Eg0K - alpha * T * T / (T + beta);
}


}} // namespace plask::phys
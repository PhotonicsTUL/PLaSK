#include "functions.h"
#include "constants.h"

namespace plask { namespace phys {

double Varshni(double Eg0K, double alpha, double beta, double T) {
    return Eg0K - alpha * T * T / (T + beta);
}

double PhotonEnergy(double wl) {
    return h_eV * c * 1e9 / wl;
}


}} // namespace plask::phys

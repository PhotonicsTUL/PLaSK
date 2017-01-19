#ifndef PLASK__SOLVER__THERMAL_STATIC_COMMON_H
#define PLASK__SOLVER__THERMAL_STATIC_COMMON_H

#include <plask/plask.hpp>

namespace plask { namespace thermal { namespace tstatic {

/// Boundary condition: convection
struct Convection
{
    double coeff;   ///< convection coefficient [W/(m^2*K)]
    double ambient; ///< ambient temperature [K]
    Convection(double coeff, double amb): coeff(coeff), ambient(amb) {}
    Convection() = default;
    friend inline std::ostream& operator<<(std::ostream& out, const Convection& to_print) {
        return out << to_print.coeff << "(" << to_print.ambient << "K)";
    }
};

/// Boundary condition: radiation
struct Radiation
{
    double emissivity;  ///< surface emissivity [-]
    double ambient;     ///< ambient temperature [K]
    Radiation(double emiss, double amb): emissivity(emiss), ambient(amb) {}
    Radiation() = default;
    friend inline std::ostream& operator<<(std::ostream& out, const Radiation& to_print) {
        return out << to_print.emissivity << "(" << to_print.ambient << "K)";
    }
};

/// Choice of matrix factorization algorithms
enum Algorithm {
    ALGORITHM_CHOLESKY, ///< Cholesky factorization
    ALGORITHM_GAUSS,    ///< Gauss elimination of asymmetrix matrix (slower but safer as it uses pivoting)
    ALGORITHM_ITERATIVE ///< Conjugate gradient iterative solver
};


    
    
}} // # namespace thermal::tstatic
    
template <> inline thermal::tstatic::Convection parseBoundaryValue<thermal::tstatic::Convection>(const XMLReader& tag_with_value)
{
    return thermal::tstatic::Convection(tag_with_value.requireAttribute<double>("coeff"), tag_with_value.requireAttribute<double>("ambient"));
}

template <> inline thermal::tstatic::Radiation parseBoundaryValue<thermal::tstatic::Radiation>(const XMLReader& tag_with_value)
{
    return thermal::tstatic::Radiation(tag_with_value.requireAttribute<double>("emissivity"), tag_with_value.requireAttribute<double>("ambient"));
}


} // # namespace plask::thermal::tstatic

#endif // PLASK__SOLVER__THERMAL_STATIC_COMMON_H


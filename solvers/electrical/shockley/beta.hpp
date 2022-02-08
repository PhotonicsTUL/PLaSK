#ifndef PLASK__MODULE_ELECTRICAL_BETA2D_H
#define PLASK__MODULE_ELECTRICAL_BETA2D_H

#include "electr2d.hpp"
#include "electr3d.hpp"

namespace plask { namespace electrical { namespace shockley {

/**
 * Solver performing calculations in 2D Cartesian or Cylindrical space using finite element method
 */
template <typename GeometryT>
struct PLASK_SOLVER_API BetaSolver : public std::conditional<std::is_same<GeometryT, Geometry3D>::value,
                                                             ElectricalFem3DSolver,
                                                             ElectricalFem2DSolver<GeometryT>>::type {
    typedef typename std::conditional<std::is_same<GeometryT, Geometry3D>::value,
                                      ElectricalFem3DSolver,
                                      ElectricalFem2DSolver<GeometryT>>::type BaseClass;

  protected:
    std::vector<double> js;    ///< p-n junction parameter [A/m^2]
    std::vector<double> beta;  ///< p-n junction parameter [1/V]

    /** Compute voltage drop of the active region
     *  \param n active region number
     *  \param U junction voltage
     *  \param jy vertical current [kA/cm²]
     *  \param T temperature [K]
     */
    Tensor2<double> activeCond(size_t n, double PLASK_UNUSED(U), double jy, double PLASK_UNUSED(T)) override {
        jy = abs(jy);
        return Tensor2<double>(0., 10. * jy * this->active[n].height * getBeta(n) / log(1e7 * jy / getJs(n) + 1.));
    }

  public:
    /// Return beta.
    double getBeta(size_t n) const {
        if (beta.size() <= n) throw Exception("{0}: no beta given for junction {1}", this->getId(), n);
        return beta[n];
    }
    /// Set new beta and invalidate the solver.
    void setBeta(size_t n, double beta) {
        if (this->beta.size() <= n) {
            this->beta.reserve(n + 1);
            for (size_t s = this->beta.size(); s <= n; ++s) this->beta.push_back(NAN);
        }
        this->beta[n] = beta;
        this->invalidate();
    }

    /// Return js
    double getJs(size_t n) const {
        if (js.size() <= n) throw Exception("{0}: no js given for junction {1}", this->getId(), n);
        return js[n];
    }
    /// Set new js and invalidate the solver
    void setJs(size_t n, double js) {
        if (this->js.size() <= n) {
            this->js.reserve(n + 1);
            for (size_t s = this->js.size(); s <= n; ++s) this->js.push_back(1.);
        }
        this->js[n] = js;
        this->invalidate();
    }

    void loadConfiguration(XMLReader& source, Manager& manager) override;

    BetaSolver(const std::string& name = "");

    std::string getClassName() const override;

    ~BetaSolver();
};

}}}  // namespace plask::electrical::shockley

#endif

/** \file
 * Sample Python wrapper for your solver.
 */
#include <cmath>
#include <plask/python.hpp>
#include <plask/python_util/ufunc.h>
using namespace plask;
using namespace plask::python;

#include "../freecarrier.h"
using namespace plask::gain::freecarrier;

#ifndef NDEBUG
template <typename GeometryT>
static py::object FreeCarrier_detEl(FreeCarrierGainSolver<GeometryT>* self, py::object E, size_t reg=0, size_t well=0) {
    self->initCalculation();
    typename FreeCarrierGainSolver<GeometryT>::ActiveRegionParams params(self, self->regions[reg], self->getT0());
    return PARALLEL_UFUNC<double>([self,&params,well](double x){return self->detEl(x, params, well);}, E);
}

template <typename GeometryT>
static py::object FreeCarrier_detHh(FreeCarrierGainSolver<GeometryT>* self, py::object E, size_t reg=0, size_t well=0) {
    self->initCalculation();
    typename FreeCarrierGainSolver<GeometryT>::ActiveRegionParams params(self, self->regions[reg], self->getT0());
    return PARALLEL_UFUNC<double>([self,&params,well](double x){return self->detHh(x, params, well);}, E);
}

template <typename GeometryT>
static py::object FreeCarrier_detLh(FreeCarrierGainSolver<GeometryT>* self, py::object E, size_t reg=0, size_t well=0) {
    self->initCalculation();
    typename FreeCarrierGainSolver<GeometryT>::ActiveRegionParams params(self, self->regions[reg], self->getT0());
    return PARALLEL_UFUNC<double>([self,&params,well](double x){return self->detLh(x, params, well);}, E);
}

template <typename GeometryT>
static py::object FreeCarrierGainSolver_getN(FreeCarrierGainSolver<GeometryT>* self, py::object F, py::object pT, size_t reg=0) {
    double T = (pT == py::object())? self->getT0() : py::extract<double>(pT);
    self->initCalculation();
    typename FreeCarrierGainSolver<GeometryT>::ActiveRegionParams params(self, self->params0[reg], T);
    return PARALLEL_UFUNC<double>([self,T,reg,&params](double x){return self->getN(x, T, params);}, F);
}

template <typename GeometryT>
static py::object FreeCarrierGainSolver_getP(FreeCarrierGainSolver<GeometryT>* self, py::object F, py::object pT, size_t reg=0) {
    double T = (pT == py::object())? self->getT0() : py::extract<double>(pT);
    self->initCalculation();
    typename FreeCarrierGainSolver<GeometryT>::ActiveRegionParams params(self, self->params0[reg], T);
    return PARALLEL_UFUNC<double>([self,T,reg,&params](double x){return self->getP(x, T, params);}, F);
}
#endif

template <typename GeometryT>
static py::object FreeCarrier_getLevels(FreeCarrierGainSolver<GeometryT>& self, py::object To)
{
    static const char* names[3] = { "el", "hh", "lh" };

    //TODO consider temperature
    self.initCalculation();
    py::list result;
    for (size_t reg = 0; reg < self.regions.size(); ++reg) {
        py::dict info;
        for (size_t i = 0; i < 3; ++i) {
            py::list lst;
            for (const auto& l: self.params0[reg].levels[i]) lst.append(l.E);
            info[names[i]] = lst;
        }
        result.append(info);
    }
    return result;
}

template <typename GeometryT>
static py::object FreeCarrier_getFermiLevels(FreeCarrierGainSolver<GeometryT>* self, double N, py::object To, int reg)
{
    double T = (To == py::object())? self->getT0() : py::extract<double>(To);
    if (reg < 0) reg = self->regions.size() + reg;
    if (reg < 0 || reg >= self->regions.size()) throw IndexError("{}: Bad active region index", self->getId());
    self->initCalculation();
    double Fc{NAN}, Fv{NAN};
    typename FreeCarrierGainSolver<GeometryT>::ActiveRegionParams params(self, self->params0[reg], T);
    self->findFermiLevels(Fc, Fv, N, T, params);
    return py::make_tuple(Fc, Fv);
}

template <typename GeometryT>
static shared_ptr<GainSpectrum<GeometryT>> FreeCarrierGetGainSpectrum2(FreeCarrierGainSolver<GeometryT>* solver, double c0, double c1) {
    return solver->getGainSpectrum(Vec<2>(c0,c1));
}

template <typename GeometryT>
static py::object FreeCarrierGainSpectrum__call__(GainSpectrum<GeometryT>& self, py::object wavelengths) {
   // return PARALLEL_UFUNC<double>([&](double x){return self.getGain(x);}, wavelengths);
    try {
        return py::object(self.getGain(py::extract<double>(wavelengths)));
    } catch (py::error_already_set) {
        PyErr_Clear();

        PyArrayObject* inarr = (PyArrayObject*)PyArray_FROM_OT(wavelengths.ptr(), NPY_DOUBLE);
        if (inarr == NULL || PyArray_TYPE(inarr) != NPY_DOUBLE) {
            Py_XDECREF(inarr);
            throw TypeError("{}: Wavelengths for spectrum must be a scalar float or one-dimensional array of floats",
                            self.solver->getId());
        }
        if(PyArray_NDIM(inarr) != 1) {
            Py_DECREF(inarr);
            throw TypeError("{}: Wavelengths for spectrum must be a scalar float or one-dimensional array of floats",
                            self.solver->getId());
        }

        npy_intp size = PyArray_DIMS(inarr)[0];
        npy_intp dims[] = { size, 2 };
        PyObject* outarr = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
        if (outarr == nullptr) {
            Py_DECREF(inarr);
            throw plask::CriticalException("Cannot create array for gain");
        }

        double* indata = static_cast<double*>(PyArray_DATA(inarr));
        Tensor2<double>* outdata = static_cast<Tensor2<double>*>(PyArray_DATA((PyArrayObject*)outarr));

        npy_intp instride = PyArray_STRIDES(inarr)[0] / sizeof(double);

        std::exception_ptr error;

        #pragma omp parallel for
        for (npy_intp i = 0; i < size; ++i) {
            if (!error) try {
                outdata[i] = self.getGain(indata[i*instride]);
            } catch (...) {
                #pragma omp critical
                error = std::current_exception();
            }
        }
        if (error) {
            Py_XDECREF(inarr);
            std::rethrow_exception(error);
        }

        Py_DECREF(inarr);

        return py::object(py::handle<>(outarr));
    }
}


BOOST_PYTHON_MODULE(freecarrier)
{
    plask_import_array();

    {CLASS(FreeCarrierGainSolver<Geometry2DCartesian>, "FreeCarrier2D", u8"Quantum-well gain using free-carrier approximation for two-dimensional Cartesian geometry.")
#ifndef NDEBUG
        solver.def("det_El", &FreeCarrier_detEl<Geometry2DCartesian>, (arg("E"), arg("reg")=0, arg("well")=0));
        solver.def("det_Hh", &FreeCarrier_detHh<Geometry2DCartesian>, (arg("E"), arg("reg")=0, arg("well")=0));
        solver.def("det_Lh", &FreeCarrier_detLh<Geometry2DCartesian>, (arg("E"), arg("reg")=0, arg("well")=0));
        solver.def("getN", &FreeCarrierGainSolver_getN<Geometry2DCartesian>, (arg("F"), arg("T")=py::object(), arg("reg")=0));
        solver.def("getP", &FreeCarrierGainSolver_getP<Geometry2DCartesian>, (arg("F"), arg("T")=py::object(), arg("reg")=0));
#endif
//         RW_FIELD(quick_levels,
//                  "Compute levels only once and simply shift for different temperatures?\n\n"
//                  "Setting this to True strongly increases computation speed, but canis  make the results\n"
//                  "less accurate for high temperatures.");
        solver.def("get_energy_levels", &FreeCarrier_getLevels<Geometry2DCartesian>, arg("T")=py::object(),
            u8"Get energy levels in quantum wells.\n\n"
            u8"Compute energy levels in quantum wells for electrons, heavy holes and\n"
            u8"light holes.\n\n"
            u8"Args:\n"
            u8"    T (float or ``None``): Temperature to get the levels. If this argument is\n"
            u8"                           ``None``, the estimates for temperature :py:attr:`T0`\n"
            u8"                           are returned.\n\n"
            u8"Returns:\n"
            u8"    list: List with dictionaries with keys `el`, `hh`, and `lh` with levels for\n"
            u8"          electrons, heavy holes and light holes. Each list element corresponds\n"
            u8"          to one active region.\n"
        );
        solver.def("get_fermi_levels", &FreeCarrier_getFermiLevels<Geometry2DCartesian>, (arg("n"), arg("T")=py::object(), arg("reg")=0),
            u8"Get quasi Fermi levels.\n\n"
            u8"Compute quasi-Fermi levels in specified active region.\n"
            u8"Args:\n"
            u8"    n (float): Carriers concentration to determine the levels for\n"
            u8"               [1/cm\\ :sup:`3`\\ ].\n"
            u8"    T (float or ``None``): Temperature to get the levels. If this argument is\n"
            u8"                           ``None``, the estimates for temperature :py:attr:`T0`\n"
            u8"                           are returned.\n\n"
            u8"    reg (int): Active region number.\n"
            u8"Returns:\n"
            u8"    tuple: Two-element tuple with quasi-Fermi levels for electrons and holes.\n"
        );
        RW_PROPERTY(T0, getT0, setT0, "Reference temperature.\n\nIn this temperature levels estimates are computed.");
        RW_PROPERTY(matrix_element, getMatrixElem, setMatrixElem,
                    u8"Momentum matrix element.\n\n"
                    u8"Value of the squared matrix element in gain computations. If it is not set it\n"
                    u8"is estimated automatically. (float [eV×m0])");
        RW_PROPERTY(lifetime, getLifeTime, setLifeTime, "Average carriers lifetime.\n\n"
            "This parameter is used for gain spectrum broadening. (float [ps])");
        RW_PROPERTY(strained, getStrained, setStrained,
                    u8"Boolean attribute indicating if the solver should consider strain in the active\n"
                    u8"region.\n\n"
                    u8"If set to ``True`` then there must a layer with the role *substrate* in\n"
                    u8"the geometry. The strain is computed by comparing the atomic lattice constants\n"
                    u8"of the substrate and the quantum wells.");
        RECEIVER(inTemperature, "");
        RECEIVER(inCarriersConcentration, "");
        PROVIDER(outGain, "");
        solver.def("spectrum", &__Class__::getGainSpectrum, py::arg("point"), py::with_custodian_and_ward_postcall<0,1>(),
                   u8"Get gain spectrum at given point.\n\n"
                   u8"Args:\n"
                   u8"    point (vec): Point to get gain at.\n"
                   u8"    c0, c1 (float): Coordinates of the point to get gain at.\n\n"
                   u8"Returns:\n"
                   u8"    :class:`XFermiCyl.Spectrum`: Spectrum object.\n");
        solver.def("spectrum", FreeCarrierGetGainSpectrum2<Geometry2DCartesian>, (py::arg("c0"), "c1"), py::with_custodian_and_ward_postcall<0,1>());

        py::scope scope = solver;
        py::class_<GainSpectrum<Geometry2DCartesian>,plask::shared_ptr<GainSpectrum<Geometry2DCartesian>>, boost::noncopyable>("Spectrum",
            u8"Gain spectrum object. You can call it like a function to get gains for different vavelengths.",
            py::no_init)
            .def("__call__", &FreeCarrierGainSpectrum__call__<Geometry2DCartesian>, py::arg("lam"),
                 u8"Get gain at specified wavelength.\n\n"
                 u8"Args:\n"
                 u8"    lam (float): Wavelength to get the gain at.\n"
            )
        ;
    }

    {CLASS(FreeCarrierGainSolver<Geometry2DCylindrical>, "FreeCarrierCyl", u8"Quantum-well gain using free-carrier approximation for cylindrical geometry.")
#ifndef NDEBUG
        solver.def("det_El", &FreeCarrier_detEl<Geometry2DCylindrical>, (arg("E"), arg("reg")=0, arg("well")=0));
        solver.def("det_Hh", &FreeCarrier_detHh<Geometry2DCylindrical>, (arg("E"), arg("reg")=0, arg("well")=0));
        solver.def("det_Lh", &FreeCarrier_detLh<Geometry2DCylindrical>, (arg("E"), arg("reg")=0, arg("well")=0));
        solver.def("getN", &FreeCarrierGainSolver_getN<Geometry2DCylindrical>, (arg("F"), arg("T")=py::object(), arg("reg")=0));
        solver.def("getP", &FreeCarrierGainSolver_getP<Geometry2DCylindrical>, (arg("F"), arg("T")=py::object(), arg("reg")=0));
#endif
//         RW_FIELD(quick_levels,
//                  "Compute levels only once and simply shift for different temperatures?\n\n"
//                  "Setting this to True strongly increases computation speed, but can make the results\n"
//                  "less accurate for high temperatures.");
        solver.def("get_energy_levels", &FreeCarrier_getLevels<Geometry2DCylindrical>, arg("T")=py::object(),
            u8"Get energy levels in quantum wells.\n\n"
            u8"Compute energy levels in quantum wells for electrons, heavy holes and\n"
            u8"light holes.\n\n"
            u8"Args:\n"
            u8"    T (float or ``None``): Temperature to get the levels. If this argument is\n"
            u8"                           ``None``, the estimates for temperature :py:attr:`T0`\n"
            u8"                           are returned.\n\n"
            u8"Returns:\n"
            u8"    list: List with dictionaries with keys `el`, `hh`, and `lh` with levels for\n"
            u8"          electrons, heavy holes and light holes. Each list element corresponds\n"
            u8"          to one active region.\n"
        );
        solver.def("get_fermi_levels", &FreeCarrier_getFermiLevels<Geometry2DCylindrical>, (arg("n"), arg("T")=py::object(), arg("reg")=0),
            u8"Get quasi-Fermi levels.\n\n"
            u8"Compute quasi-Fermi levels in specified active region.\n"
            u8"Args:\n"
            u8"    n (float): Carriers concentration to determine the levels for\n"
            u8"               [1/cm\\ :sup:`3`\\ ].\n"
            u8"    T (float or ``None``): Temperature to get the levels. If this argument is\n"
            u8"                           ``None``, the estimates for temperature :py:attr:`T0`\n"
            u8"                           are returned.\n\n"
            u8"    reg (int): Active region number.\n"
            u8"Returns:\n"
            u8"    tuple: Two-element tuple with quasi-Fermi levels for electrons and holes.\n"
        );
        RW_PROPERTY(T0, getT0, setT0, "Reference temperature.\n\nIn this temperature levels estimates are computed.");
        RW_PROPERTY(matrix_element, getMatrixElem, setMatrixElem,
                    u8"Momentum matrix element.\n\n"
                    u8"Value of the squared matrix element in gain computations. If it is not set it\n"
                    u8"is estimated automatically. (float [eV×m0])");
        RW_PROPERTY(lifetime, getLifeTime, setLifeTime, "Average carriers lifetime.\n\n"
            "This parameter is used for gain spectrum broadening. (float [ps])");
        RW_PROPERTY(strained, getStrained, setStrained,
                    u8"Boolean attribute indicating if the solver should consider strain in the active\n"
                    u8"region.\n\n"
                    u8"If set to ``True`` then there must a layer with the role *substrate* in\n"
                    u8"the geometry. The strain is computed by comparing the atomic lattice constants\n"
                    u8"of the substrate and the quantum wells.");
        RECEIVER(inTemperature, "");
        RECEIVER(inBandEdges, "");
        RECEIVER(inCarriersConcentration, "");
        RECEIVER(inFermiLevels, "");
        PROVIDER(outGain, "");
        PROVIDER(outEnergyLevels, "");
        solver.def("spectrum", &__Class__::getGainSpectrum, py::arg("point"), py::with_custodian_and_ward_postcall<0,1>(),
                   u8"Get gain spectrum at given point.\n\n"
                   u8"Args:\n"
                   u8"    point (vec): Point to get gain at.\n"
                   u8"    c0, c1 (float): Coordinates of the point to get gain at.\n\n"
                   u8"Returns:\n"
                   u8"    :class:`XFermiCyl.Spectrum`: Spectrum object.\n");
        solver.def("spectrum", FreeCarrierGetGainSpectrum2<Geometry2DCylindrical>, (py::arg("c0"), "c1"), py::with_custodian_and_ward_postcall<0,1>());

        py::scope scope = solver;
        py::class_<GainSpectrum<Geometry2DCylindrical>,plask::shared_ptr<GainSpectrum<Geometry2DCylindrical>>, boost::noncopyable>("Spectrum",
            u8"Gain spectrum object. You can call it like a function to get gains for different vavelengths.",
            py::no_init)
            .def("__call__", &FreeCarrierGainSpectrum__call__<Geometry2DCylindrical>, py::arg("lam"),
                 u8"Get gain at specified wavelength.\n\n"
                 u8"Args:\n"
                 u8"    lam (float): Wavelength to get the gain at.\n"
            )
        ;
    }
}


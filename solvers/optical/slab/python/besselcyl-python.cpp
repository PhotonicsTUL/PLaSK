#include "besselcyl-python.h"
#include "slab-python.h"


namespace plask { namespace solvers { namespace slab { namespace python {


std::string BesselSolverCyl_Mode_str(const BesselSolverCyl::Mode& self) {
    dcomplex lam = 2e3*M_PI / self.k0;
    return format("<m: %d, lam: (%.3f%+.3gj) nm, power: %.2g mW>", self.m, lam.real(), lam.imag(), self.power);
}
std::string BesselSolverCyl_Mode_repr(const BesselSolverCyl::Mode& self) {
    return format("BesselCyl.Mode(m=%d, lam=%g, power=%g)", self.m, str(2e3*M_PI / self.k0), self.power);
}

py::object BesselSolverCyl_getDeterminant(py::tuple args, py::dict kwargs) {
    if (py::len(args) != 1)
        throw TypeError("get_determinant() takes exactly one non-keyword argument (%1% given)", py::len(args));
    BesselSolverCyl* self = py::extract<BesselSolverCyl*>(args[0]);

    enum What {
        WHAT_NOTHING = 0,
        WHAT_WAVELENGTH,
        WHAT_K0,
    };
    What what = WHAT_NOTHING;
    py::object array;
    int m = 1;

    boost::optional<dcomplex> lambda;
    py::stl_input_iterator<std::string> begin(kwargs), end;
    for (auto i = begin; i != end; ++i) {
        if (*i == "lam" || *i == "wavelength") {
            if (PyArray_Check(py::object(kwargs[*i]).ptr())) {
                if (what) throw TypeError("Only one key may be an array");
                what = WHAT_WAVELENGTH; array = kwargs[*i];
            } else
                lambda.reset(py::extract<dcomplex>(kwargs[*i]));
        } else if (*i == "k0")
            if (PyArray_Check(py::object(kwargs[*i]).ptr())) {
                if (what) throw TypeError("Only one key may be an array");
                what = WHAT_K0; array = kwargs[*i];
            } else
                lambda.reset(2e3*M_PI / dcomplex(py::extract<dcomplex>(kwargs[*i])));
        else if (*i == "dispersive")
            throw TypeError("Dispersive argument has been removes. Set solver.lam0 attribute.");
        else if (*i == "m")
            m = py::extract<int>(kwargs[*i]);
        else
            throw TypeError("get_determinant() got unexpected keyword argument '%1%'", *i);
    }
    if (lambda) self->setWavelength(*lambda);

    BesselSolverCyl::ParamGuard guard(self);

    self->setM(m);
    
    switch (what) {
        case WHAT_NOTHING:
            return py::object(self->getDeterminant());
        case WHAT_WAVELENGTH:
            return UFUNC<dcomplex>(
                [self](dcomplex x) -> dcomplex { self->setWavelength(x); return self->getDeterminant(); },
                array
            );
        case WHAT_K0:
            return UFUNC<dcomplex>(
                [self](dcomplex x) -> dcomplex { self->setK0(x); return self->getDeterminant(); },
                array
            );
    }
    return py::object();
}

static py::object BesselSolverCyl_getFieldVectorE(BesselSolverCyl& self, int num, double z) {
    if (num < 0) num = self.modes.size() + num;
    if (num >= self.modes.size()) throw IndexError("Bad mode number %d", num);
    return arrayFromVec2D<NPY_CDOUBLE>(self.getFieldVectorE(num, z), false, 2);
}

static py::object BesselSolverCyl_getFieldVectorH(BesselSolverCyl& self, int num, double z) {
    if (num < 0) num = self.modes.size() + num;
    if (num >= self.modes.size()) throw IndexError("Bad mode number %d", num);
    return arrayFromVec2D<NPY_CDOUBLE>(self.getFieldVectorH(num, z), false, 2);
}



void export_BesselSolverCyl()
{
    plask_import_array();

    CLASS(BesselSolverCyl, "BesselCyl",
        "Optical Solver using Bessel expansion in cylindrical coordinates.\n\n"
        "It calculates optical modes and optical field distribution using Bessel slab method\n"
        "and reflection transfer in two-dimensional cylindrical space.")
        export_base(solver);
//     solver.add_property("material_mesh", &__Class__::getMesh, "Regular mesh with points in which material is sampled.");
    PROVIDER(outWavelength, "");
    PROVIDER(outLoss, "");
    METHOD(find_mode, findMode, 
           "Compute the mode near the specified effective index.\n\n"
           "Only one of the following arguments can be given through a keyword.\n"
           "It is the starting point for search of the specified parameter.\n\n"
           "Args:\n"
           "    lam (complex): Wavelength.\n"
           "    k0 (complex): Normalized frequency.\n"
           "    neff (complex): Longitudinal effective index.\n"
           "    ktran (complex): Transverse wavevector.\n",
           "lam", arg("m")=1
          );
    RW_PROPERTY(size, getSize, setSize, "Orthogonal expansion size.");
    solver.def("get_determinant", py::raw_function(BesselSolverCyl_getDeterminant),
               "Compute discontinuity matrix determinant.\n\n"
               "Arguments can be given through keywords only.\n\n"
               "Args:\n"
               "    lam (complex): Wavelength.\n"
               "    k0 (complex): Normalized frequency.\n");
    solver.def("get_electric_coefficients", BesselSolverCyl_getFieldVectorE, (py::arg("num"), "level"),
               "Get Bessel expansion coefficients for the electric field.\n\n"
               "This is a low-level function returning $E_s$ and $E_p$ Bessel expansion\n"
               "coefficients. Please refer to the detailed solver description for their\n"
               "interpretation.\n\n"
               "Args:\n"
               "    num (int): Computed mode number.\n"
               "    level (float): Vertical lever at which the coefficients are computed.\n\n"
               ":rtype: numpy.ndarray\n"
              );
    solver.def("get_magnetic_coefficients", BesselSolverCyl_getFieldVectorH, (py::arg("num"), "level"),
               "Get Bessel expansion coefficients for the magnetic field.\n\n"
               "This is a low-level function returning $H_s$ and $H_p$ Bessel expansion\n"
               "coefficients. Please refer to the detailed solver description for their\n"
               "interpretation.\n\n"
               "Args:\n"
               "    num (int): Computed mode number.\n"
               "    level (float): Vertical lever at which the coefficients are computed.\n\n"
               ":rtype: numpy.ndarray\n"
              );
//     solver.def("compute_reflectivity", &FourierSolver_computeReflectivity<FourierSolver2D>,
//                "Compute reflection coefficient on the perpendicular incidence [%].\n\n"
//                "Args:\n"
//                "    lam (float or array of floats): Incident light wavelength.\n"
//                "    polarization: Specification of the incident light polarization.\n"
//                "        It should be a string of the form 'E\\ *#*\\ ', where *#* is the axis\n"
//                "        name of the non-vanishing electric field component.\n"
//                "    side (`top` or `bottom`): Side of the structure where the incident light is\n"
//                "        present.\n"
//                , (py::arg("lam"), "polarization", "side");
//     solver.def("compute_transmittivity", &FourierSolver_computeTransmittivity<FourierSolver2D>,
//                "Compute transmission coefficient on the perpendicular incidence [%].\n\n"
//                "Args:\n"
//                "    lam (float or array of floats): Incident light wavelength.\n"
//                "    polarization: Specification of the incident light polarization.\n"
//                "        It should be a string of the form 'E\\ *#*\\ ', where *#* is the axis name\n"
//                "        of the non-vanishing electric field component.\n"
//                "    side (`top` or `bottom`): Side of the structure where the incident light is\n"
//                "        present.\n"
//                , (py::arg("lam"), "polarization", "side"));
//     solver.def("compute_reflected_orders", &FourierSolver2D_reflectedAmplitudes,
//                "Compute Fourier coefficients of the reflected field on the perpendicular incidence [-].\n\n"
//                "Args:\n"
//                "    lam (float): Incident light wavelength.\n"
//                "    polarization: Specification of the incident light polarization.\n"
//                "        It should be a string of the form 'E\\ *#*\\ ', where *#* is the axis\n"
//                "        name of the non-vanishing electric field component.\n"
//                "    side (`top` or `bottom`): Side of the structure where the incident light is\n"
//                "        present.\n"
//                , (py::arg("lam"), "polarization", "side"));
//     solver.def("compute_transmitted_orders", &FourierSolver2D_transmittedAmplitudes,
//                "Compute Fourier coefficients of the reflected field on the perpendicular incidence [-].\n\n"
//                "Args:\n"
//                "    lam (float): Incident light wavelength.\n"
//                "    polarization: Specification of the incident light polarization.\n"
//                "        It should be a string of the form 'E\\ *#*\\ ', where *#* is the axis\n"
//                "        name of the non-vanishing electric field component.\n"
//                "    side (`top` or `bottom`): Side of the structure where the incident light is\n"
//                "        present.\n"
//                , (py::arg("lam"), "polarization", "side"));
//     solver.add_property("mirrors", FourierSolver2D_getMirrors, FourierSolver2D_setMirrors,
//                "Mirror reflectivities. If None then they are automatically estimated from the\n"
//                "Fresnel equations.");
    solver.add_property("pml", py::make_function(&Solver_getPML<BesselSolverCyl>, py::with_custodian_and_ward_postcall<0,1>()),
                        &Solver_setPML<BesselSolverCyl>,
                        "Side Perfectly Matched Layers boundary conditions.\n\n"
                        PML_ATTRS_DOC
                       );
    RO_FIELD(modes, "Computed modes.");
//     solver.def("reflected", &FourierSolver_getReflected<FourierSolver2D>, py::with_custodian_and_ward_postcall<0,1>(),
//                "Access to the reflected field.\n\n"
//                "Args:\n"
//                "    lam (float): Incident light wavelength.\n"
//                "    polarization: Specification of the incident light polarization.\n"
//                "        It should be a string of the form 'E\\ *#*\\ ', where *#* is the axis name\n"
//                "        of the non-vanishing electric field component.\n"
//                "    side (`top` or `bottom`): Side of the structure where the incident light is\n"
//                "        present.\n\n"
//                ":rtype: Fourier2D.Reflected\n"
//                , (py::arg("lam"), "polarization", "side"));

#ifndef NDEBUG
    solver.add_property("wavelength", &SlabBase::getWavelength, &Solver_setWavelength<__Class__>, "Wavelength of the light [nm].");
    solver.add_property("k0", &__Class__::getK0, &Solver_setK0<__Class__>, "Normalized frequency of the light [1/µm].");
    solver.add_property("m", &__Class__::getM, &__Class__::setM, "Angular dependence parameter.");
    
    METHOD(epsVmm, epsVmm, "J_{m-1}(gr) \\varepsilon^{-1} J_{m-1}(kr) r dr", "layer");
    METHOD(epsVpp, epsVpp, "J_{m+1}(gr) \\varepsilon^{-1} J_{m+1}(kr) r dr", "layer");
    METHOD(epsTmm, epsTmm, "J_{m-1}(gr) (\\varepsilon_{rr} + \\varepsilon_{\\varphi\\varphi}) J_{m-1}(kr) r dr", "layer");
    METHOD(epsTpp, epsTpp, "J_{m+1}(gr) (\\varepsilon_{rr} + \\varepsilon_{\\varphi\\varphi}) J_{m+1}(kr) r dr", "layer");
    METHOD(epsTmp, epsTmp, "J_{m-1}(gr) (\\varepsilon_{rr} - \\varepsilon_{\\varphi\\varphi}) J_{m+1}(kr) r dr", "layer");
    METHOD(epsTpm, epsTpm, "J_{m+1}(gr) (\\varepsilon_{rr} - \\varepsilon_{\\varphi\\varphi}) J_{m-1}(kr) r dr", "layer");
    METHOD(epsDm, epsDm, "J_{m-1}(gr) d \\varepsilon^{-1}/dr J_m(kr) r dr", "layer");
    METHOD(epsDp, epsDp, "J_{m+1}(gr) d \\varepsilon^{-1}/dr J_m(kr) r dr", "layer");

    METHOD(muVmm, muVmm, "J_{m-1}(gr) \\mu^{-1} J_{m-1}(kr) r dr");
    METHOD(muVpp, muVpp, "J_{m+1}(gr) \\mu^{-1} J_{m+1}(kr) r dr");
    METHOD(muTmm, muTmm, "J_{m-1}(gr) (\\mu_{rr} + \\mu_{\\varphi\\varphi}) J_{m-1}(kr) r dr");
    METHOD(muTpp, muTpp, "J_{m+1}(gr) (\\mu_{rr} + \\mu_{\\varphi\\varphi}) J_{m+1}(kr) r dr");
    METHOD(muTmp, muTmp, "J_{m-1}(gr) (\\mu_{rr} - \\mu_{\\varphi\\varphi}) J_{m+1}(kr) r dr");
    METHOD(muTpm, muTpm, "J_{m+1}(gr) (\\mu_{rr} - \\mu_{\\varphi\\varphi}) J_{m-1}(kr) r dr");
    METHOD(muDm, muDm, "J_{m-1}(gr) d \\mu^{-1}/dr J_m(kr) r dr");
    METHOD(muDp, muDp, "J_{m+1}(gr) d \\mu^{-1}/dr J_m(kr) r dr");
#endif

    py::scope scope = solver;

    register_vector_of<BesselSolverCyl::Mode>("Modes");
    py::class_<BesselSolverCyl::Mode>("Mode", "Detailed information about the mode.", py::no_init)
        .add_property("lam", &getModeWavelength<BesselSolverCyl::Mode>, "Mode wavelength [nm].")
        .add_property("wavelength", &getModeWavelength<BesselSolverCyl::Mode>, "Mode wavelength [nm].")
        .def_readonly("k0", &BesselSolverCyl::Mode::k0, "Mode normalized frequency [1/µm].")
        .def_readonly("m", &BesselSolverCyl::Mode::m, "Angular mode order.")
        .def_readwrite("power", &BesselSolverCyl::Mode::power, "Total power emitted into the mode.")
        .def("__str__", &BesselSolverCyl_Mode_str)
        .def("__repr__", &BesselSolverCyl_Mode_repr)
    ;
}

}}}} // namespace plask::solvers::slab::python
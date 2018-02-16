#include <cmath>
#include <plask/python.hpp>
#include <plask/python_util/ufunc.h>
using namespace plask;
using namespace plask::python;

#include "../simple_optical.h"
using namespace plask::optical::simple_optical;

#define ROOTDIGGER_ATTRS_DOC \
    u8".. rubric:: Attributes:\n\n" \
    u8".. autosummary::\n\n" \
    u8"   ~optical.effective.RootParams.alpha\n" \
    u8"   ~optical.effective.RootParams.lambd\n" \
    u8"   ~optical.effective.RootParams.initial_range\n" \
    u8"   ~optical.effective.RootParams.maxiter\n" \
    u8"   ~optical.effective.RootParams.maxstep\n" \
    u8"   ~optical.effective.RootParams.method\n" \
    u8"   ~optical.effective.RootParams.tolf_max\n" \
    u8"   ~optical.effective.RootParams.tolf_min\n" \
    u8"   ~optical.effective.RootParams.tolx\n\n" \
    u8":rtype: RootParams\n"
    
#define SEARCH_ARGS_DOC \
    u8"    start (complex): Start of the search range (0 means automatic).\n" \
    u8"    end (complex): End of the search range (0 means automatic).\n" \
    u8"    resteps (integer): Number of steps on the real axis during the search.\n" \
    u8"    imsteps (integer): Number of steps on the imaginary axis during the search.\n" \
    u8"    eps (complex): required precision of the search.\n" \

static py::object SimpleOptical_getDeterminant(SimpleOptical& self, py::object val)
{
   return UFUNC<dcomplex>([&](dcomplex x){return self.getVertDeterminant(x);}, val);
}

BOOST_PYTHON_MODULE(simple_optical)
{
    if (!plask_import_array()) throw(py::error_already_set());


    //MD: Klasy solverów w Pythonie muszą mieć końcówkę Cyl, 2D lub 3D — w zależności od tego na jakiej geometrii liczą
    //MD: W zasadzie powinny być dwie, a nawet trzy — dla każdej geometrii osobno (na poziomie C++ można użyć szablonów, by nie pisać tego samego wiele razy)

    {CLASS(SimpleOptical, "SimpleOpticalCyl", "Short solver description.")
     METHOD(findMode, findMode, "This is method to find wavelength of mode", (arg("lam"), arg("m")=0));
     RO_FIELD(stripe_root,
                 u8"Configuration of the root searching algorithm for vertical component of the mode\n"
                 u8"in a single stripe.\n\n"
                 ROOTDIGGER_ATTRS_DOC
                );
     solver.def("get_vert_determinant", &SimpleOptical_getDeterminant, "Get vertical modal determinant for debuging purposes", (arg("wavelength")) );
     PROVIDER(outLightMagnitude, "");
     PROVIDER(outRefractiveIndex, "");
     RW_PROPERTY(vat, getStripeX, setStripeX, u8"Horizontal position of the main stripe (with dominant mode).");
    }

}
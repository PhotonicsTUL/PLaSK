#include <cmath>
#include <plask/python.hpp>
#include <plask/python_util/ufunc.h>
using namespace plask;
using namespace plask::python;

#include "../simple_optical.h"
using namespace plask::optical::simple_optical;


BOOST_PYTHON_MODULE(simple_optical)
{
    if (!plask_import_array()) throw(py::error_already_set());


    //MD: Klasy solverów w Pythonie muszą mieć końcówkę Cyl, 2D lub 3D — w zależności od tego na jakiej geometrii liczą
    //MD: W zasadzie powinny być dwie, a nawet trzy — dla każdej geometrii osobno (na poziomie C++ można użyć szablonów, by nie pisać tego samego wiele razy)

    {CLASS(SimpleOptical, "SimpleOpticalCyl", "Short solver description.")
     METHOD(findMode, findMode, "This is method to find wavelength of mode", (arg("lam"), arg("m")=0));
     METHOD(get_vert_determinant, getVertDeterminant, "Get vertical modal determinant for debuging purposes", (arg("wavelength")) );
     //MD: proszę użyć UFUNC (przykład w funkcji `EffectiveIndex2D_getDeterminant` w pliku solvers/optical/effective/python/effective.cpp
     PROVIDER(outLightMagnitude, "");
     PROVIDER(outRefractiveIndex, "");
     METHOD(getLightMagnitude, getLightMagnitude, "This method return electric field");
     //MD: metoda `getLightMagnitude` nie powinna być eksportowana do Pythona — jest ona wykorzystywana tylko przez provider

     //MD: brakuje atrybutów konfiguracji solvera: parametrów rootdiggera oraz współrzędnej poziomej `x`
    }

}

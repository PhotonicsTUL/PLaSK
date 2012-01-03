#include <boost/python.hpp>
namespace py = boost::python;

#include <config.h>
#include <plask/exceptions.h>
#include <plask/material/material.h>

namespace plask { namespace python {

/**
 * Base class for all the materials derived in Python
 */
struct CustomMaterial : public Material {};

/**
 * Wrapper for CustomMaterial class.
 * For all virtual functions it calls Python derivatives
 */
struct CustomMaterialWrap : public CustomMaterial, py::wrapper<CustomMaterial>
{
    virtual std::string getName() const { return py::extract<std::string>(this->get_override("name")); }
};
/**
 * Function registering custom material class to plask
 * \param name name of the material
 * \param material_class Python class object of the custom material
 */
void registerMaterial(const std::string& name, py::object material_class)
{
    // TODO
}

void initMaterial() {

    py::object materials_module { py::handle<>(py::borrowed(PyImport_AddModule("plask.material"))) };
    py::scope().attr("material") = materials_module;
    py::scope scope = materials_module;

    scope.attr("__doc__") =
        "The material database. Many semiconductor materials used in photonics are defined here. "
        "We have made a significant effort to ensure their physical properties to be the most precise "
        "as the current state of the art. However, you can derive an abstract class plask.material.CustomMaterial "
        "to create your own one."; //TODO maybe more extensive description

    py::class_<Material, shared_ptr<Material>, boost::noncopyable>("Material", "Base class for all materials.", py::no_init)
        .add_property("name", &Material::getName)
        .def("getName", &Material::getName)
    ;

    py::class_<CustomMaterialWrap, shared_ptr<CustomMaterialWrap>, py::bases<Material>, boost::noncopyable>("CustomMaterial");

    def("_registerMaterial", registerMaterial);
}

}} // namespace plask::python

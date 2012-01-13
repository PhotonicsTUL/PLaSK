#include <boost/python.hpp>
namespace py = boost::python;

#include <config.h>
#include <plask/exceptions.h>
#include <plask/material/material.h>

namespace plask { namespace python {

/// Map of all custom Python materials
std::map<std::string, py::object> custom_materials;

// Hack to cheat Boost assertion
struct WrappedMaterial : public Material {};
/**
 * Wrapper for Material class.
 * For all virtual functions it calls Python derivatives
 */
struct MaterialWrap : public WrappedMaterial, py::wrapper<WrappedMaterial>
{
    virtual std::string name() const {
        if (py::override override = this->get_override("name")) return override();
        else return "Material";
    }
};

/**
 * Function constructing custom Python material whre read from XML file
 *
 * \param name plain material name
 * By now the following parameters are ignored
 * \param composition amounts of elements, with NaN for each element for composition was not written
 * \param dopant_amount_type type of amount of dopand, needed to interpretation of @a dopant_amount
 * \param dopant_amount amount of dopand, is ignored if @a dopant_amount_type is @c NO_DOPANT
 */
inline shared_ptr<Material> constructCustomMaterial(const std::string& name, const std::vector<double>& composition,
                                                plask::MaterialsDB::DOPANT_AMOUNT_TYPE dopant_amount_type, double dopant_amount) {
    py::object material = custom_materials[name]();

    //TODO check for constructors with different signatures
    return py::extract<shared_ptr<Material>>(material);
}


/**
 * Function registering custom material class to plask
 * \param name name of the material
 * \param material_class Python class object of the custom material
 */
void registerMaterial(const std::string& name, py::object material_class, MaterialsDB& db)
{
    //TODO issue a warning if material with such name already exists
    custom_materials[name] = material_class;
    db.add(name, &constructCustomMaterial);
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


    py::class_<MaterialsDB, shared_ptr<MaterialsDB>> materialsDB("MaterialsDB", "Material database class"); materialsDB
        .def("get", (shared_ptr<Material> (MaterialsDB::*)(const std::string&, const std::string&) const) &MaterialsDB::get, "Get material of given name and doping")
        .def("get", (shared_ptr<Material> (MaterialsDB::*)(const std::string&) const) &MaterialsDB::get, "Get material of given name and doping")
    ;


    py::class_<Material, shared_ptr<Material>, boost::noncopyable>("Material", "Base class for all materials.", py::no_init)
        .def("name", &Material::name)
    ;

    py::class_<MaterialWrap, shared_ptr<MaterialWrap>, py::bases<Material>, boost::noncopyable>("Material", "Base class for all materials.");

    py::def("registerMaterial", registerMaterial);
}

}} // namespace plask::python

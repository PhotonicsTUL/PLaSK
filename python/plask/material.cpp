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
 * \param doping_amount_type type of amount of dopand, needed to interpretation of @a doping_amount
 * \param doping_amount amount of dopand, is ignored if @a doping_amount_type is @c NO_DOPANT
 */
inline shared_ptr<Material> constructCustomMaterial(const std::string& name, const std::vector<double>& composition,
                                                    MaterialsDB::DOPING_AMOUNT_TYPE doping_amount_type, double doping_amount) {
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


/**
 * \return the list of all materials in database
 */
py::list MaterialsDB_list(const MaterialsDB& DB) {
    py::list materials;
    for (auto material : DB.constructors) {
        //TODO strip doping part from name
        materials.append(material.first);
    }
    return materials;
}

/**
 * \return iterator over registered material names
 */
py::object MaterialsDB_iter(const MaterialsDB& DB) {
    return MaterialsDB_list(DB).attr("__iter__")();
}

/**
 * Create material basing on its name and additional parameters
 *
 * \param DB database to search (self in Python class)
 * \param name generic name of the material (e.g. "AlGaAs")
 * \param args list of concentrations
 * \param kwargs doping in the form: Mg=1e18, n=2e19; maybe also concentrations: Al=0.3
 * \return found material object
 **/
shared_ptr<Material> MaterialsDB_factory(const MaterialsDB& DB, const std::string& name, py::tuple args, py::dict kwargs) {
    auto material = DB.constructors.find(name);
    if (material == DB.constructors.end()) {
        PyErr_SetString(PyExc_KeyError, ("Material " + name + " does not exist in database").c_str());
        throw py::error_already_set();
    }
    //TODO Parse args and kwargs
    return material->second(name, {}, MaterialsDB::NO_DOPING, 0.0);
}

void initMaterial() {

    py::object materials_module { py::handle<>(py::borrowed(PyImport_AddModule("plask.material"))) };
    py::scope().attr("material") = materials_module;
    py::scope scope = materials_module;

    scope.attr("__doc__") =
        "The material database. Many semiconductor materials used in photonics are defined here. "
        "We have made a significant effort to ensure their physical properties to be the most precise "
        "as the current state of the art. However, you can derive an abstract class plask.material.Material "
        "to create your own materials."; //TODO maybe more extensive description


    py::class_<MaterialsDB, shared_ptr<MaterialsDB>> materialsDB("MaterialsDB", "Material database class"); materialsDB
        .def("get", (shared_ptr<Material> (MaterialsDB::*)(const std::string&, const std::string&) const) &MaterialsDB::get, "Get material of given name and doping")
        .def("get", (shared_ptr<Material> (MaterialsDB::*)(const std::string&) const) &MaterialsDB::get, "Get material of given name and doping")
        .add_property("materials", &MaterialsDB_list, "Return the list of all materials in database")
        .def("factory", &MaterialsDB_factory, "Return material based on its generic name and parameters passes in args and kwargs")
        .def("__iter__", &MaterialsDB_iter)
    ;


    // Common material interface
    py::class_<Material, shared_ptr<Material>, boost::noncopyable>("Material", "Base class for all materials.", py::no_init)
        .def("name", &Material::name)
    ;


    // Wrapper and registration for custom materials
    py::class_<MaterialWrap, shared_ptr<MaterialWrap>, py::bases<Material>, boost::noncopyable>("Material", "Base class for all materials.");
    py::def("registerMaterial", registerMaterial);
}

}} // namespace plask::python

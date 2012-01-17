#include <boost/python.hpp>
namespace py = boost::python;

#include <config.h>
#include <plask/utils/string.h>
#include <plask/exceptions.h>
#include <plask/material/material.h>

namespace plask { namespace python {

/**
 * Wrapper for Material class.
 * For all virtual functions it calls Python derivatives
 */
struct MaterialWrap : public Material
{
    PyObject* self;

    MaterialWrap () : self(self) { std::cerr << "MaterialWrap::MaterialWrap(self)\n"; }

    ~MaterialWrap() { /*py::decref(self);*/ std::cerr << "MaterialWrap::~MaterialWrap\n"; }

    virtual std::string name() const {
        return py::extract<std::string>(py::object(py::detail::borrowed_reference(self)).attr("name"));
    }

    virtual double VBO(double T) const {
        return py::call_method<double>(self, "VBO", T);
    }

    static shared_ptr<Material> __init__() {
        MaterialWrap* ptr = new MaterialWrap();
        auto sptr = shared_ptr<Material>(ptr);
        auto obj = py::object(sptr);
        ptr->self = obj.ptr();
        return sptr;
    }

};


/**
 * Object constructing custom Python material whre read from XML file
 *
 * \param name plain material name
 *
 * Other parameters are ignored
 */
class PythonMaterialConstructor : public MaterialsDB::MaterialConstructor
{
    py::object material_class;

  public:
    PythonMaterialConstructor(py::object material_class) : material_class(material_class) {}

    inline shared_ptr<Material> operator()(const std::vector<double>& composition, MaterialsDB::DOPING_AMOUNT_TYPE doping_amount_type, double doping_amount) const
    {

        // We pass composition parameters as *args to constructor
        py::list args;
        bool all_nan = true;
        for (auto i = composition.begin(); i != composition.end(); ++i) {
            if (!isnan(*i)) { all_nan = false; break; }
        }
        if (!all_nan) {
            for (auto i = composition.begin(); i != composition.end(); ++i) {
                args.append(*i);
            }
        }

        // We pass doping information in **kwargs
        py::dict kwargs;
        if (doping_amount_type !=  MaterialsDB::NO_DOPING) {
            kwargs[ doping_amount_type == MaterialsDB::DOPANT_CONCENTRATION ? "dc" : "cc" ] = doping_amount;
        }

        py::object material = material_class(*py::tuple(args), **kwargs);

        MaterialWrap* ptr = dynamic_cast<MaterialWrap*>((Material*)py::extract<Material*>(material));
        ptr->self = py::incref(material.ptr());
        return shared_ptr<Material>(ptr);
    }
};



/**
 * Function registering custom material class to plask
 * \param name name of the material
 * \param material_class Python class object of the custom material
 */
void registerMaterial(const std::string& name, py::object material_class, shared_ptr<MaterialsDB> db)
{
    //TODO issue a warning if material with such name already exists
    PythonMaterialConstructor* constructor = new PythonMaterialConstructor(material_class);
    db->add(name, constructor);
}

/**
 * \return the list of all materials in database
 */
py::list MaterialsDB_list(const MaterialsDB& DB) {
    py::list materials;
    for (auto material = DB.constructors.begin(); material != DB.constructors.end(); ++material )
        materials.append(material->first);
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

    //TODO Parse args and kwargs

    return DB.get(name, {}, MaterialsDB::NO_DOPING, 0.0);
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

    py::class_<MaterialsDB, shared_ptr<MaterialsDB>, boost::noncopyable> materialsDB("MaterialsDB", "Material database class"); materialsDB
        .def("get", (shared_ptr<Material> (MaterialsDB::*)(const std::string&, const std::string&) const) &MaterialsDB::get, "Get material of given name and doping")
        .def("get", (shared_ptr<Material> (MaterialsDB::*)(const std::string&) const) &MaterialsDB::get, "Get material of given name and doping")
        .add_property("materials", &MaterialsDB_list, "Return the list of all materials in database")
        .def("factory", &MaterialsDB_factory, "Return material based on its generic name and parameters passes in args and kwargs")
        .def("__iter__", &MaterialsDB_iter)
    ;

    // Common material interface
    py::class_<Material, shared_ptr<Material>, boost::noncopyable>("Material", "Base class for all materials.", py::no_init)
        .def("__init__", py::make_constructor(&MaterialWrap::__init__))
        .add_property("name", &Material::name)
        .def("VBO", &Material::VBO)
    ;

    py::def("registerMaterial", &registerMaterial);
}

}} // namespace plask::python

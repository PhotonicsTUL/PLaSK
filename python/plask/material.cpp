#include "globals.h"
#include <boost/python/raw_function.hpp>
#include <boost/python/stl_iterator.hpp>

#include <config.h>
#include <plask/utils/string.h>
#include <plask/exceptions.h>
#include <plask/material/material.h>

#include "../util/raw_constructor.h"


namespace plask { namespace python {

/**
 * Wrapper for Material class.
 * For all virtual functions it calls Python derivatives
 */
class MaterialWrap : public Material
{
    template <typename T>
    inline T attr(const char* attribute) const {
        return py::extract<T>(py::object(py::detail::borrowed_reference(self)).attr(attribute));
    }

    bool overriden(char const* name) const
    {
        py::converter::registration const& r = py::converter::registered<Material>::converters;
        PyTypeObject* class_object = r.get_class_object();
        if (self)
        {
            if (py::handle<> m = py::handle<>(py::allow_null(::PyObject_GetAttrString(self, const_cast<char*>(name))))) {
                PyObject* borrowed_f = 0;
                if (PyMethod_Check(m.get()) && ((PyMethodObject*)m.get())->im_self == self && class_object->tp_dict != 0)
                    borrowed_f = PyDict_GetItemString(class_object->tp_dict, const_cast<char*>(name));
                if (borrowed_f != ((PyMethodObject*)m.get())->im_func) return true;
            }
        }
        return false;
    }

    struct EmptyBase : public Material {
        virtual std::string name() const { return ""; }
    };

    shared_ptr<Material> base;
    PyObject* self;

  public:
    MaterialWrap () : base(new EmptyBase) {}
    MaterialWrap (shared_ptr<Material> base) : base(base) {}

    virtual std::string name() const {
        return attr<std::string>("name");
    }

    virtual double VBO(double T) const {
        if (overriden("VBO")) return py::call_method<double>(self, "VBO", T);
        return base->VBO(T);
    }

    static shared_ptr<Material> __init__(py::tuple args, py::dict kwargs) {
        if (py::len(args) > 2 || py::len(kwargs) != 0) {
            PyErr_SetString(PyExc_TypeError, "wrong number of arguments");
            throw py::error_already_set();
        }
        MaterialWrap* ptr;
        if (py::len(args) == 2) {
            shared_ptr<Material> base = py::extract<shared_ptr<Material>>(args[1]);
            ptr = new MaterialWrap(base);
        } else {
            ptr = new MaterialWrap();
        }
        auto sptr = shared_ptr<Material>(ptr);
        ptr->self = py::object(args[0]).ptr();  // key line !!!
        return sptr;
    }

};


/**
 * Object constructing custom Python material where read from XML file
 *
 * \param name plain material name
 *
 * Other parameters are ignored
 */
class PythonMaterialConstructor : public MaterialsDB::MaterialConstructor
{
    py::object material_class;
    std::string dopant;

  public:
    PythonMaterialConstructor(py::object material_class, std::string dope="") : material_class(material_class), dopant(dope) {}

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
            kwargs["dope"] = dopant;
            kwargs[ doping_amount_type == MaterialsDB::DOPANT_CONCENTRATION ? "dc" : "cc" ] = doping_amount;
        }

        py::object material = material_class(*py::tuple(args), **kwargs);

        return py::extract<shared_ptr<Material>>(material);
    }
};



/**
 * Function registering custom material class to plask
 * \param name name of the material
 * \param material_class Python class object of the custom material
 */
void registerMaterial(const std::string& name, py::object material_class, shared_ptr<MaterialsDB> db)
{
    std::string dopant = std::get<1>(splitString2(name, ':'));
    db->add(name, new PythonMaterialConstructor(material_class, dopant));
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
 **/
shared_ptr<Material> MaterialsDB_get(py::tuple args, py::dict kwargs) {

    if (py::len(args) != 2) {
        PyErr_SetString(PyExc_ValueError, "MaterialsDB.get(self, name, **kwargs) takes exactly two non-keyword arguments");
        throw py::error_already_set();
    }

    const MaterialsDB* DB = py::extract<MaterialsDB*>(args[0]);
    std::string name = py::extract<std::string>(args[1]);

    // Test if we have just a name string
    if (py::len(kwargs) == 0) return DB->get(name);

    // Otherwise parse other args

    // Get element names and check for compositions in kwargs
    std::vector<std::string> elements;
    std::string element;
    int l = name.length();
    for (int i = 0; i < l; ++i) {
        char c = name[i];
        if (c < 'a' || c > 'z') {
            if (element != "") elements.push_back(element);
            element = "";
        }
        element += c;
    }
    elements.push_back(element);
    std::vector<double> composition;
    for (auto element = elements.begin(); element != elements.end(); ++element) {
        double c;
        try {
            c = py::extract<double>(kwargs[*element]);
        } catch (py::error_already_set) {
            c = nan("");
            PyErr_Clear();
        }
        composition.push_back(c);
    }

    // Get doping
    bool doping = false;
    try {
        std::string dopant = py::extract<std::string>(kwargs["dope"]);
        name = name + ":" + dopant;
        doping = true;
    } catch (py::error_already_set) {
        PyErr_Clear();
    }
    MaterialsDB::DOPING_AMOUNT_TYPE doping_type = MaterialsDB::NO_DOPING;
    double concentation = 0;

    if (doping) {
        py::object cobj;
        bool has_dc = false;
        try {
            cobj = kwargs["dc"];
            doping_type = MaterialsDB::DOPANT_CONCENTRATION;
            has_dc = true;
        } catch (py::error_already_set) {
            PyErr_Clear();
        }
        try {
            cobj = kwargs["cc"];
            doping_type = MaterialsDB::CARRIER_CONCENTRATION;
        } catch (py::error_already_set) {
            PyErr_Clear();
        }
        if (doping_type == MaterialsDB::NO_DOPING) {
            PyErr_SetString(PyExc_ValueError, "neither dopant nor carrier concentrations specified");
            throw py::error_already_set();
        } else if (doping_type == MaterialsDB::CARRIER_CONCENTRATION && has_dc) {
            PyErr_SetString(PyExc_ValueError, "dopant and carrier concentrations specified simultanously");
            throw py::error_already_set();
        }

        concentation = py::extract<double>(cobj);
    }

    return DB->get(name, composition, doping_type, concentation);
}

py::list Material__completeComposition(py::object src, unsigned int pattern) {
    std::vector<double> in;
    py::stl_input_iterator<double> begin(src), end;
    for (auto i = begin; i != end; ++i) in.push_back(*i);
    std::vector<double> out = Material::completeComposition(in, pattern);
    py::list dst;
    for (auto i = out.begin(); i != out.end(); ++i) dst.append(*i);
    return dst;
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
        .def("get", py::raw_function(&MaterialsDB_get), "Get material of given name and doping")
        .add_property("materials", &MaterialsDB_list, "Return the list of all materials in database")
        .def("__iter__", &MaterialsDB_iter)
    ;

    // Common material interface
    py::class_<Material, shared_ptr<Material>, boost::noncopyable>("Material", "Base class for all materials.", py::no_init)
        .def("__init__", raw_constructor(&MaterialWrap::__init__))
        .def("_completeComposition", &Material__completeComposition, "Fix incomplete material composition basing on patten")
        .staticmethod("_completeComposition")
        .add_property("name", &Material::name)
        .def("VBO", &Material::VBO)
    ;

    py::def("registerMaterial", &registerMaterial);
}

}} // namespace plask::python

#include "python_globals.h"
#include <boost/python/raw_function.hpp>
#include <boost/python/stl_iterator.hpp>
#include <algorithm>

#include <plask/config.h>
#include <plask/utils/string.h>
#include <plask/exceptions.h>
#include <plask/material/db.h>

#include "../util/raw_constructor.h"

namespace plask { namespace python {

typedef std::pair<double,double> DDPair;
typedef std::tuple<dcomplex,dcomplex,dcomplex,dcomplex,dcomplex> NrTensorT;

namespace detail {
    struct DDpair_fromto_Python
    {
        DDpair_fromto_Python() {
            boost::python::converter::registry::push_back(&convertible, &construct, boost::python::type_id<DDPair>());
            boost::python::to_python_converter<DDPair, DDpair_fromto_Python>();
        }

        static void* convertible(PyObject* obj) {
            if (!PySequence_Check(obj) && !PyFloat_Check(obj) && !PyInt_Check(obj)) return NULL;
            return obj;
        }

        static void construct(PyObject* obj, boost::python::converter::rvalue_from_python_stage1_data* data) {
            void* storage = ((boost::python::converter::rvalue_from_python_storage<DDPair>*)data)->storage.bytes;
            double first, second;
            if (PyFloat_Check(obj) || PyInt_Check(obj)) {
                first = second = py::extract<double>(obj);
            } else if (PySequence_Length(obj) == 2) {
                auto src = py::object(py::handle<>(py::borrowed(obj)));
                auto ofirst = src[0];
                auto osecond = src[1];
                first = py::extract<double>(ofirst);
                second = py::extract<double>(osecond);
            } else {
                throw TypeError("float or sequence of exactly two floats required");
            }
            new(storage) DDPair(first, second);
            data->convertible = storage;
        }

        static PyObject* convert(const DDPair& pair)  {
            py::tuple tuple = py::make_tuple(pair.first, pair.second);
            return boost::python::incref(tuple.ptr());
        }
    };


    struct ComplexTensor_fromto_Python
    {
        ComplexTensor_fromto_Python() {
            boost::python::converter::registry::push_back(&convertible, &construct, boost::python::type_id<NrTensorT>());
            boost::python::to_python_converter<NrTensorT, ComplexTensor_fromto_Python>();
        }

        static void* convertible(PyObject* obj) {
            if (!PySequence_Check(obj)) return NULL;
            return obj;
        }

        static void construct(PyObject* obj, boost::python::converter::rvalue_from_python_stage1_data* data) {
            py::object src = py::object(py::handle<>(py::borrowed(obj)));
            void* storage = ((boost::python::converter::rvalue_from_python_storage<NrTensorT>*)data)->storage.bytes;
            dcomplex vals[5];
            int idx[5] = { 0, 1, 2, 3, 4 };
            auto seq = py::object(py::handle<>(py::borrowed(obj)));
            if (py::len(seq) == 2) { idx[1] = 0; idx[2] = 1; idx[3] = idx[4] = -1; }
            else if (py::len(seq) == 3) { idx[3] = idx[4] = -1; }
            else if (py::len(seq) != 5)
                throw TypeError("sequence of exactly 2, 3, or 5 complex required");
            for (int i = 0; i < 5; ++i) {
                if (idx[i] != -1)  vals[i] = py::extract<dcomplex>(seq[idx[i]]);
                else vals[i] = 0.;
            }
            new(storage) NrTensorT(vals[0], vals[1], vals[2], vals[3], vals[4]);
            data->convertible = storage;
        }

        static PyObject* convert(const NrTensorT& src)  {
            py::tuple tuple = py::make_tuple(std::get<0>(src), std::get<1>(src), std::get<2>(src), std::get<3>(src), std::get<4>(src));
            return boost::python::incref(tuple.ptr());
        }
    };
}

/**
 * Wrapper for Material class.
 * For all virtual functions it calls Python derivatives
 */
class PythonMaterial : public Material
{
    shared_ptr<Material> base;
    PyObject* self;

    template <typename T>
    inline T attr(const char* attribute) const {
        return py::extract<T>(py::object(py::detail::borrowed_reference(self)).attr(attribute));
    }

    PyMethodObject* overriden(char const* name) const
    {
        py::converter::registration const& r = py::converter::registered<Material>::converters;
        PyTypeObject* class_object = r.get_class_object();
        if (self) {
            if (PyObject* mo = PyObject_GetAttrString(self, const_cast<char*>(name))) {
                if (PyMethod_Check(mo)) {
                    PyMethodObject* m = (PyMethodObject*)mo;
                    PyObject* borrowed_f = nullptr;
                    if(m->im_self == self && class_object->tp_dict != 0)
                        borrowed_f = PyDict_GetItemString(class_object->tp_dict, const_cast<char*>(name));
                    if (borrowed_f != m->im_func) return m;
                }
            }
        }
        return nullptr;
    }

    template <typename R, typename F, typename... Args>
    inline R override(const char* name, F f, Args... args) const {
        if (overriden(name)) return py::call_method<R>(self, name, args...);
        return ((*base).*f)(args...);
    }

    DDPair call_thermk(double T, double t) const {
        PyMethodObject* m = overriden("thermk");
        if (m) {
            if(PyObject* fc = PyObject_GetAttrString(m->im_func, "func_code")) {
                if(PyObject* ac = PyObject_GetAttrString(fc, "co_argcount")) {
                    const int count = PyInt_AsLong(ac);
                    if (count == 2) return py::call_method<DDPair>(self, "thermk", T);
                    else if (count == 3) return py::call_method<DDPair>(self, "thermk", T, t);
                    else if (count < 2) throw TypeError("thermk() takes at least 2 arguments (%1%) given", count);
                    else throw TypeError("thermk() takes at most 3 arguments (%1%) given", count);
                    Py_DECREF(ac);
                }
                Py_DECREF(fc);
            }

        }
        return base->thermk(T);
    }

  public:
    PythonMaterial () : base(new EmptyMaterial) {}
    PythonMaterial (shared_ptr<Material> base) : base(base) {}

    static shared_ptr<Material> __init__(py::tuple args, py::dict kwargs) {
        if (py::len(args) > 2 || py::len(kwargs) != 0) {
            throw TypeError("wrong number of arguments");
        }
        PythonMaterial* ptr;
        int len = py::len(args);
        if (len == 2) {
            shared_ptr<Material> base = py::extract<shared_ptr<Material>>(args[1]);
            ptr = new PythonMaterial(base);
        } else if (len == 1) {
            ptr = new PythonMaterial();
        } else {
            throw TypeError("__init__ takes at most 2 arguments (%d given)", len);
        }
        auto sptr = shared_ptr<Material>(ptr);
        ptr->self = py::object(args[0]).ptr();  // key line !!!
        return sptr;
    }


    // Here there are overridden methods from Material class

    virtual std::string name() const {
        py::object cls = py::object(py::detail::borrowed_reference(self)).attr("__class__");
        py::object oname;
        try {
            oname = cls.attr("__dict__")["name"];
        } catch (py::error_already_set) {
            PyErr_Clear();
            oname = cls.attr("__name__");
        }
        return py::extract<std::string>(oname);
    }

    virtual std::string str() const {
        if (overriden("__str__")) return py::call_method<std::string>(self, "__str__");
        else return name();
    }

    virtual Material::ConductivityType condtype() const {
        py::object cls = py::object(py::detail::borrowed_reference(self)).attr("__class__");
        py::object octype;
        try {
            octype = cls.attr("__dict__")["condtype"];
        } catch (py::error_already_set) {
            PyErr_Clear();
            return base->condtype();
        }
        return py::extract<Material::ConductivityType>(octype);
    }

    virtual Material::Kind kind() const { return attr<Material::Kind>("kind"); }
    virtual double lattC(double T, char x) const { return override<double>("lattC", &Material::lattC, T, x); }
    virtual double Eg(double T, const char Point) const { return override<double>("Eg", &Material::Eg, T, Point); }
    virtual double CBO(double T, const char Point) const { return override<double>("CBO", &Material::CBO, T, Point); }
    virtual double VBO(double T) const { return override<double>("VBO", &Material::VBO, T); }
    virtual double Dso(double T) const { return override<double>("Dso", &Material::Dso, T); }
    virtual double Mso(double T) const { return override<double>("Mso", &Material::Mso, T); }
    virtual DDPair Me(double T, const char Point) const { return override<DDPair>("Me", &Material::Me, T, Point); }
    virtual DDPair Mhh(double T, const char Point) const { return override<DDPair>("Mhh", &Material::Mhh, T, Point); }
    virtual DDPair Mlh(double T, const char Point) const { return override<DDPair>("Mlh", &Material::Mlh, T, Point); }
    virtual DDPair Mh(double T, char EqType) const { return override<DDPair>("Mh", &Material::Mh, T, EqType); }
    virtual double ac(double T) const { return override<double>("ac", &Material::Mso, T); }
    virtual double av(double T) const { return override<double>("av", &Material::Mso, T); }
    virtual double b(double T) const { return override<double>("b", &Material::Mso, T); }
    virtual double c11(double T) const { return override<double>("c11", &Material::Mso, T); }
    virtual double c12(double T) const {return override<double>("c12", &Material::Mso, T); }
    virtual double eps(double T) const { return override<double>("eps", &Material::eps, T); }
    virtual double chi(double T, const char Point) const { return override<double>("chi", (double (Material::*)(double, char) const) &Material::chi, T, Point); }
    virtual double Nc(double T, const char Point) const { return override<double>("Nc", &Material::Nc, T, Point); }
    virtual double Nv(double T) const { return override<double>("Nv", &Material::Nv, T); }
    virtual double Ni(double T) const { return override<double>("Ni", &Material::Ni, T); }
    virtual double Nf(double T) const { return override<double>("Nf", &Material::Nf, T); }
    virtual double EactD(double T) const { return override<double>("EactD", &Material::EactD, T); }
    virtual double EactA(double T) const { return override<double>("EactA", &Material::EactA, T); }
    virtual DDPair mob(double T) const { return override<DDPair>("mob", &Material::mob, T); }
    virtual DDPair cond(double T) const { return override<DDPair>("cond", &Material::cond, T); }
    virtual double A(double T) const { return override<double>("A", &Material::A, T); }
    virtual double B(double T) const { return override<double>("B", &Material::B, T); }
    virtual double C(double T) const { return override<double>("C", &Material::C, T); }
    virtual double D(double T) const { return override<double>("D", &Material::D, T); }
    virtual DDPair thermk(double T) const { return call_thermk(T, INFINITY); }
    virtual DDPair thermk(double T, double t) const { return call_thermk(T, t); }
    virtual double dens(double T) const { return override<double>("dens", &Material::dens, T); }
    virtual double cp(double T) const { return override<double>("cp", &Material::cp, T); }
    virtual double nr(double wl, double T) const { return override<double>("nr", &Material::nr, wl, T); }
    virtual double absp(double wl, double T) const { return override<double>("absp", &Material::absp, wl, T); }
    virtual dcomplex nR(double wl, double T) const {
        if (overriden("nR")) return py::call_method<dcomplex>(self, "nR", wl, T);
        return dcomplex(override<double>("nr", &Material::nr, wl, T), -7.95774715459e-09*override<double>("absp", &Material::absp, wl,T)*wl);
    }
    virtual NrTensorT nR_tensor(double wl, double T) const {
        if (overriden("nR_tensor")) return py::call_method<NrTensorT>(self, "nR_tensor", wl, T);
        dcomplex n (override<double>("nr", &Material::nr, wl, T), -7.95774715459e-09*override<double>("absp", &Material::absp, wl,T)*wl);
        return NrTensorT(n, n, n, 0., 0.);
    }

    // End of overriden methods

};


/**
 * Object constructing custom simple Python material when read from XML file
 *
 * \param name plain material name
 *
 * Other parameters are ignored
 */
class PythonSimpleMaterialConstructor: public MaterialsDB::MaterialConstructor
{
    py::object material_class;
    std::string dopant;

  public:
    PythonSimpleMaterialConstructor(const std::string& name, py::object material_class, std::string dope="") :
        MaterialsDB::MaterialConstructor(name), material_class(material_class), dopant(dope) {}

    inline shared_ptr<Material> operator()(const Material::Composition& composition, Material::DopingAmountType doping_amount_type, double doping_amount) const
    {
        py::tuple args;
        py::dict kwargs;
        // Doping information
        if (doping_amount_type !=  Material::NO_DOPING) {
            kwargs["dp"] = dopant;
            kwargs[ doping_amount_type == Material::DOPANT_CONCENTRATION ? "dc" : "cc" ] = doping_amount;
        }
        return py::extract<shared_ptr<Material>>(material_class(*args, **kwargs));
    }
};

/**
 * Object constructing custom complex Python material whene read from XML file
 *
 * \param name plain material name
 *
 * Other parameters are ignored
 */
class PythonComplexMaterialConstructor : public MaterialsDB::MaterialConstructor
{
    py::object material_class;
    std::string dopant;

  public:
    PythonComplexMaterialConstructor(const std::string& name, py::object material_class, std::string dope="") :
        MaterialsDB::MaterialConstructor(name), material_class(material_class), dopant(dope) {}

    inline shared_ptr<Material> operator()(const Material::Composition& composition, Material::DopingAmountType doping_amount_type, double doping_amount) const
    {
        py::dict kwargs;
        // Composition
        for (auto c : composition) kwargs[c.first] = c.second;
        // Doping information
        if (doping_amount_type !=  Material::NO_DOPING) {
            kwargs["dp"] = dopant;
            kwargs[ doping_amount_type == Material::DOPANT_CONCENTRATION ? "dc" : "cc" ] = doping_amount;
        }

        py::tuple args;
        py::object material = material_class(*args, **kwargs);

        return py::extract<shared_ptr<Material>>(material);
    }
};



/**
 * Function registering custom simple material class to plask
 * \param name name of the material
 * \param material_class Python class object of the custom material
 */
void registerSimpleMaterial(const std::string& name, py::object material_class, MaterialsDB& db)
{
    std::string dopant = splitString2(name, ':').second;
    db.addSimple(make_shared<PythonSimpleMaterialConstructor>(name, material_class, dopant));
}

/**
 * Function registering custom complex material class to plask
 * \param name name of the material
 * \param material_class Python class object of the custom material
 */
void registerComplexMaterial(const std::string& name, py::object material_class, MaterialsDB& db)
{
    std::string dopant = splitString2(name, ':').second;
    db.addComplex(make_shared<PythonComplexMaterialConstructor>(name, material_class, dopant));
}

/**
 * \return the list of all materials in database
 */
py::list MaterialsDB_list(const MaterialsDB& DB) {
    py::list materials;
    for (auto material: DB) materials.append(material->materialName);
    return materials;
}

/**
 * \return iterator over registered material names
 */
py::object MaterialsDB_iter(const MaterialsDB& DB) {
    return MaterialsDB_list(DB).attr("__iter__")();
}

/**
 * \return true is the material with given name is in the database
 */
bool MaterialsDB_contains(const MaterialsDB& DB, const std::string& name) {
    for (auto material: DB) if (material->materialName == name) return true;
    return false;
}


/**
 * Create material basing on its name and additional parameters
 **/
shared_ptr<Material> MaterialsDB_get(py::tuple args, py::dict kwargs) {

    if (py::len(args) != 2) {
        throw ValueError("MaterialsDB.get(self, name, **kwargs) takes exactly two non-keyword arguments");
    }

    const MaterialsDB* DB = py::extract<MaterialsDB*>(args[0]);
    std::string name = py::extract<std::string>(args[1]);

    // Test if we have just a name string
    if (py::len(kwargs) == 0) return DB->get(name);

    // Otherwise parse other args

    // Get doping
    bool doping = false;
    std::string dopant = "";
    int doping_keys = 0;
    try {
        dopant = py::extract<std::string>(kwargs["dp"]);
        doping = true;
        ++doping_keys;
    } catch (py::error_already_set) {
        PyErr_Clear();
    }
    Material::DopingAmountType doping_type = Material::NO_DOPING;
    double concentration = 0;
    py::object cobj;
    bool has_dc = false;
    try {
        cobj = kwargs["dc"];
        doping_type = Material::DOPANT_CONCENTRATION;
        has_dc = true;
        ++doping_keys;
    } catch (py::error_already_set) {
        PyErr_Clear();
    }
    try {
        cobj = kwargs["cc"];
        doping_type = Material::CARRIER_CONCENTRATION;
        ++doping_keys;
    } catch (py::error_already_set) {
        PyErr_Clear();
    }
    if (doping_type == Material::CARRIER_CONCENTRATION && has_dc) {
        throw ValueError("doping and carrier concentrations specified simultanously");
    }
    if (doping) {
        if (doping_type == Material::NO_DOPING) {
            throw ValueError("dopant specified, but neither doping nor carrier concentrations given correctly");
        } else {
            concentration = py::extract<double>(cobj);
        }
    } else {
        if (doping_type != Material::NO_DOPING) {
            throw ValueError("%s concentration given, but no dopant specified", has_dc?"doping":"carrier");
        }
    }

    std::size_t sep = name.find(':');
    if (sep != std::string::npos) {
        if (doping) {
            throw ValueError("doping specified in **kwargs, but name contains ':'");
        } else {
           Material::parseDopant(name.substr(sep+1), dopant, doping_type, concentration);
        }
    }

    py::list keys = kwargs.keys();

    // Test if kwargs contains only doping information
    if (py::len(keys) == doping_keys) {
        std::string full_name = name; full_name += ":"; full_name += dopant;
        return DB->get(full_name, std::vector<double>(), doping_type, concentration);
    }

    // So, kwargs contains compostion
    std::vector<std::string> objects = Material::parseObjectsNames(name);
    py::object none;
    // test if only correct objects are given
    for (int i = 0; i < py::len(keys); ++i) {
        std::string k = py::extract<std::string>(keys[i]);
        if (k != "dp" && k != "dc" && k != "cc" && std::find(objects.begin(), objects.end(), k) == objects.end()) {
            throw TypeError("'%s' not allowed in material %s", k, name);
        }
    }
    // make composition map
    Material::Composition composition;
    for (auto e: objects) {
        py::object v;
        try {
            v = kwargs[e];
        } catch (py::error_already_set) {
            PyErr_Clear();
        }
        composition[e] = (v != none) ? py::extract<double>(v): std::numeric_limits<double>::quiet_NaN();
    }

    return DB->get(composition, dopant, doping_type, concentration);
}

/**
 * Converter for Python string to material using default database.
 * Allows to create geometry objects as \c rectange(2,1,"GaAs")
 */
struct Material_from_Python_string {

    Material_from_Python_string() {
        boost::python::converter::registry::push_back(&convertible, &construct, boost::python::type_id<shared_ptr<Material>>());
    }

    // Determine if obj can be converted into Material
    static void* convertible(PyObject* obj) {
        if (!PyString_Check(obj)) return 0;
        return obj;
    }

    static void construct(PyObject* obj, boost::python::converter::rvalue_from_python_stage1_data* data) {
        std::string value = PyString_AsString(obj);

        // Grab pointer to memory into which to construct the new Material
        void* storage = ((boost::python::converter::rvalue_from_python_storage<shared_ptr<Material>>*)data)->storage.bytes;

        new(storage) shared_ptr<Material>(MaterialsDB::getDefault().get(value));

        // Stash the memory chunk pointer for later use by boost.python
        data->convertible = storage;
    }
};


py::dict Material__completeComposition(py::dict src, std::string name) {
    py::list keys = src.keys();
    Material::Composition comp;
    py::object none;
    for(int i = 0; i < py::len(keys); ++i) {
        std::string k = py::extract<std::string>(keys[i]);
        if (k != "dp" && k != "dc" && k != "cc") {
            py::object s = src[keys[i]];
            comp[py::extract<std::string>(keys[i])] = (s != none) ? py::extract<double>(s): std::numeric_limits<double>::quiet_NaN();
        }
    }
    if (name != "") {
        std::string basename = splitString2(name, ':').first;
        std::vector<std::string> objects = Material::parseObjectsNames(basename);
        for (auto c: comp) {
            if (std::find(objects.begin(), objects.end(), c.first) == objects.end()) {
                throw TypeError("'%s' not allowed in material %s", c.first, name);
            }
        }
    }

    comp = Material::completeComposition(comp);

    py::dict result;
    for (auto c: comp) result[c.first] = c.second;
    return result;
}


std::string Material__str__(const Material& self) {
    return self.str();
}

std::string Material__repr__(const Material& self) {
    return format("plask.materials.Material('%1%')", Material__str__(self));
}

// Exception translators

void initMaterials() {

    py::object materials_module { py::handle<>(py::borrowed(PyImport_AddModule("plask.material"))) };
    py::scope().attr("material") = materials_module;
    py::scope scope = materials_module;

    py::class_<MaterialsDB, shared_ptr<MaterialsDB>/*, boost::noncopyable*/> materialsDB("MaterialsDB",
        "Material database class\n\n"
        "    The material database. Many semiconductor materials used in photonics are defined here.\n"
        "    We have made a significant effort to ensure their physical properties to be the most precise\n"
        "    as the current state of the art. However, you can derive an abstract class plask.Material\n"
        "    to create your own materials.\n" //TODO maybe more extensive description
        ); materialsDB
        .def("get_default", &MaterialsDB::getDefault, "Get default database", py::return_value_policy<py::reference_existing_object>())
        .staticmethod("get_default")
        .def("load", &MaterialsDB::loadToDefault, "Load materials from library lib to default database", py::arg("lib"))
        .staticmethod("load") // TODO make it non-static
        .def("load_all", &MaterialsDB::loadAllToDefault, "Load all materials from specified directory to default database", py::arg("directory")=plaskMaterialsPath())
        .staticmethod("load_all") // TODO make it non-static
        .def("get", py::raw_function(&MaterialsDB_get), "Get material of given name and doping")
        .add_property("all", &MaterialsDB_list, "List of all materials in database")
        .def("__iter__", &MaterialsDB_iter)
        .def("__contains__", &MaterialsDB_contains)
    ;

    // Common material interface
    py::class_<Material, shared_ptr<Material>, boost::noncopyable> MaterialClass("Material", "Base class for all materials.", py::no_init);
    MaterialClass
        .def("__init__", raw_constructor(&PythonMaterial::__init__))
        .def("complete_composition", &Material__completeComposition, (py::arg("composition"), py::arg("name")=""),
             "Fix incomplete material composition basing on pattern")
        .staticmethod("complete_composition")
        .add_property("name", &Material::name)
        .add_property("kind", &Material::kind)
        .def("__str__", &Material__str__)
        .def("__repr__", &Material__repr__)

        .def("lattC", &Material::lattC, (py::arg("T")=300., py::arg("x")), "Get lattice constant [A]")
        .def("Eg", &Material::Eg, (py::arg("T")=300., py::arg("point")='G'), "Get energy gap Eg [eV]")
        .def("CBO", &Material::CBO, (py::arg("T")=300., py::arg("point")='G'), "Get conduction band offset CBO [eV]")
        .def("VBO", &Material::VBO, (py::arg("T")=300.), "Get valance band offset VBO [eV]")
        .def("Dso", &Material::Dso, (py::arg("T")=300.), "Get split-off energy Dso [eV]")
        .def("Mso", &Material::Mso, (py::arg("T")=300.), "Get split-off mass Mso [m0]")
        .def("Me", &Material::Me, (py::arg("T")=300., py::arg("point")='G'), "Get split-off mass Mso [m0]")
        .def("Mhh", &Material::Mhh, (py::arg("T")=300., py::arg("point")='G'), "Get heavy hole effective mass Mhh [m0]")
        .def("Mlh", &Material::Mlh, (py::arg("T")=300., py::arg("point")='G'), "Get light hole effective mass Mlh [m0]")
        .def("Mh", &Material::Mh, (py::arg("T")=300., py::arg("eq")/*='G'*/), "Get hole effective mass Mh [m0]")
        .def("ac", &Material::ac, (py::arg("T")=300.), "Get hydrostatic deformation potential for the conduction band ac [eV]")
        .def("av", &Material::av, (py::arg("T")=300.), "Get hydrostatic deformation potential for the valence band av [eV]")
        .def("b", &Material::b, (py::arg("T")=300.), "Get shear deformation potential b [eV]")
        .def("c11", &Material::c11, (py::arg("T")=300.), "Get elastic constant c11 [GPa]")
        .def("c12", &Material::c12, (py::arg("T")=300.), "Get elastic constant c12 [GPa]")
        .def("eps", &Material::eps, (py::arg("T")=300.), "Get dielectric constant EpsR")
        .def("chi", (double (Material::*)(double, char) const)&Material::chi, (py::arg("T")=300., py::arg("point")='G'), "Get electron affinity Chi [eV]")
        .def("Nc", &Material::Nc, (py::arg("T")=300., py::arg("point")='G'), "Get effective density of states in the conduction band Nc [m**(-3)]")
        .def("Nv", &Material::Nv, (py::arg("T")=300.), "Get effective density of states in the valence band Nv [m**(-3)]")
        .def("Ni", &Material::Ni, (py::arg("T")=300.), "Get intrinsic carrier concentration Ni [m**(-3)]")
        .def("Nf", &Material::Nf, (py::arg("T")=300.), "Get free carrier concentration N [m**(-3)]")
        .def("EactD", &Material::EactD, (py::arg("T")=300.), "Get donor ionisation energy EactD [eV]")
        .def("EactA", &Material::EactA, (py::arg("T")=300.), "Get acceptor ionisation energy EactA [eV]")
        .def("mob", &Material::mob, (py::arg("T")=300.), "Get mobility [m**2/(V*s)]")
        .def("cond", &Material::cond, (py::arg("T")=300.), "Get electrical conductivity Sigma [S/m]")
        .add_property("condtype", &Material::condtype, "Electrical conductivity type")
        .def("A", &Material::A, (py::arg("T")=300.), "Get monomolecular recombination coefficient A [1/s]")
        .def("B", &Material::B, (py::arg("T")=300.), "Get radiative recombination coefficient B [m**3/s]")
        .def("C", &Material::C, (py::arg("T")=300.), "Get Auger recombination coefficient C [m**6/s]")
        .def("D", &Material::D, (py::arg("T")=300.), "Get ambipolar diffusion coefficient D [m**2/s]")
        .def("thermk", (DDPair (Material::*)(double, double) const)&Material::thermk, (py::arg("T")=300., py::arg("thickness")=INFINITY), "Get thermal conductivity [W/(m*K)]")
        .def("dens", &Material::dens, (py::arg("T")=300.), "Get density [kg/m**3]")
        .def("cp", &Material::cp, (py::arg("T")=300.), "Get specific heat at constant pressure [J/(kg*K)]")
        .def("nr", &Material::nr, (py::arg("wl"), py::arg("T")=300.), "Get refractive index nr")
        .def("absp", &Material::absp, (py::arg("wl"), py::arg("T")=300.), "Get absorption coefficient alpha")
        .def("nR", &Material::nR, (py::arg("wl"), py::arg("T")=300.), "Get complex refractive index nR")
        .def("nR_tensor", &Material::nR_tensor, (py::arg("wl"), py::arg("T")=300.), "Get complex refractive index tensor nR")
    ;

    Material_from_Python_string();
    register_exception<NoSuchMaterial>(PyExc_ValueError);
    register_exception<MaterialMethodNotApplicable>(PyExc_TypeError);

    // Make std::pair<double,double> and std::tuple<dcomplex,dcomplex,dcomplex,dcomplex,dcomplex> understandable
    detail::DDpair_fromto_Python();
    detail::ComplexTensor_fromto_Python();

    py_enum<Material::Kind> MaterialKind("Kind", "Kind of the material"); MaterialKind
        .value("NONE", Material::NONE)
        .value("SEMICONDUCTOR", Material::SEMICONDUCTOR)
        .value("OXIDE", Material::OXIDE)
        .value("DIELECTRIC", Material::DIELECTRIC)
        .value("METAL", Material::METAL)
        .value("LIQUID_CRYSTAL", Material::LIQUID_CRYSTAL)
        .value("MIXED", Material::MIXED)
    ;

    py_enum<Material::ConductivityType> MaterialConductivityType("ConductivityType", "Conductivity type of the material"); MaterialConductivityType
        .value("N", Material::CONDUCTIVITY_N)
        .value("I", Material::CONDUCTIVITY_I)
        .value("P", Material::CONDUCTIVITY_P)
        .value("OTHER", Material::CONDUCTIVITY_OTHER)
        .value("UNDETERMINED", Material::CONDUCTIVITY_UNDETERMINED)
    ;

    py::def("_register_material_simple", &registerSimpleMaterial, (py::arg("name"), py::arg("material"), py::arg("database")=MaterialsDB::getDefault()),
            "Register new simple material class to the database");

    py::def("_register_material_complex", &registerComplexMaterial, (py::arg("name"), py::arg("material"), py::arg("database")=MaterialsDB::getDefault()),
            "Register new complex material class to the database");
}

}} // namespace plask::python

#include "globals.h"
#include <boost/python/raw_function.hpp>
#include <boost/python/stl_iterator.hpp>
#include <algorithm>

#include <plask/config.h>
#include <plask/utils/string.h>
#include <plask/exceptions.h>
#include <plask/material/db.h>

#include "../util/raw_constructor.h"

namespace plask { namespace python {

/**
 * Wrapper for Material class.
 * For all virtual functions it calls Python derivatives
 */
class MaterialWrap : public Material
{
    shared_ptr<Material> base;
    PyObject* self;

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

    template <typename R, typename F, typename... Args>
    inline R override(const char* name, F f, Args... args) const {
        if (overriden(name)) return py::call_method<double>(self, name, args...);
        return ((*base).*f)(args...);
    }

    struct EmptyBase : public Material {
        virtual std::string name() const { return ""; }
    };

  public:
    MaterialWrap () : base(new EmptyBase) {}
    MaterialWrap (shared_ptr<Material> base) : base(base) {}

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


    // Here there are overrided methods from Material clas

    /// @return material name
    virtual std::string name() const { return attr<std::string>("name"); }
    virtual double lattC(double T, char x) const { return override<double>("lattC", &Material::lattC, T, x); }
    virtual double Eg(double T, const char Point) const { return override<double>("Eg", &Material::Eg, T, Point); }
    virtual double CBO(double T, const char Point) const { return override<double>("CBO", &Material::CBO, T, Point); }
    virtual double VBO(double T) const { return override<double>("VBO", &Material::VBO, T); }
    virtual double Dso(double T) const { return override<double>("Dso", &Material::Dso, T); }
    virtual double Mso(double T) const { return override<double>("Mso", &Material::Mso, T); }
    virtual double Me(double T, const char Point) const { return override<double>("Me", &Material::Me, T, Point); }
    virtual double Me_v(double T, const char Point) const { return override<double>("Me_v", &Material::Me_v, T, Point); }
    virtual double Me_l(double T, const char Point) const { return override<double>("Me_l", &Material::Me_l, T, Point); }
    virtual double Mhh(double T, const char Point) const { return override<double>("Mhh", &Material::Mhh, T, Point); }
    virtual double Mhh_v(double T, const char Point) const { return override<double>("Mhh_v", &Material::Mhh_v, T, Point); }
    virtual double Mhh_l(double T, const char Point) const { return override<double>("Mhh_l", &Material::Mhh_l, T, Point); }
    virtual double Mlh(double T, const char Point) const { return override<double>("Mlh", &Material::Mlh, T, Point); }
    virtual double Mlh_v(double T, const char Point) const { return override<double>("Mlh_v", &Material::Mlh_v, T, Point); }
    virtual double Mlh_l(double T, const char Point) const { return override<double>("Mlh_l", &Material::Mlh_l, T, Point); }
    virtual double Mh(double T, char EqType) const { return override<double>("Mh", &Material::Mh, T, EqType); }
    virtual double Mh_v(double T, const char Point) const { return override<double>("Mh_v", &Material::Mh_v, T, Point); }
    virtual double Mh_l(double T, const char Point) const { return override<double>("Mh_l", &Material::Mh_l, T, Point); }
    virtual double eps(double T) const { return override<double>("eps", &Material::eps, T); }
    virtual double chi(double T, const char Point) const { return override<double>("chi", (double (Material::*)(double, char) const) &Material::chi, T, Point); }
    virtual double Nc(double T, const char Point) const { return override<double>("Nc", (double (Material::*)(double, char) const) &Material::Nc, T, Point); }
    virtual double Ni(double T) const { return override<double>("Ni", &Material::Ni, T); }
    virtual double Nf(double T) const { return override<double>("Nf", &Material::Nf, T); }
    virtual double EactD(double T) const { return override<double>("EactD", &Material::EactD, T); }
    virtual double EactA(double T) const { return override<double>("EactA", &Material::EactA, T); }
    virtual double mob(double T) const { return override<double>("mob", &Material::mob, T); }
    virtual double cond(double T) const { return override<double>("cond", &Material::cond, T); }
    virtual double cond_v(double T) const { return override<double>("cond_v", &Material::cond_v, T); }
    virtual double cond_l(double T) const { return override<double>("cond_l", &Material::cond_l, T); }
    virtual double res(double T) const { return override<double>("res", &Material::res, T); }
    virtual double res_v(double T) const { return override<double>("res_v", &Material::res_v, T); }
    virtual double res_l(double T) const { return override<double>("res_l", &Material::res_l, T); }
    virtual double A(double T) const { return override<double>("A", &Material::A, T); }
    virtual double B(double T) const { return override<double>("B", &Material::B, T); }
    virtual double C(double T) const { return override<double>("C", &Material::C, T); }
    virtual double D(double T) const { return override<double>("D", &Material::D, T); }
    virtual double condT(double T, double t) const { return override<double>("condT", &Material::condT, T, t); }
    virtual double condT_v(double T, double t) const { return override<double>("condT_v", &Material::condT_v, T, t); }
    virtual double condT_l(double T, double t) const { return override<double>("condT_l", &Material::condT_l, T, t); }
    virtual double dens(double T) const { return override<double>("dens", &Material::dens, T); }
    virtual double specHeat(double T) const { return override<double>("specHeat", &Material::specHeat, T); }
    virtual double nr(double WaveLen, double T) const { return override<double>("nr", &Material::nr, WaveLen, T); }
    virtual double absp(double WaveLen, double T) const { return override<double>("absp", &Material::absp, WaveLen, T); }
    virtual dcomplex Nr(double WaveLen, double T) const { return override<dcomplex>("Nr", &Material::Nr, WaveLen, T); }

    // End of overriden methods

};


/**
 * Object constructing custom simple Python material when read from XML file
 *
 * \param name plain material name
 *
 * Other parameters are ignored
 */
class PythonSimpleMaterialConstructor : public MaterialsDB::MaterialConstructor
{
    py::object material_class;
    std::string dopant;

  public:
    PythonSimpleMaterialConstructor(const std::string& name, py::object material_class, std::string dope="") :
        MaterialsDB::MaterialConstructor(name), material_class(material_class), dopant(dope) {}

    inline shared_ptr<Material> operator()(const Material::Composition& composition, Material::DOPING_AMOUNT_TYPE doping_amount_type, double doping_amount) const
    {
        py::tuple args;
        py::dict kwargs;
        // Doping information
        if (doping_amount_type !=  Material::NO_DOPING) {
            kwargs["dopant"] = dopant;
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

    inline shared_ptr<Material> operator()(const Material::Composition& composition, Material::DOPING_AMOUNT_TYPE doping_amount_type, double doping_amount) const
    {
        py::dict kwargs;
        // Composition
        for (auto c : composition) kwargs[c.first] = c.second;
        // Doping information
        if (doping_amount_type !=  Material::NO_DOPING) {
            kwargs["dopant"] = dopant;
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
    std::string dopant = std::get<1>(splitString2(name, ':'));
    db.addSimple(new PythonSimpleMaterialConstructor(name, material_class, dopant));
}

/**
 * Function registering custom complex material class to plask
 * \param name name of the material
 * \param material_class Python class object of the custom material
 */
void registerComplexMaterial(const std::string& name, py::object material_class, MaterialsDB& db)
{
    std::string dopant = std::get<1>(splitString2(name, ':'));
    db.addComplex(new PythonComplexMaterialConstructor(name, material_class, dopant));
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

    // Get doping
    bool doping = false;
    std::string dopant = "";
    int doping_keys = 0;
    try {
        dopant = py::extract<std::string>(kwargs["dopant"]);
        doping = true;
        ++doping_keys;
    } catch (py::error_already_set) {
        PyErr_Clear();
    }
    Material::DOPING_AMOUNT_TYPE doping_type = Material::NO_DOPING;
    double concentation = 0;
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
        PyErr_SetString(PyExc_ValueError, "doping and carrier concentrations specified simultanously");
        throw py::error_already_set();
    }
    if (doping) {
        if (doping_type == Material::NO_DOPING) {
            PyErr_SetString(PyExc_ValueError, "dopant specified, but neither doping nor carrier concentrations given correctly");
            throw py::error_already_set();
        }
    } else {
        if (doping_type != Material::NO_DOPING) {
            PyErr_SetString(PyExc_ValueError, format("%s concentation given, but no dopant specified", has_dc?"doping":"carrier").c_str());
            throw py::error_already_set();
        }
    }
    concentation = py::extract<double>(cobj);

    std::size_t sep = name.find(':');
    if (sep != std::string::npos) {
        if (doping) {
            PyErr_SetString(PyExc_ValueError, "doping specified in **kwargs, but name contains ':'");
            throw py::error_already_set();
        } else {
           Material::parseDopant(name.substr(sep+1), dopant, doping_type, concentation);
        }
    }

    py::list keys = kwargs.keys();

     // Test if kwargs contains only doping information
    if (py::len(keys) == doping_keys) {
        std::string full_name;
        if (doping_type == Material::DOPANT_CONCENTRATION) full_name = format("%s:%s=%g", name, dopant, concentation);
        else full_name = format("%s:%s n=%g", name, dopant, concentation);;
        return  DB->get(full_name);
    }

    // So, kwargs contains compostion
    std::vector<std::string> elements = Material::parseElementsNames(name);
    py::object none;
    // test if only correct elements are given
    for (int i = 0; i < py::len(keys); ++i) {
        std::string k = py::extract<std::string>(keys[i]);
        if (k != "dopant" && k != "dc" && k != "cc" && std::find(elements.begin(), elements.end(), k) == elements.end()) {
            PyErr_SetString(PyExc_KeyError, format("%s not allowed in material %s", k, name).c_str());
            throw py::error_already_set();
        }
    }
    // make composition map
    Material::Composition composition;
    for (auto e: elements) {
        py::object v;
        try {
            v = kwargs[e];
        } catch (py::error_already_set) {
            PyErr_Clear();
        }
        composition[e] = (v != none) ? py::extract<double>(v): std::numeric_limits<double>::quiet_NaN();
    }

    return DB->get(composition, dopant, doping_type, concentation);
}

py::dict Material__completeComposition(py::dict src) {
    py::list keys = src.keys();
    Material::Composition comp;
    py::object none;
    for(int i = 0; i < py::len(keys); ++i) {
        std::string k = py::extract<std::string>(keys[i]);
        if (k != "dopant" && k != "dc" && k != "cc") {
            py::object s = src[keys[i]];
            comp[py::extract<std::string>(keys[i])] = (s != none) ? py::extract<double>(s): std::numeric_limits<double>::quiet_NaN();
        }
    }
    comp = Material::completeComposition(comp);

    py::dict result;
    for (auto c: comp) result[c.first] = c.second;
    return result;
}

void initMaterial() {

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
        .def("getDefault", &MaterialsDB::getDefault, "Get default database", py::return_value_policy<py::reference_existing_object>())
        .staticmethod("getDefault")
        .def("get", py::raw_function(&MaterialsDB_get), "Get material of given name and doping")
        .add_property("all", &MaterialsDB_list, "List of all materials in database")
        .def("__iter__", &MaterialsDB_iter)
    ;

    // Common material interface
    py::class_<Material, shared_ptr<Material>, boost::noncopyable>("Material", "Base class for all materials.", py::no_init)
        .def("__init__", raw_constructor(&MaterialWrap::__init__))
        .def("_completeComposition", &Material__completeComposition, "Fix incomplete material composition basing on patten")
        .staticmethod("_completeComposition")
        .add_property("name", &Material::name)

        .def("lattC", &Material::lattC, (py::arg("T")=300., py::arg("x")), "Get lattice constant [A]")
        .def("Eg", &Material::Eg, (py::arg("T")=300., py::arg("point")='G'), "Get energy gap Eg [eV]")
        .def("CBO", &Material::CBO, (py::arg("T")=300., py::arg("point")='G'), "Get conduction band offset CBO [eV]")
        .def("VBO", &Material::VBO, (py::arg("T")=300.), "Get valance band offset VBO [eV]")
        .def("Dso", &Material::Dso, (py::arg("T")=300.), "Get split-off energy Dso [eV]")
        .def("Mso", &Material::Mso, (py::arg("T")=300.), "Get split-off mass Mso [m0]")
        .def("Me", &Material::Me, (py::arg("T")=300., py::arg("point")='G'), "Get split-off mass Mso [m0]")
        .def("Me_v", &Material::Me_v, (py::arg("T")=300., py::arg("point")='G'), "Get electron effective mass Me in vertical direction [m0]")
        .def("Me_l", &Material::Me_l, (py::arg("T")=300., py::arg("point")='G'), "Get electron effective mass Me in lateral direction [m0]")
        .def("Mhh", &Material::Mhh, (py::arg("T")=300., py::arg("point")='G'), "Get heavy hole effective mass Mhh [m0]")
        .def("Mhh_v", &Material::Mhh_v, (py::arg("T")=300., py::arg("point")='G'), "Get heavy hole effective mass Mhh [m0]")
        .def("Mhh_l", &Material::Mhh_l, (py::arg("T")=300., py::arg("point")='G'), "Get heavy hole effective mass Me in lateral direction [m0]")
        .def("Mlh", &Material::Mlh, (py::arg("T")=300., py::arg("point")='G'), "Get light hole effective mass Mlh [m0]")
        .def("Mlh_v", &Material::Mlh_v, (py::arg("T")=300., py::arg("point")='G'), "Get light hole effective mass Me in vertical direction [m0]")
        .def("Mlh_l", &Material::Mlh_l, (py::arg("T")=300., py::arg("point")='G'), "Get light hole effective mass Me in lateral direction [m0]")
        .def("Mh", &Material::Mh, (py::arg("T")=300., py::arg("eq")/*='G'*/), "Get hole effective mass Mh [m0]")
        .def("Mh_v", &Material::Mh_v, (py::arg("T")=300., py::arg("point")='G'), "Get hole effective mass Me in vertical direction [m0]")
        .def("Mh_l", &Material::Mh_l, (py::arg("T")=300., py::arg("point")='G'), "Get hole effective mass Me in lateral direction [m0]")
        .def("eps", &Material::eps, (py::arg("T")=300.), "Get dielectric constant EpsR")
        .def("chi", (double (Material::*)(double, char) const)&Material::chi, (py::arg("T")=300., py::arg("point")='G'), "Get electron affinity Chi [eV]")
        .def("Nc", (double (Material::*)(double, char) const)&Material::Nc, (py::arg("T")=300., py::arg("point")='G'), "Get effective density of states in the conduction band Nc [m**(-3)]")
        .def("Ni", &Material::Ni, (py::arg("T")=300.), "Get intrinsic carrier concentration Ni [m**(-3)]")
        .def("Nf", &Material::Nf, (py::arg("T")=300.), "Get free carrier concentration N [m**(-3)]")
        .def("EactD", &Material::EactD, (py::arg("T")=300.), "Get donor ionisation energy EactD [eV]")
        .def("EactA", &Material::EactA, (py::arg("T")=300.), "Get acceptor ionisation energy EactA [eV]")
        .def("mob", &Material::mob, (py::arg("T")=300.), "Get mobility [m**2/(V*s)]")
        .def("cond", &Material::cond, (py::arg("T")=300.), "Get electrical conductivity Sigma [S/m]")
        .def("cond_v", &Material::cond_v, (py::arg("T")=300.), "Get electrical conductivity in vertical direction Sigma [S/m]")
        .def("cond_l", &Material::cond_l, (py::arg("T")=300.), "Get electrical conductivity in lateral direction Sigma [S/m]")
        .def("res", &Material::res, (py::arg("T")=300.), "Get electrical resistivity [Ohm*m]")
        .def("res_v", &Material::res_v, (py::arg("T")=300.), "Get electrical resistivity in vertical direction [Ohm*m]")
        .def("res_l", &Material::res_l, (py::arg("T")=300.), "Get electrical resistivity in vertical direction [Ohm*m]")
        .def("A", &Material::A, (py::arg("T")=300.), "Get monomolecular recombination coefficient A [1/s]")
        .def("B", &Material::B, (py::arg("T")=300.), "Get radiative recombination coefficient B [m**3/s]")
        .def("C", &Material::C, (py::arg("T")=300.), "Get Auger recombination coefficient C [m**6/s]")
        .def("D", &Material::D, (py::arg("T")=300.), "Get ambipolar diffusion coefficient D [m**2/s]")
        .def("condT", &Material::condT, (py::arg("T")=300., py::arg("thickness")), "Get thermal conductivity [W/(m*K)]")
        .def("condT_v", &Material::condT_v, (py::arg("T")=300., py::arg("thickness")), "Get thermal conductivity in vertical direction k [W/(m*K)]")
        .def("condT_l", &Material::condT_l, (py::arg("T")=300., py::arg("thickness")), "Get thermal conductivity in lateral direction k [W/(m*K)]")
        .def("dens", &Material::dens, (py::arg("T")=300.), "Get density [kg/m**3]")
        .def("specHeat", &Material::specHeat, (py::arg("T")=300.), "Get specific heat at constant pressure [J/(kg*K)]")
        .def("nr", &Material::nr, (py::arg("wl"), py::arg("T")=300.), "Get refractive index nR")
        .def("absp", &Material::absp, (py::arg("wl"), py::arg("T")=300.), "Get absorption coefficient alpha")
        .def("Nr", &Material::Nr, (py::arg("wl"), py::arg("T")=300.), "Get refractive index nR")

        ;

    py::def("_register_material_simple", &registerSimpleMaterial, (py::arg("name"), py::arg("material"), py::arg("database")=MaterialsDB::getDefault()),
            "Register new simple material class to the database");

    py::def("_register_material_complex", &registerComplexMaterial, (py::arg("name"), py::arg("material"), py::arg("database")=MaterialsDB::getDefault()),
            "Register new complex material class to the database");

    scope.attr("database") = MaterialsDB::getDefault();
}

}} // namespace plask::python

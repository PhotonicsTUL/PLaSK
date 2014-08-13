#include "python_globals.h"
#include "python_material.h"
#include <boost/python/raw_function.hpp>
#include <boost/python/stl_iterator.hpp>
#include <algorithm>

#include <plask/config.h>
#include <plask/utils/string.h>
#include <plask/exceptions.h>
#include <plask/material/db.h>
#include <plask/material/info.h>

#include "../util/raw_constructor.h"

namespace plask { namespace python {

namespace detail {
    struct Tensor2_fromto_Python
    {
        Tensor2_fromto_Python() {
            boost::python::converter::registry::push_back(&convertible, &construct, boost::python::type_id<Tensor2<double>>());
            boost::python::to_python_converter<Tensor2<double>, Tensor2_fromto_Python>();
        }

        static void* convertible(PyObject* obj) {
            if (!PySequence_Check(obj) && !PyFloat_Check(obj) && !PyInt_Check(obj)) return NULL;
            return obj;
        }

        static void construct(PyObject* obj, boost::python::converter::rvalue_from_python_stage1_data* data) {
            void* storage = ((boost::python::converter::rvalue_from_python_storage<Tensor2<double>>*)data)->storage.bytes;
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
            new(storage) Tensor2<double>(first, second);
            data->convertible = storage;
        }

        static PyObject* convert(const Tensor2<double>& pair)  {
            py::tuple tuple = py::make_tuple(pair.c00, pair.c11);
            return boost::python::incref(tuple.ptr());
        }
    };

    struct ComplexTensor_fromto_Python
    {
        ComplexTensor_fromto_Python() {
            boost::python::converter::registry::push_back(&convertible, &construct, boost::python::type_id<Tensor3<dcomplex>>());
            boost::python::to_python_converter<Tensor3<dcomplex>, ComplexTensor_fromto_Python>();
        }

        static void* convertible(PyObject* obj) {
            if (!PySequence_Check(obj)) return NULL;
            return obj;
        }

        static void construct(PyObject* obj, boost::python::converter::rvalue_from_python_stage1_data* data) {
            py::object src = py::object(py::handle<>(py::borrowed(obj)));
            void* storage = ((boost::python::converter::rvalue_from_python_storage<Tensor3<dcomplex>>*)data)->storage.bytes;
            dcomplex vals[4];
            int idx[4] = { 0, 1, 2, 3 };
            auto seq = py::object(py::handle<>(py::borrowed(obj)));
            if (py::len(seq) == 2) { idx[1] = 0; idx[2] = 1; idx[3] = -1; }
            else if (py::len(seq) == 3) { idx[3] = -1; }
            else if (py::len(seq) != 4)
                throw TypeError("sequence of exactly 2, 3, or 4 complex required");
            for (int i = 0; i < 4; ++i) {
                if (idx[i] != -1)  vals[i] = py::extract<dcomplex>(seq[idx[i]]);
                else vals[i] = 0.;
            }
            new(storage) Tensor3<dcomplex>(vals[0], vals[1], vals[2], vals[3]);
            data->convertible = storage;
        }

        static PyObject* convert(const Tensor3<dcomplex>& src)  {
            py::tuple tuple = py::make_tuple(src.c00, src.c11, src.c22, src.c01);
            return boost::python::incref(tuple.ptr());
        }
    };

    struct StringFromMaterial
    {
        StringFromMaterial() {
            boost::python::converter::registry::push_back(&convertible, &construct, boost::python::type_id<std::string>());
        }

        static void* convertible(PyObject* obj) {
            return boost::python::converter::implicit_rvalue_convertible_from_python(obj, boost::python::converter::registered<Material>::converters)? obj : 0;
        }

        static void construct(PyObject* obj, boost::python::converter::rvalue_from_python_stage1_data* data) {
            void* storage = (( boost::python::converter::rvalue_from_python_storage<std::string>*)data)->storage.bytes;
            py::arg_from_python<Material*> get_source(obj);
            bool convertible = get_source.convertible();
            BOOST_VERIFY(convertible);
            new (storage) std::string(get_source()->str());
            data->convertible = storage;
        }

        static PyObject* convert(const Tensor2<double>& pair)  {
            py::tuple tuple = py::make_tuple(pair.c00, pair.c11);
            return boost::python::incref(tuple.ptr());
        }
    };

}

/**
 * Converter for Python string to material using default database.
 * Allows to create geometry objects as \c rectange(2,1,"GaAs")
 */
struct MaterialFromPythonString {

    MaterialFromPythonString() {
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


/**
 * Wrapper for Material class.
 * For all virtual functions it calls Python derivatives
 */
class PythonMaterial : public Material
{
    shared_ptr<Material> base;
    PyObject* self;

    static std::map<PyObject*, std::unique_ptr<MaterialCache>> cacheMap;
    MaterialCache* cache;

    bool overriden(char const* name) const {
        py::converter::registration const& r = py::converter::registered<Material>::converters;
        PyTypeObject* class_object = r.get_class_object();
        if (self) {
            py::handle<> mh(PyObject_GetAttrString(self, const_cast<char*>(name)));
            if (mh && PyMethod_Check(mh.get())) {
                PyMethodObject* mo = (PyMethodObject*)mh.get();
                PyObject* borrowed_f = nullptr;
                if(mo->im_self == self && class_object->tp_dict != 0)
                    borrowed_f = PyDict_GetItemString(class_object->tp_dict, const_cast<char*>(name));
                if (borrowed_f != mo->im_func) return true;
            }
        }
        return false;
    }

    template <typename R, typename F, typename... Args>
    inline R call(const char* name, F f, const boost::optional<R>& cached, Args... args) const {
        if (cached) return *cached;
        OmpLockGuard<OmpNestLock> lock(python_omp_lock);
        if (overriden(name)) {
            return py::call_method<R>(self, name, args...);
        }
        return ((*base).*f)(args...);
    }

  public:
    PythonMaterial(): base(new EmptyMaterial) {}
    PythonMaterial(shared_ptr<Material> base): base(base) {
        if (!base) base = shared_ptr<Material>(new EmptyMaterial);
    }

    static shared_ptr<Material> __init__(py::tuple args, py::dict kwargs) {
        int len = py::len(args);
        if (len > 2 || py::len(kwargs) != 0) {
            throw TypeError("__init__ takes at most 2 arguments (%d given)", len);
        }
        PythonMaterial* ptr;
        if (len == 2) {
            shared_ptr<Material> base;
            try {
                base = MaterialsDB::getDefault().get(py::extract<std::string>(args[1]));
            } catch (py::error_already_set) {
                PyErr_Clear();
                base = py::extract<shared_ptr<Material>>(args[1]);
            }
            ptr = new PythonMaterial(base);
        } else {
            ptr = new PythonMaterial();
        }
        auto sptr = shared_ptr<Material>(ptr);
        py::object self(args[0]);
        ptr->self = self.ptr();  // key line !!!
        // Update cache
        py::object cls = self.attr("__class__");
        auto found = cacheMap.find(cls.ptr());
        if (found != cacheMap.end())
            ptr->cache = found->second.get();
        else {
            std::string cls_name = py::extract<std::string>(cls.attr("__name__"));
            // MaterialCache* cache = cacheMap.emplace(cls.ptr(), std::unique_ptr<MaterialCache>(new MaterialCache)).first->second.get();
            MaterialCache* cache = (cacheMap[cls.ptr()] = std::move(std::unique_ptr<MaterialCache>(new MaterialCache))).get();
            ptr->cache = cache;
            #define CHECK_CACHE(Type, fun, name, ...) \
                if (PyObject_HasAttrString(self.ptr(), name) && PyFunction_Check(py::object(self.attr(name)).ptr())) { \
                    cache->fun.reset(py::extract<Type>(self.attr(name)())); \
                    writelog(LOG_DEBUG, "Caching parameter '" name "' in material class '%1%'", cls_name); \
                }
            CHECK_CACHE(double, lattC, "lattC", 300., py::object())
            CHECK_CACHE(double, Eg, "Eg", 300., 0., "G")
            CHECK_CACHE(double, CB, "CB", 300., 0., "G")
            CHECK_CACHE(double, VB, "VB", 300., 0., "H")
            CHECK_CACHE(double, Dso, "Dso", 300., 0.)
            CHECK_CACHE(double, Mso, "Mso", 300., 0.)
            CHECK_CACHE(Tensor2<double>, Me, "Me", 300., 0.)
            CHECK_CACHE(Tensor2<double>, Mhh, "Mhh", 300., 0.)
            CHECK_CACHE(Tensor2<double>, Mlh, "Mlh", 300., 0.)
            CHECK_CACHE(Tensor2<double>, Mh, "Mh", 300., 0.)
            CHECK_CACHE(double, ac, "ac", 300.)
            CHECK_CACHE(double, av, "av", 300.)
            CHECK_CACHE(double, b, "b", 300.)
            CHECK_CACHE(double, d, "d", 300.)
            CHECK_CACHE(double, c11, "c11", 300.)
            CHECK_CACHE(double, c12, "c12", 300.)
            CHECK_CACHE(double, c44, "c44", 300.)
            CHECK_CACHE(double, eps, "eps", 300.)
            CHECK_CACHE(double, chi, "chi", 300., 0., "G")
            CHECK_CACHE(double, Nc, "Nc", 300., 0., "G")
            CHECK_CACHE(double, Nv, "Nv", 300., 0., "G")
            CHECK_CACHE(double, Ni, "Ni", 300.)
            CHECK_CACHE(double, Nf, "Nf", 300.)
            CHECK_CACHE(double, EactD, "EactD", 300.)
            CHECK_CACHE(double, EactA, "EactA", 300.)
            CHECK_CACHE(Tensor2<double>, mob, "mob", 300.)
            CHECK_CACHE(Tensor2<double>, cond, "cond", 300.)
            CHECK_CACHE(double, A, "A", 300.)
            CHECK_CACHE(double, B, "B", 300.)
            CHECK_CACHE(double, C, "C", 300.)
            CHECK_CACHE(double, D, "D", 300.)
            CHECK_CACHE(Tensor2<double>, thermk, "thermk", 300., INFINITY)
            CHECK_CACHE(double, dens, "dens", 300.)
            CHECK_CACHE(double, cp, "cp", 300.)
            CHECK_CACHE(double, nr, "nr", py::object(), 300., 0.)
            CHECK_CACHE(double, absp, "absp", py::object(), 300.)
            CHECK_CACHE(dcomplex, Nr, "Nr", py::object(), 300., 0.)
            CHECK_CACHE(Tensor3<dcomplex>, NR, "NR", py::object(), 300., 0.)
        }
        return sptr;
    }


    // Here there are overridden methods from Material class

    OmpLockGuard<OmpNestLock> lock() const override {
        return OmpLockGuard<OmpNestLock>(python_omp_lock);
    }

    virtual bool isEqual(const Material& other) const override {
        auto theother = static_cast<const PythonMaterial&>(other);
        py::object oself { py::borrowed(self) },
                   oother { py::object(py::borrowed(theother.self)) };

        if (overriden("__eq__")) {
            OmpLockGuard<OmpNestLock> lock(python_omp_lock);
            return py::call_method<bool>(self, "__eq__", oother);
        }

        return *base == *theother.base &&
                oself.attr("__class__") == oother.attr("__class__") &&
                oself.attr("__dict__") == oother.attr("__dict__");
    }

    virtual std::string name() const override {
        py::object cls = py::object(py::borrowed(self)).attr("__class__");
        py::object oname;
        try {
            oname = cls.attr("__dict__")["name"];
        } catch (py::error_already_set) {
            PyErr_Clear();
            oname = cls.attr("__name__");
        }
        return py::extract<std::string>(oname);
    }

    virtual std::string str() const override {
        if (overriden("__str__")) {
            OmpLockGuard<OmpNestLock> lock(python_omp_lock);
            return py::call_method<std::string>(self, "__str__");
        }
        else return name();
    }

    virtual Material::ConductivityType condtype() const override {
        py::object cls = py::object(py::borrowed(self)).attr("__class__");
        py::object octype;
        try {
            octype = cls.attr("__dict__")["condtype"];
        } catch (py::error_already_set) {
            PyErr_Clear();
            return base->condtype();
        }
        return py::extract<Material::ConductivityType>(octype);
    }

    virtual Material::Kind kind() const override {
        py::object cls = py::object(py::borrowed(self)).attr("__class__");
        py::object okind;
        try {
            okind = cls.attr("__dict__")["kind"];
        } catch (py::error_already_set) {
            PyErr_Clear();
            return base->kind();
        }
        return py::extract<Material::Kind>(okind);
    }

    virtual double lattC(double T, char x) const override { return call<double>("lattC", &Material::lattC, cache->lattC, T, x); }
    virtual double Eg(double T, double e, char point) const override { return call<double>("Eg", &Material::Eg, cache->Eg, T, e, point); }
    virtual double CB(double T, double e, char point) const override {
        try { return call<double>("CB", &Material::CB, cache->CB, T, e, point); }
        catch (NotImplemented) { return VB(T, e, point, 'H') + Eg(T, e, point); }  // D = µ kB T / e
    }
    virtual double VB(double T, double e, char point, char hole) const  override{ return call<double>("VB", &Material::VB, cache->VB, T, e, point, hole); }
    virtual double Dso(double T, double e) const override { return call<double>("Dso", &Material::Dso, cache->Dso, T, e); }
    virtual double Mso(double T, double e) const override { return call<double>("Mso", &Material::Mso, cache->Mso, T, e); }
    virtual Tensor2<double> Me(double T, double e, char point) const override { return call<Tensor2<double>>("Me", &Material::Me, cache->Me, T, e, point); }
    virtual Tensor2<double> Mhh(double T, double e) const override { return call<Tensor2<double>>("Mhh", &Material::Mhh, cache->Mhh, T, e); }
    virtual Tensor2<double> Mlh(double T, double e) const override { return call<Tensor2<double>>("Mlh", &Material::Mlh, cache->Mlh, T, e); }
    virtual Tensor2<double> Mh(double T, double e) const override { return call<Tensor2<double>>("Mh", &Material::Mh, cache->Mh, T, e); }
    virtual double ac(double T) const override { return call<double>("ac", &Material::ac, cache->ac, T); }
    virtual double av(double T) const override { return call<double>("av", &Material::av, cache->av, T); }
    virtual double b(double T) const override { return call<double>("b", &Material::b, cache->b, T); }
    virtual double d(double T) const override { return call<double>("d", &Material::d, cache->d, T); }
    virtual double c11(double T) const override { return call<double>("c11", &Material::c11, cache->c11, T); }
    virtual double c12(double T) const override { return call<double>("c12", &Material::c12, cache->c12, T); }
    virtual double c44(double T) const override { return call<double>("c44", &Material::c44, cache->c44, T); }
    virtual double eps(double T) const override { return call<double>("eps", &Material::eps, cache->eps, T); }
    virtual double chi(double T, double e, char point) const override { return call<double>("chi", &Material::chi, cache->chi, T, e, point); }
    virtual double Nc(double T, double e, char point) const override { return call<double>("Nc", &Material::Nc, cache->Nc, T, e, point); }
    virtual double Nv(double T, double e, char point) const override { return call<double>("Nv", &Material::Nv, cache->Nv, T, e, point); }
    virtual double Ni(double T) const override { return call<double>("Ni", &Material::Ni, cache->Ni, T); }
    virtual double Nf(double T) const override { return call<double>("Nf", &Material::Nf, cache->Nf, T); }
    virtual double EactD(double T) const override { return call<double>("EactD", &Material::EactD, cache->EactD, T); }
    virtual double EactA(double T) const override { return call<double>("EactA", &Material::EactA, cache->EactA, T); }
    virtual Tensor2<double> mob(double T) const override { return call<Tensor2<double>>("mob", &Material::mob, cache->mob, T); }
    virtual Tensor2<double> cond(double T) const override { return call<Tensor2<double>>("cond", &Material::cond, cache->cond, T); }
    virtual double A(double T) const override { return call<double>("A", &Material::A, cache->A, T); }
    virtual double B(double T) const override { return call<double>("B", &Material::B, cache->B, T); }
    virtual double C(double T) const override { return call<double>("C", &Material::C, cache->C, T); }
    virtual double D(double T) const override {
        try { return call<double>("D", &Material::D, cache->D, T); }
        catch (NotImplemented) { return mob(T).c00 * T * 8.6173423e-5; }  // D = µ kB T / e
    }
    virtual Tensor2<double> thermk(double T, double t) const override { return call<Tensor2<double>>("thermk", &Material::thermk, cache->thermk, T, t); }
    virtual double dens(double T) const override { return call<double>("dens", &Material::dens, cache->dens, T); }
    virtual double cp(double T) const override { return call<double>("cp", &Material::cp, cache->cp, T); }
    virtual double nr(double wl, double T, double n) const override { return call<double>("nr", &Material::nr, cache->nr, wl, T, n); }
    virtual double absp(double wl, double T) const override { return call<double>("absp", &Material::absp, cache->absp, wl, T); }
    virtual dcomplex Nr(double wl, double T, double n) const override {
        if (cache->Nr) return *cache->Nr;
        if (overriden("Nr")) {
            OmpLockGuard<OmpNestLock> lock(python_omp_lock);
            return py::call_method<dcomplex>(self, "Nr", wl, T, n);
        }
        if (cache->nr || cache->absp || overriden("nr") || overriden("absp"))
            return dcomplex(call<double>("nr", &Material::nr, cache->nr, wl, T, n), -7.95774715459e-09*call<double>("absp", &Material::absp, cache->absp, wl,T)*wl);
        return base->Nr(wl, T, n);
    }
    virtual Tensor3<dcomplex> NR(double wl, double T, double n) const override {
        if (cache->NR) return *cache->NR;
        if (overriden("NR")) {
            OmpLockGuard<OmpNestLock> lock(python_omp_lock);
            return py::call_method<Tensor3<dcomplex>>(self, "NR", wl, T, n);
        }
        if (cache->Nr || overriden("Nr")) {
            dcomplex nr;
            if (cache->Nr)
                nr = *cache->Nr;
            else {
                OmpLockGuard<OmpNestLock> lock(python_omp_lock);
                nr = py::call_method<dcomplex>(self, "Nr", wl, T, n);
            }
            return Tensor3<dcomplex>(nr, nr, nr, 0.);
        }
        if (cache->nr || cache->absp || overriden("nr") || overriden("absp")) {
            dcomplex nr (call<double>("nr", &Material::nr, cache->nr, wl, T, n), -7.95774715459e-09*call<double>("absp", &Material::absp, cache->absp, wl,T)*wl);
            return Tensor3<dcomplex>(nr, nr, nr, 0.);
        }
        return base->NR(wl, T, n);
    }

    // End of overriden methods

};

std::map<PyObject*, std::unique_ptr<MaterialCache>> PythonMaterial::cacheMap;

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

namespace detail {
    inline bool getRanges(const MaterialInfo::PropertyInfo&, py::dict&) { return false; }

    template <typename... Args>
    inline bool getRanges(const MaterialInfo::PropertyInfo& info, py::dict& ranges, MaterialInfo::ARGUMENT_NAME arg, Args... args) {
        auto range = info.getArgumentRange(arg);
        if (!isnan(range.first) ||   !isnan(range.second)) {
            ranges[MaterialInfo::ARGUMENT_NAME_STRING[unsigned(arg)]] = py::make_tuple(range.first, range.second);
            getRanges(info, ranges, args...);
            return true;
        }
        return getRanges(info, ranges, args...);
    }

    template <typename... Args>
    void getPropertyInfo(py::dict& result, const MaterialInfo& minfo, MaterialInfo::PROPERTY_NAME prop, Args... args) {
        if (boost::optional<plask::MaterialInfo::PropertyInfo> info = minfo.getPropertyInfo(prop)) {
            py::dict data;
            if (info->getSource() != "") data["source"] = info->getSource();
            if (info->getComment() != "") data["comment"] = info->getComment();
            py::list links;
            for (const auto& link: info->getLinks()) {
                if (link.comment == "")
                    links.append(py::make_tuple(link.className, MaterialInfo::PROPERTY_NAME_STRING[unsigned(link.property)]));
                else
                    links.append(py::make_tuple(link.className, MaterialInfo::PROPERTY_NAME_STRING[unsigned(link.property)], link.comment));
            }
            if (links) data["seealso"] = links;
            py::dict ranges;
            if (getRanges(*info, ranges, MaterialInfo::doping, args...)) data["ranges"] = ranges;
            result[MaterialInfo::PROPERTY_NAME_STRING[unsigned(prop)]] = data;
        }
    }

}

py::dict getMaterialInfo(const std::string& name) {
    boost::optional<MaterialInfo> minfo = MaterialInfo::DB::getDefault().get(name);
    py::dict result;
    if (!minfo) return result;
    detail::getPropertyInfo(result, *minfo, MaterialInfo::kind);
    detail::getPropertyInfo(result, *minfo, MaterialInfo::lattC, MaterialInfo::T, MaterialInfo::e);
    detail::getPropertyInfo(result, *minfo, MaterialInfo::Eg, MaterialInfo::T, MaterialInfo::e);
    detail::getPropertyInfo(result, *minfo, MaterialInfo::CB, MaterialInfo::T, MaterialInfo::e);
    detail::getPropertyInfo(result, *minfo, MaterialInfo::VB, MaterialInfo::T, MaterialInfo::e);
    detail::getPropertyInfo(result, *minfo, MaterialInfo::Dso, MaterialInfo::T, MaterialInfo::e);
    detail::getPropertyInfo(result, *minfo, MaterialInfo::Mso, MaterialInfo::T, MaterialInfo::e);
    detail::getPropertyInfo(result, *minfo, MaterialInfo::Me, MaterialInfo::T, MaterialInfo::e);
    detail::getPropertyInfo(result, *minfo, MaterialInfo::Mhh, MaterialInfo::T, MaterialInfo::e);
    detail::getPropertyInfo(result, *minfo, MaterialInfo::Mlh, MaterialInfo::T, MaterialInfo::e);
    detail::getPropertyInfo(result, *minfo, MaterialInfo::Mh, MaterialInfo::T, MaterialInfo::e);
    detail::getPropertyInfo(result, *minfo, MaterialInfo::ac, MaterialInfo::T);
    detail::getPropertyInfo(result, *minfo, MaterialInfo::av, MaterialInfo::T);
    detail::getPropertyInfo(result, *minfo, MaterialInfo::b, MaterialInfo::T);
    detail::getPropertyInfo(result, *minfo, MaterialInfo::d, MaterialInfo::T);
    detail::getPropertyInfo(result, *minfo, MaterialInfo::c11, MaterialInfo::T);
    detail::getPropertyInfo(result, *minfo, MaterialInfo::c12, MaterialInfo::T);
    detail::getPropertyInfo(result, *minfo, MaterialInfo::c44, MaterialInfo::T);
    detail::getPropertyInfo(result, *minfo, MaterialInfo::eps, MaterialInfo::T);
    detail::getPropertyInfo(result, *minfo, MaterialInfo::chi, MaterialInfo::T);
    detail::getPropertyInfo(result, *minfo, MaterialInfo::Nc, MaterialInfo::T);
    detail::getPropertyInfo(result, *minfo, MaterialInfo::Nv, MaterialInfo::T);
    detail::getPropertyInfo(result, *minfo, MaterialInfo::Ni, MaterialInfo::T);
    detail::getPropertyInfo(result, *minfo, MaterialInfo::Nf, MaterialInfo::T);
    detail::getPropertyInfo(result, *minfo, MaterialInfo::EactD, MaterialInfo::T);
    detail::getPropertyInfo(result, *minfo, MaterialInfo::EactA, MaterialInfo::T);
    detail::getPropertyInfo(result, *minfo, MaterialInfo::mob, MaterialInfo::T);
    detail::getPropertyInfo(result, *minfo, MaterialInfo::cond, MaterialInfo::T);
    detail::getPropertyInfo(result, *minfo, MaterialInfo::condtype);
    detail::getPropertyInfo(result, *minfo, MaterialInfo::A, MaterialInfo::T);
    detail::getPropertyInfo(result, *minfo, MaterialInfo::B, MaterialInfo::T);
    detail::getPropertyInfo(result, *minfo, MaterialInfo::C, MaterialInfo::T);
    detail::getPropertyInfo(result, *minfo, MaterialInfo::D, MaterialInfo::T);
    detail::getPropertyInfo(result, *minfo, MaterialInfo::thermk, MaterialInfo::T, MaterialInfo::h);
    detail::getPropertyInfo(result, *minfo, MaterialInfo::dens, MaterialInfo::T);
    detail::getPropertyInfo(result, *minfo, MaterialInfo::cp, MaterialInfo::T);
    detail::getPropertyInfo(result, *minfo, MaterialInfo::nr, MaterialInfo::wl, MaterialInfo::T, MaterialInfo::n);
    detail::getPropertyInfo(result, *minfo, MaterialInfo::absp, MaterialInfo::wl, MaterialInfo::T, MaterialInfo::n);
    detail::getPropertyInfo(result, *minfo, MaterialInfo::Nr, MaterialInfo::wl, MaterialInfo::T, MaterialInfo::n);
    detail::getPropertyInfo(result, *minfo, MaterialInfo::NR, MaterialInfo::wl, MaterialInfo::T, MaterialInfo::n);
    return result;
}

void initMaterials() {

    py::object materials_module { py::handle<>(py::borrowed(PyImport_AddModule("plask.material"))) };
    py::scope().attr("material") = materials_module;
    py::scope scope = materials_module;

    scope.attr("__doc__") =
        "Materials and material database.\n\n"
    ;

    py::class_<MaterialsDB, shared_ptr<MaterialsDB>/*, boost::noncopyable*/> materialsDB("MaterialsDB",
        "Material database class\n\n"
        "Many semiconductor materials used in photonics are defined here. We have made\n"
        "a significant effort to ensure their physical properties to be the most precise\n"
        "as the current state of the art. However, you can derive an abstract class\n"
        ":class:`plask.Material` to create your own materials.\n"
        ); materialsDB
        .def("get_default", &MaterialsDB::getDefault, "Get default database.", py::return_value_policy<py::reference_existing_object>())
        .staticmethod("get_default")
        .def("load", &MaterialsDB::loadToDefault,
             "Load materials from library ``lib`` to default database.\n\n"
             "This method can be used to extend the database with custom materials provided\n"
             "in a binary library.\n\n"
             "Args:\n"
             "    lib (str): Library name to load.\n",
             py::arg("lib"))
        .staticmethod("load") // TODO make it non-static
        .def("load_all", &MaterialsDB::loadAllToDefault,
             "Load all materials from specified directory to default database.\n\n"
             "This method can be used to extend the database with custom materials provided\n"
             "in binary libraries.\n\n"
             "Args:\n"
             "    dir (str): Directory name to load materials from.\n",
             py::arg("dir")=plaskMaterialsPath())
        .staticmethod("load_all") // TODO make it non-static
        .def("get", py::raw_function(&MaterialsDB_get),
             "Get material of given name and doping."
             )
        .def("__getitem__", (shared_ptr<Material>(MaterialsDB::*)(const std::string&)const)&MaterialsDB::get)
        .add_property("all", &MaterialsDB_list, "List of all materials in the database.")
        .def("__iter__", &MaterialsDB_iter)
        .def("__contains__", &MaterialsDB_contains)
        .def("info", &getMaterialInfo, py::arg("name"),
             "Get information dictionary on built-in material.\n\n"
             "Args:\n"
             "    name (str): material name without doping amount and composition.\n"
             "                (e.g. 'GaAs:Si', 'AlGaAs')."
            )
        .staticmethod("info")
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
        .def("__eq__", (bool(Material::*)(const Material&)const)&Material::operator==)

        .def("lattC", &Material::lattC, (py::arg("T")=300., py::arg("x")), "Get lattice constant [A]")
        .def("Eg", &Material::Eg, (py::arg("T")=300., py::arg("e")=0, py::arg("point")='G'), "Get energy gap Eg [eV]")
        .def("CB", &Material::CB, (py::arg("T")=300., py::arg("e")=0, py::arg("point")='G'), "Get conduction band level CB [eV]")
        .def("VB", &Material::VB, (py::arg("T")=300., py::arg("e")=0, py::arg("point")='G', py::arg("hole")='H'), "Get valance band level VB [eV]")
        .def("Dso", &Material::Dso, (py::arg("T")=300., py::arg("e")=0), "Get split-off energy Dso [eV]")
        .def("Mso", &Material::Mso, (py::arg("T")=300., py::arg("e")=0), "Get split-off mass Mso [m0]")
        .def("Me", &Material::Me, (py::arg("T")=300., py::arg("e")=0, py::arg("point")='G'), "Get split-off mass Mso [m0]")
        .def("Mhh", &Material::Mhh, (py::arg("T")=300., py::arg("e")=0), "Get heavy hole effective mass Mhh [m0]")
        .def("Mlh", &Material::Mlh, (py::arg("T")=300., py::arg("e")=0), "Get light hole effective mass Mlh [m0]")
        .def("Mh", &Material::Mh, (py::arg("T")=300., py::arg("e")=0), "Get hole effective mass Mh [m0]")
        .def("ac", &Material::ac, (py::arg("T")=300.), "Get hydrostatic deformation potential for the conduction band ac [eV]")
        .def("av", &Material::av, (py::arg("T")=300.), "Get hydrostatic deformation potential for the valence band av [eV]")
        .def("b", &Material::b, (py::arg("T")=300.), "Get shear deformation potential b [eV]")
        .def("d", &Material::d, (py::arg("T")=300.), "Get shear deformation potential d [eV]")
        .def("c11", &Material::c11, (py::arg("T")=300.), "Get elastic constant c11 [GPa]")
        .def("c12", &Material::c12, (py::arg("T")=300.), "Get elastic constant c12 [GPa]")
        .def("c44", &Material::c44, (py::arg("T")=300.), "Get elastic constant c44 [GPa]")
        .def("eps", &Material::eps, (py::arg("T")=300.), "Get dielectric constant EpsR")
        .def("chi", &Material::chi, (py::arg("T")=300., py::arg("e")=0, py::arg("point")='G'), "Get electron affinity Chi [eV]")
        .def("Nc", &Material::Nc, (py::arg("T")=300., py::arg("e")=0, py::arg("point")='G'), "Get effective density of states in the conduction band Nc [m**(-3)]")
        .def("Nv", &Material::Nv, (py::arg("T")=300., py::arg("e")=0, py::arg("point")='G'), "Get effective density of states in the valence band Nv [m**(-3)]")
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
        .def("thermk", &Material::thermk, (py::arg("T")=300., py::arg("h")=INFINITY), "Get thermal conductivity [W/(m*K)]")
        .def("dens", &Material::dens, (py::arg("T")=300.), "Get density [kg/m**3]")
        .def("cp", &Material::cp, (py::arg("T")=300.), "Get specific heat at constant pressure [J/(kg*K)]")
        .def("nr", &Material::nr, (py::arg("wl"), py::arg("T")=300., py::arg("n")=0.), "Get refractive index nr")
        .def("absp", &Material::absp, (py::arg("wl"), py::arg("T")=300.), "Get absorption coefficient alpha")
        .def("Nr", &Material::Nr, (py::arg("wl"), py::arg("T")=300., py::arg("n")=0.), "Get complex refractive index Nr")
        .def("NR", &Material::NR, (py::arg("wl"), py::arg("T")=300., py::arg("n")=0.), "Get complex refractive index tensor Nr")
    ;

    MaterialFromPythonString();
    register_exception<NoSuchMaterial>(PyExc_ValueError);
    register_exception<MaterialMethodNotApplicable>(PyExc_TypeError);

    // Make std::pair<double,double> and std::tuple<dcomplex,dcomplex,dcomplex,dcomplex,dcomplex> understandable
    detail::Tensor2_fromto_Python();
    detail::ComplexTensor_fromto_Python();

    py_enum<Material::Kind>()
        .value("NONE", Material::NONE)
        .value("SEMICONDUCTOR", Material::SEMICONDUCTOR)
        .value("OXIDE", Material::OXIDE)
        .value("DIELECTRIC", Material::DIELECTRIC)
        .value("METAL", Material::METAL)
        .value("LIQUID_CRYSTAL", Material::LIQUID_CRYSTAL)
        .value("MIXED", Material::MIXED)
    ;

    py_enum<Material::ConductivityType>()
        .value("N", Material::CONDUCTIVITY_N)
        .value("I", Material::CONDUCTIVITY_I)
        .value("P", Material::CONDUCTIVITY_P)
        .value("OTHER", Material::CONDUCTIVITY_OTHER)
        .value("UNDETERMINED", Material::CONDUCTIVITY_UNDETERMINED)
    ;

    detail::StringFromMaterial();

    py::def("_register_material_simple", &registerSimpleMaterial, (py::arg("name"), py::arg("material"), py::arg("database")=MaterialsDB::getDefault()),
            "Register new simple material class to the database");

    py::def("_register_material_complex", &registerComplexMaterial, (py::arg("name"), py::arg("material"), py::arg("database")=MaterialsDB::getDefault()),
            "Register new complex material class to the database");

    // Material info
}

}} // namespace plask::python

#include "python_globals.h"
#include "python_material.h"
#include <boost/python/raw_function.hpp>
#include <boost/python/stl_iterator.hpp>
#include <algorithm>

#include <plask/config.h>
#include <plask/utils/string.h>
#include <plask/exceptions.h>
#include <plask/material/mixed.h>
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

    static shared_ptr<Material> __init__(py::tuple args, py::dict kwargs);

    // Here there are overridden methods from Material class

    OmpLockGuard<OmpNestLock> lock() const override {
        return OmpLockGuard<OmpNestLock>(python_omp_lock);
    }

    bool isEqual(const Material& other) const override {
        OmpLockGuard<OmpNestLock> lock(python_omp_lock);
        auto theother = static_cast<const PythonMaterial&>(other);
        py::object oself { py::borrowed(self) },
                   oother { py::object(py::borrowed(theother.self)) };

        if (overriden("__eq__")) {
            return py::call_method<bool>(self, "__eq__", oother);
        }

        return *base == *theother.base &&
                oself.attr("__class__") == oother.attr("__class__") &&
                oself.attr("__dict__") == oother.attr("__dict__");
    }

    std::string name() const override {
        OmpLockGuard<OmpNestLock> lock(python_omp_lock);
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

    std::string str() const override {
        OmpLockGuard<OmpNestLock> lock(python_omp_lock);
        if (overriden("__str__")) {
            return py::call_method<std::string>(self, "__str__");
        }
        else return name();
    }

    Material::ConductivityType condtype() const override {
        OmpLockGuard<OmpNestLock> lock(python_omp_lock);
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

    Material::Kind kind() const override {
        OmpLockGuard<OmpNestLock> lock(python_omp_lock);
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

    double lattC(double T, char x) const override { return call<double>("lattC", &Material::lattC, cache->lattC, T, x); }
    double Eg(double T, double e, char point) const override { return call<double>("Eg", &Material::Eg, cache->Eg, T, e, point); }
    double CB(double T, double e, char point) const override {
        try { return call<double>("CB", &Material::CB, cache->CB, T, e, point); }
        catch (NotImplemented) { return VB(T, e, point, 'H') + Eg(T, e, point); }  // D = µ kB T / e
    }
    double VB(double T, double e, char point, char hole) const  override{ return call<double>("VB", &Material::VB, cache->VB, T, e, point, hole); }
    double Dso(double T, double e) const override { return call<double>("Dso", &Material::Dso, cache->Dso, T, e); }
    double Mso(double T, double e) const override { return call<double>("Mso", &Material::Mso, cache->Mso, T, e); }
    Tensor2<double> Me(double T, double e, char point) const override { return call<Tensor2<double>>("Me", &Material::Me, cache->Me, T, e, point); }
    Tensor2<double> Mhh(double T, double e) const override { return call<Tensor2<double>>("Mhh", &Material::Mhh, cache->Mhh, T, e); }
    Tensor2<double> Mlh(double T, double e) const override { return call<Tensor2<double>>("Mlh", &Material::Mlh, cache->Mlh, T, e); }
    Tensor2<double> Mh(double T, double e) const override { return call<Tensor2<double>>("Mh", &Material::Mh, cache->Mh, T, e); }
    double ac(double T) const override { return call<double>("ac", &Material::ac, cache->ac, T); }
    double av(double T) const override { return call<double>("av", &Material::av, cache->av, T); }
    double b(double T) const override { return call<double>("b", &Material::b, cache->b, T); }
    double d(double T) const override { return call<double>("d", &Material::d, cache->d, T); }
    double c11(double T) const override { return call<double>("c11", &Material::c11, cache->c11, T); }
    double c12(double T) const override { return call<double>("c12", &Material::c12, cache->c12, T); }
    double c44(double T) const override { return call<double>("c44", &Material::c44, cache->c44, T); }
    double eps(double T) const override { return call<double>("eps", &Material::eps, cache->eps, T); }
    double chi(double T, double e, char point) const override { return call<double>("chi", &Material::chi, cache->chi, T, e, point); }
    double Na() const override { return call<double>("Na", &Material::Na, cache->Na); }
    double Nc(double T, double e, char point) const override { return call<double>("Nc", &Material::Nc, cache->Nc, T, e, point); }
    double Nd() const override { return call<double>("Nd", &Material::Nd, cache->Nd); }
    double Nv(double T, double e, char point) const override { return call<double>("Nv", &Material::Nv, cache->Nv, T, e, point); }
    double Ni(double T) const override { return call<double>("Ni", &Material::Ni, cache->Ni, T); }
    double Nf(double T) const override { return call<double>("Nf", &Material::Nf, cache->Nf, T); }
    double EactD(double T) const override { return call<double>("EactD", &Material::EactD, cache->EactD, T); }
    double EactA(double T) const override { return call<double>("EactA", &Material::EactA, cache->EactA, T); }
    Tensor2<double> mob(double T) const override { return call<Tensor2<double>>("mob", &Material::mob, cache->mob, T); }
    Tensor2<double> cond(double T) const override { return call<Tensor2<double>>("cond", &Material::cond, cache->cond, T); }
    double A(double T) const override { return call<double>("A", &Material::A, cache->A, T); }
    double B(double T) const override { return call<double>("B", &Material::B, cache->B, T); }
    double C(double T) const override { return call<double>("C", &Material::C, cache->C, T); }
    double D(double T) const override {
        try { return call<double>("D", &Material::D, cache->D, T); }
        catch (NotImplemented) { return mob(T).c00 * T * 8.6173423e-5; }  // D = µ kB T / e
    }
    Tensor2<double> thermk(double T, double t) const override { return call<Tensor2<double>>("thermk", &Material::thermk, cache->thermk, T, t); }
    double dens(double T) const override { return call<double>("dens", &Material::dens, cache->dens, T); }
    double cp(double T) const override { return call<double>("cp", &Material::cp, cache->cp, T); }
    double nr(double wl, double T, double n) const override { return call<double>("nr", &Material::nr, cache->nr, wl, T, n); }
    double absp(double wl, double T) const override { return call<double>("absp", &Material::absp, cache->absp, wl, T); }
    dcomplex Nr(double wl, double T, double n) const override {
        if (cache->Nr) return *cache->Nr;
        OmpLockGuard<OmpNestLock> lock(python_omp_lock);
        if (overriden("Nr")) {
            return py::call_method<dcomplex>(self, "Nr", wl, T, n);
        }
        if (cache->nr || cache->absp || overriden("nr") || overriden("absp"))
            return dcomplex(call<double>("nr", &Material::nr, cache->nr, wl, T, n), -7.95774715459e-09*call<double>("absp", &Material::absp, cache->absp, wl,T)*wl);
        return base->Nr(wl, T, n);
    }
    Tensor3<dcomplex> NR(double wl, double T, double n) const override {
        if (cache->NR) return *cache->NR;
        OmpLockGuard<OmpNestLock> lock(python_omp_lock);
        if (overriden("NR")) {
            return py::call_method<Tensor3<dcomplex>>(self, "NR", wl, T, n);
        }
        if (cache->Nr || overriden("Nr")) {
            dcomplex nr;
            if (cache->Nr)
                nr = *cache->Nr;
            else {
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

    Tensor2<double> mobe(double T) const override { return call<Tensor2<double>>("mobe", &Material::mobe, cache->mobe, T); }
    Tensor2<double> mobh(double T) const override { return call<Tensor2<double>>("mobh", &Material::mobh, cache->mobh, T); }
    double Ae(double T) const override { return call<double>("Ae", &Material::Ae, cache->Ae, T); }
    double Ah(double T) const override { return call<double>("Ah", &Material::Ah, cache->Ah, T); }
    double Ce(double T) const override { return call<double>("Ce", &Material::Ce, cache->Ce, T); }
    double Ch(double T) const override { return call<double>("Ch", &Material::Ch, cache->Ch, T); }
    double e13(double T) const override { return call<double>("e13", &Material::e13, cache->e13, T); }
    double e15(double T) const override { return call<double>("e15", &Material::e15, cache->e15, T); }
    double e33(double T) const override { return call<double>("e33", &Material::e33, cache->e33, T); }
    double c13(double T) const override { return call<double>("c13", &Material::c13, cache->c13, T); }
    double c33(double T) const override { return call<double>("c33", &Material::c33, cache->c33, T); }
    double Psp(double T) const override { return call<double>("Psp", &Material::Psp, cache->Psp, T); }

    // End of overriden methods

};

std::map<PyObject*, std::unique_ptr<MaterialCache>> PythonMaterial::cacheMap;

/**
 * Base class for Python material constructors
 */
struct PythonMaterialConstructor: public MaterialsDB::MaterialConstructor
{
    py::object material_class;
    MaterialsDB::ProxyMaterialConstructor base_constructor;
    const bool simple;

    PythonMaterialConstructor(const std::string& name, const py::object& cls, const py::object& base, bool simple):
        MaterialsDB::MaterialConstructor(name), material_class(cls), simple(simple)
    {
        if (base == py::object()) return;

        py::extract<std::string> base_str(base);
        if (base_str.check()) {
            base_constructor = MaterialsDB::ProxyMaterialConstructor(base_str);
        } else {
            base_constructor = MaterialsDB::ProxyMaterialConstructor(py::extract<shared_ptr<Material>>(base));
        }
    }

    shared_ptr<Material> operator()(const Material::Composition& composition, Material::DopingAmountType doping_amount_type, double doping_amount) const override
    {
        py::tuple args;
        py::dict kwargs;
        // Composition
        for (auto c : composition) kwargs[c.first] = c.second;
        // Doping information
        if (doping_amount_type !=  Material::NO_DOPING) {
            kwargs[ doping_amount_type == Material::DOPANT_CONCENTRATION ? "dc" : "cc" ] = doping_amount;
        }
        return py::extract<shared_ptr<Material>>(material_class(*args, **kwargs));
    }

    bool isSimple() const override { return simple; }
};

/**
 * Function registering custom simple material class to plask
 * \param name name of the material
 * \param material_class Python class object of the custom material
 * \param base base material specification
 */
void registerSimpleMaterial(const std::string& name, py::object material_class, const py::object& base)
{
    auto constructor = make_shared<PythonMaterialConstructor>(name, material_class, base, true);
    MaterialsDB::getDefault().addSimple(constructor);
    material_class.attr("_factory") = py::object(constructor);
}

/**
 * Function registering custom complex material class to plask
 * \param name name of the material
 * \param material_class Python class object of the custom material
 * \param base base material specification
 */
void registerComplexMaterial(const std::string& name, py::object material_class, const py::object& base)
{
    auto constructor = make_shared<PythonMaterialConstructor>(name, material_class, base, false);
    MaterialsDB::getDefault().addComplex(constructor);
    material_class.attr("_factory") = py::object(constructor);
}

//parse material parameters from full_name and extra parameters in kwargs
static Material::Parameters kwargs2MaterialComposition(const std::string& full_name, const py::dict& kwargs)
{
    Material::Parameters result(full_name, true);

    bool had_doping_key = false;
    py::object cobj;
    try {
        cobj = kwargs["dc"];
        if (result.hasDoping()) throw ValueError("doping or carrier concentrations specified in both full name and argument");
        result.dopingAmountType = Material::DOPANT_CONCENTRATION;
        had_doping_key = true;
    } catch (py::error_already_set) {
        PyErr_Clear();
    }
    try {
        cobj = kwargs["cc"];
        if (had_doping_key) throw ValueError("doping and carrier concentrations specified simultaneously");
        if (result.hasDoping()) throw ValueError("doping or carrier concentrations specified in both full name and argument");
        result.dopingAmountType = Material::CARRIER_CONCENTRATION;
        had_doping_key = true;
    } catch (py::error_already_set) {
        PyErr_Clear();
    }
    if (had_doping_key) {
        if (!result.hasDopantName())
            throw ValueError("%s concentration given for undoped material",
                             (result.dopingAmountType==Material::DOPANT_CONCENTRATION)?"doping":"carrier");
        result.dopingAmount = py::extract<double>(cobj);
    } else {
        if (result.hasDopantName() && !result.hasDoping())
            throw ValueError("dopant specified, but neither doping nor carrier concentrations given correctly");
    }

    py::list keys = kwargs.keys();

    // Test if kwargs contains only doping information
    if (py::len(keys) == int(had_doping_key)) return result;

    if (!result.composition.empty()) throw ValueError("composition specified in both full name and arguments");

    // So, kwargs contains composition
    std::vector<std::string> objects = Material::parseObjectsNames(result.name);
    py::object none;
    // test if only correct objects are given
    for (int i = 0; i < py::len(keys); ++i) {
        std::string k = py::extract<std::string>(keys[i]);
        if (k != "dc" && k != "cc" && std::find(objects.begin(), objects.end(), k) == objects.end()) {
            throw TypeError("'%s' not allowed in material %s", k, result.name);
        }
    }
    // make composition map
    for (auto e: objects) {
        py::object v;
        try {
            v = kwargs[e];
        } catch (py::error_already_set) {
            PyErr_Clear();
        }
        result.composition[e] = (v != none) ? py::extract<double>(v): std::numeric_limits<double>::quiet_NaN();
    }

    return result;
}

shared_ptr<Material> PythonMaterial::__init__(py::tuple args, py::dict kwargs)
{
    int len = py::len(args);

    if (len > 1) {
        throw TypeError("__init__ takes exactly 1 non-keyword arguments (%d given)", len);
    }

    py::object self(args[0]);
    py::object cls = self.attr("__class__");

    shared_ptr<PythonMaterial> ptr;

    if (PyObject_HasAttrString(cls.ptr(), "_factory")) {
        shared_ptr<PythonMaterialConstructor> factory
            = py::extract<shared_ptr<PythonMaterialConstructor>>(cls.attr("_factory"));
        Material::Parameters p = kwargs2MaterialComposition(factory->base_constructor.materialName, kwargs);
        ptr.reset(new PythonMaterial(factory->base_constructor(p.completeComposition(), p.dopingAmountType, p.dopingAmount)));
    } else {
        ptr.reset(new PythonMaterial());
    }

    ptr->self = self.ptr();  // key line !!!

    self.attr("base") = ptr->base;

    // Update cache
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
        CHECK_CACHE(double, Na, "Na")
        CHECK_CACHE(double, Nc, "Nc", 300., 0., "G")
        CHECK_CACHE(double, Nd, "Nd")
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

        CHECK_CACHE(Tensor2<double>, mobe, "mobe", 300.)
        CHECK_CACHE(Tensor2<double>, mobh, "mobh", 300.)
        CHECK_CACHE(double, Ae, "Ae", 300.)
        CHECK_CACHE(double, Ah, "Ah", 300.)
        CHECK_CACHE(double, Ce, "Ce", 300.)
        CHECK_CACHE(double, Ch, "Ch", 300.)
        CHECK_CACHE(double, e13, "e13", 300.)
        CHECK_CACHE(double, e15, "e15", 300.)
        CHECK_CACHE(double, e33, "e33", 300.)
        CHECK_CACHE(double, c13, "c13", 300.)
        CHECK_CACHE(double, c33, "c33", 300.)
        CHECK_CACHE(double, Psp, "Psp", 300.)
    }
    return ptr;
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

    return DB->get(kwargs2MaterialComposition(name, kwargs));
}

py::dict Material__completeComposition(py::dict src, std::string name) {
    py::list keys = src.keys();
    Material::Composition comp;
    py::object none;
    for(int i = 0; i < py::len(keys); ++i) {
        std::string k = py::extract<std::string>(keys[i]);
        if (k != "dc" && k != "cc") {
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
    detail::getPropertyInfo(result, *minfo, MaterialInfo::c13, MaterialInfo::T);
    detail::getPropertyInfo(result, *minfo, MaterialInfo::c33, MaterialInfo::T);
    detail::getPropertyInfo(result, *minfo, MaterialInfo::c44, MaterialInfo::T);
    detail::getPropertyInfo(result, *minfo, MaterialInfo::e13, MaterialInfo::T);
    detail::getPropertyInfo(result, *minfo, MaterialInfo::e15, MaterialInfo::T);
    detail::getPropertyInfo(result, *minfo, MaterialInfo::e33, MaterialInfo::T);
    detail::getPropertyInfo(result, *minfo, MaterialInfo::eps, MaterialInfo::T);
    detail::getPropertyInfo(result, *minfo, MaterialInfo::chi, MaterialInfo::T);
    detail::getPropertyInfo(result, *minfo, MaterialInfo::Na);
    detail::getPropertyInfo(result, *minfo, MaterialInfo::Nc, MaterialInfo::T);
    detail::getPropertyInfo(result, *minfo, MaterialInfo::Nd);
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
             "    lib (str): Library name to load (without an extension).\n",
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
             "Get material of given name and doping.\n\n"
             ":rtype: Material\n"
             )
        .def("__call__", py::raw_function(&MaterialsDB_get), ":rtype: Material\n")
        .add_property("all", &MaterialsDB_list, "List of all materials in the database.")
        .def("__iter__", &MaterialsDB_iter)
        .def("__contains__", &MaterialsDB_contains)
        .def("is_simple", &MaterialsDB::isSimple, py::arg("name"),
             "Return ``True`` if the specified material is a simple one.\n\n"
             "Args:\n"
             "    name (str): material name without doping amount and composition.\n"
             "                (e.g. 'GaAs:Si', 'AlGaAs')."
            )
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
             "Fix incomplete material composition basing on pattern.\n\n"
             "Args:\n"
             "    composition (dict): Dictionary with incomplete composition (i.e. the one\n"
             "                        missing some elements).\n"
             "    name (str): Material name.\n\n"
             "Return:\n"
             "    dict: Dictionary with completed composition.")
        .staticmethod("complete_composition")

        .add_property("name", &Material::name, "Material name (without composition and doping amounts).")

        .add_property("dopant_name", &Material::dopantName, "Dopant material name (part of name after ':', possibly empty).")

        .add_property("name_without_dopant", &Material::nameWithoutDopant, "Material name without dopant (without ':' and part of name after it).")

        .add_property("kind", &Material::kind, "Material kind.")

        .def("__str__", &Material__str__)

        .def("__repr__", &Material__repr__)

        .def("__eq__", (bool(Material::*)(const Material&)const)&Material::operator==)

        .add_property("simple", &Material::isSimple)

        .def("lattC", &Material::lattC, (py::arg("T")=300., py::arg("x")="a"),
             "Get lattice constant [A].\n\n"
             "Args:\n"
             "    T (float): Temperature [K].\n"
             "    x (char): lattice parameter [-].\n")

        .def("Eg", &Material::Eg, (py::arg("T")=300., py::arg("e")=0, py::arg("point")="*"),
             "Get energy gap Eg [eV].\n\n"
             "Args:\n"
             "    T (float): Temperature [K].\n"
             "    e (float): Lateral strain [-].\n"
             "    point (char): Point in the Brillouin zone ('*' means minimum bandgap).\n")

        .def("CB", &Material::CB, (py::arg("T")=300., py::arg("e")=0, py::arg("point")="*"),
             "Get conduction band level CB [eV].\n\n"
             "Args:\n"
             "    T (float): Temperature [K].\n"
             "    e (float): Lateral strain [-].\n"
             "    point (char): Point in the Brillouin zone ('*' means minimum bandgap).\n")

        .def("VB", &Material::VB, (py::arg("T")=300., py::arg("e")=0, py::arg("point")="*", py::arg("hole")='H'),
             "Get valance band level VB [eV].\n\n"
             "Args:\n"
             "    T (float): Temperature [K].\n"
             "    e (float): Lateral strain [-].\n"
             "    point (char): Point in the Brillouin zone ('*' means minimum bandgap).\n"
             "    hole (char): Hole type ('H' or 'L').\n")

        .def("Dso", &Material::Dso, (py::arg("T")=300., py::arg("e")=0),
             "Get split-off energy Dso [eV].\n\n"
             "Args:\n"
             "    T (float): Temperature [K].\n"
             "    e (float): Lateral strain [-].\n")

        .def("Mso", &Material::Mso, (py::arg("T")=300., py::arg("e")=0),
             "Get split-off mass Mso [m₀].\n\n"
             "Args:\n"
             "    T (float): Temperature [K].\n"
             "    e (float): Lateral strain [-].\n")

        .def("Me", &Material::Me, (py::arg("T")=300., py::arg("e")=0, py::arg("point")="*"),
             "Get electron effective mass Me [m₀].\n\n"
             "Args:\n"
             "    T (float): Temperature [K].\n"
             "    e (float): Lateral strain [-].\n"
             "    point (char): Point in the Brillouin zone ('*' means minimum bandgap).\n")

        .def("Mhh", &Material::Mhh, (py::arg("T")=300., py::arg("e")=0),
             "Get heavy hole effective mass Mhh [m₀].\n\n"
             "Args:\n"
             "    T (float): Temperature [K].\n"
             "    e (float): Lateral strain [-].\n")

        .def("Mlh", &Material::Mlh, (py::arg("T")=300., py::arg("e")=0),
             "Get light hole effective mass Mlh [m₀].\n\n"
             "Args:\n"
             "    T (float): Temperature [K].\n"
             "    e (float): Lateral strain [-].\n")

        .def("Mh", &Material::Mh, (py::arg("T")=300., py::arg("e")=0),
             "Get hole effective mass Mh [m₀].\n\n"
             "Args:\n"
             "    T (float): Temperature [K].\n"
             "    e (float): Lateral strain [-].\n")

        .def("ac", &Material::ac, (py::arg("T")=300.),
             "Get hydrostatic deformation potential for the conduction band ac [eV].\n\n"
             "Args:\n"
             "    T (float): Temperature [K].\n")

        .def("av", &Material::av, (py::arg("T")=300.),
             "Get hydrostatic deformation potential for the valence band av [eV].\n\n"
             "Args:\n"
             "    T (float): Temperature [K].\n")

        .def("b", &Material::b, (py::arg("T")=300.),
             "Get shear deformation potential b [eV].\n\n"
             "Args:\n"
             "    T (float): Temperature [K].\n")

        .def("d", &Material::d, (py::arg("T")=300.),
             "Get shear deformation potential d [eV].\n\n"
             "Args:\n"
             "    T (float): Temperature [K].\n")

        .def("c11", &Material::c11, (py::arg("T")=300.),
             "Get elastic constant c₁₁ [GPa].\n\n"
             "Args:\n"
             "    T (float): Temperature [K].\n")

        .def("c12", &Material::c12, (py::arg("T")=300.),
             "Get elastic constant c₁₂ [GPa].\n\n"
             "Args:\n"
             "    T (float): Temperature [K].\n")

        .def("c13", &Material::c13, (py::arg("T")=300.),
             "Get elastic constant c₁₃ [GPa].\n\n"
             "Args:\n"
             "    T (float): Temperature [K].\n")

        .def("c33", &Material::c33, (py::arg("T")=300.),
             "Get elastic constant c₃₃ [GPa].\n\n"
             "Args:\n"
             "    T (float): Temperature [K].\n")

        .def("c44", &Material::c44, (py::arg("T")=300.),
             "Get elastic constant c₄₄ [GPa].\n\n"
             "Args:\n"
             "    T (float): Temperature [K].\n")

        .def("e13", &Material::c44, (py::arg("T")=300.),
             "Get piezoelectric constant e₁₃ [C/m²].\n\n"
             "Args:\n"
             "    T (float): Temperature [K].\n")

        .def("e15", &Material::c44, (py::arg("T")=300.),
             "Get piezoelectric constant e₁₅ [C/m²].\n\n"
             "Args:\n"
             "    T (float): Temperature [K].\n")

        .def("e33", &Material::c44, (py::arg("T")=300.),
             "Get piezoelectric constant e₃₃ [C/m²].\n\n"
             "Args:\n"
             "    T (float): Temperature [K].\n")

        .def("Psp", &Material::Psp, (py::arg("T")=300.),
             "Get Spontaneous polarization P [C/m²].\n\n"
             "Args:\n"
             "    T (float): Temperature [K].\n")

        .def("eps", &Material::eps, (py::arg("T")=300.),
             "Get dielectric constant ε [-].\n\n"
             "Args:\n"
             "    T (float): Temperature [K].\n")

        .def("chi", &Material::chi, (py::arg("T")=300., py::arg("e")=0, py::arg("point")="*"),
             "Get electron affinity Chi [eV].\n\n"
             "Args:\n"
             "    T (float): Temperature [K].\n"
             "    e (float): Lateral strain [-].\n"
             "    point (char): Point in the Brillouin zone ('*' means minimum bandgap).\n")

        .def("Na", &Material::Na,
                 "Get acceptor concentration Na [1/m³].\n\n"
                 "Args:-\n")

        .def("Nc", &Material::Nc, (py::arg("T")=300., py::arg("e")=0, py::arg("point")="*"),
             "Get effective density of states in the conduction band Nc [1/m³].\n\n"
             "Args:\n"
             "    T (float): Temperature [K].\n"
             "    e (float): Lateral strain [-].\n"
             "    point (char): Point in the Brillouin zone ('*' means minimum bandgap).\n")

        .def("Nd", &Material::Nd,
             "Get donor concentration Nd [1/m³].\n\n"
             "Args:-\n")

        .def("Nv", &Material::Nv, (py::arg("T")=300., py::arg("e")=0, py::arg("point")="*"),
             "Get effective density of states in the valence band Nv [1/m³].\n\n"
             "Args:\n"
             "    T (float): Temperature [K].\n"
             "    e (float): Lateral strain [-].\n"
             "    point (char): Point in the Brillouin zone ('*' means minimum bandgap).\n")

        .def("Ni", &Material::Ni, (py::arg("T")=300.),
             "Get intrinsic carrier concentration Ni [1/m³].\n\n"
             "Args:\n"
             "    T (float): Temperature [K].\n")

        .def("Nf", &Material::Nf, (py::arg("T")=300.),
             "Get free carrier concentration N [1/m³].\n\n"
             "Args:\n"
             "    T (float): Temperature [K].\n")

        .def("EactD", &Material::EactD, (py::arg("T")=300.),
             "Get donor ionisation energy EactD [eV].\n\n"
             "Args:\n"
             "    T (float): Temperature [K].\n")

        .def("EactA", &Material::EactA, (py::arg("T")=300.),
             "Get acceptor ionisation energy EactA [eV].\n\n"
             "Args:\n"
             "    T (float): Temperature [K].\n")

        .def("mob", &Material::mob, (py::arg("T")=300.),
             "Get majority carriers mobility [cm²/(V s)].\n\n"
             "Args:\n"
             "    T (float): Temperature [K].\n")

        .def("mobe", &Material::mobe, (py::arg("T")=300.),
             "Get electron mobility [cm²/(V s)].\n\n"
             "Args:\n"
             "    T (float): Temperature [K].\n")

        .def("mobh", &Material::mobh, (py::arg("T")=300.),
             "Get hole mobility [cm²/(V s)].\n\n"
             "Args:\n"
             "    T (float): Temperature [K].\n")

        .def("cond", &Material::cond, (py::arg("T")=300.),
             "Get electrical conductivity Sigma [S/m].\n\n"
             "Args:\n"
             "    T (float): Temperature [K].\n")

        .add_property("condtype", &Material::condtype,
             "Electrical conductivity type.")

        .def("A", &Material::A, (py::arg("T")=300.),
             "Get monomolecular recombination coefficient A [1/s].\n\n"
             "Args:\n"
             "    T (float): Temperature [K].\n")

        .def("Ae", &Material::Ae, (py::arg("T")=300.),
             "Get monomolecular recombination coefficient A [1/s] for electrons.\n\n"
             "Args:\n"
             "    T (float): Temperature [K].\n")

        .def("Ah", &Material::Ah, (py::arg("T")=300.),
             "Get monomolecular recombination coefficient A [1/s] for holes.\n\n"
             "Args:\n"
             "    T (float): Temperature [K].\n")

        .def("B", &Material::B, (py::arg("T")=300.),
             "Get radiative recombination coefficient B [cm³/s].\n\n"
             "Args:\n"
             "    T (float): Temperature [K].\n")

        .def("C", &Material::C, (py::arg("T")=300.),
             "Get Auger recombination coefficient C [cm⁶/s].\n\n"
             "Args:\n"
             "    T (float): Temperature [K].\n")

        .def("Ce", &Material::Ce, (py::arg("T")=300.),
             "Get Auger recombination coefficient C [cm⁶/s] for electrons.\n\n"
             "Args:\n"
             "    T (float): Temperature [K].\n")

        .def("Ch", &Material::Ch, (py::arg("T")=300.),
             "Get Auger recombination coefficient C [cm⁶/s] for holes.\n\n"
             "Args:\n"
             "    T (float): Temperature [K].\n")

        .def("D", &Material::D, (py::arg("T")=300.),
             "Get ambipolar diffusion coefficient D [cm²/s].\n\n"
             "Args:\n"
             "    T (float): Temperature [K].\n")

        .def("thermk", &Material::thermk, (py::arg("T")=300., py::arg("h")=INFINITY),
             "Get thermal conductivity [W/(m K)].\n\n"
             "Args:\n"
             "    T (float): Temperature [K].\n"
             "    h (float): Layer thickness [µm] [-].\n")

        .def("dens", &Material::dens, (py::arg("T")=300.),
             "Get density [kg/m³].\n\n"
             "Args:\n"
             "    T (float): Temperature [K].\n")

        .def("cp", &Material::cp, (py::arg("T")=300.),
             "Get specific heat at constant pressure [J/(kg K)].\n\n"
             "Args:\n"
             "    T (float): Temperature [K].\n")

        .def("nr", &Material::nr, (py::arg("wl"), py::arg("T")=300., py::arg("n")=0.),
             "Get refractive index nr [-].\n\n"
             "Args:\n"
             "    wl (float): Wavelength [nm].\n"
             "    T (float): Temperature [K].\n"
             "    n (float): Injected carriers concentration [1/cm³].\n")

        .def("absp", &Material::absp, (py::arg("wl"), py::arg("T")=300.),
             "Get absorption coefficient alpha [1/cm].\n\n"
             "Args:\n"
             "    wl (float): Wavelength [nm].\n"
             "    T (float): Temperature [K].\n")

        .def("Nr", &Material::Nr, (py::arg("wl"), py::arg("T")=300., py::arg("n")=0.),
             "Get complex refractive index Nr [-].\n\n"
             "Args:\n"
             "    wl (float): Wavelength [nm].\n"
             "    T (float): Temperature [K].\n"
             "    n (float): Injected carriers concentration [1/cm³].\n")

        .def("NR", &Material::NR, (py::arg("wl"), py::arg("T")=300., py::arg("n")=0.),
             "Get complex refractive index tensor Nr [-].\n\n"
             "Args:\n"
             "    wl (float): Wavelength [nm].\n"
             "    T (float): Temperature [K].\n"
             "    n (float): Injected carriers concentration [1/cm³].\n")
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

    py::class_<PythonMaterialConstructor, shared_ptr<PythonMaterialConstructor>, boost::noncopyable>
        ("_Constructor", py::no_init);

    py::def("_register_material_simple", &registerSimpleMaterial,
            (py::arg("name"), "material", "base"),
            "Register new simple material class to the database");

    py::def("_register_material_complex", &registerComplexMaterial,
            (py::arg("name"), "material", "base"),
            "Register new complex material class to the database");

    // Material info
}

}} // namespace plask::python

/*
 * This file is part of PLaSK (https://plask.app) by Photonics Group at TUL
 * Copyright (c) 2022 Lodz University of Technology
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 */
#include "python_globals.hpp"
#include <boost/python/raw_function.hpp>
#include <boost/python/stl_iterator.hpp>
#include <algorithm>

#include <plask/config.hpp>
#include "plask/utils/string.hpp"
#include "plask/exceptions.hpp"
#include "plask/material/mixed.hpp"
#include "plask/material/db.hpp"
#include "plask/material/info.hpp"

#include "python_util/raw_constructor.hpp"

namespace plask { namespace python {

namespace detail {

    struct StringFromMaterial {

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
        if (!PyUnicode_Check(obj)) return 0;
        return obj;
    }

    static void construct(PyObject* obj, boost::python::converter::rvalue_from_python_stage1_data* data) {
        std::string value = py::extract<std::string>(obj);

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
class PythonMaterial: public MaterialWithBase, Overriden<Material>
{
    static std::map<PyObject*, std::unique_ptr<MaterialCache>> cacheMap;

  public:
    MaterialCache* cache;

  private:
    template <typename R, typename... Args>
    inline R call_method(const char* name, Args... args) const {
        py::object result = py::call_method<py::object>(self, name, args...);
        try {
            return py::extract<R>(result);
        } catch (py::error_already_set&) {
            std::string cls_name;
            try {
                cls_name = py::extract<std::string>(py::object(py::borrowed(self)).attr("__class__").attr("__name__"));
            } catch (py::error_already_set&) {
                throw TypeError(u8"Cannot convert return value of method '{}' in unknown material class to correct type", name);
            }
            throw TypeError(u8"Cannot convert return value of method '{}' in material class '{}' to correct type", name, cls_name);
        }
    }

    template <typename R, typename F, typename... Args>
    inline R call(const char* name, F f, const plask::optional<R>& cached, Args&&... args) const {
        if (cached) return *cached;
        OmpLockGuard<OmpNestLock> lock(python_omp_lock);
        if (overriden_no_recursion(name)) {
            return call_method<R>(name, std::forward<Args>(args)...);
        }
        return ((*base).*f)(std::forward<Args>(args)...);
    }

    template <typename R, typename... Args>
    inline R call_override(const char* name, const plask::optional<R>& cached, Args&&... args) const {
        if (cached) return *cached;
        OmpLockGuard<OmpNestLock> lock(python_omp_lock);
        if (overriden_no_recursion(name)) {
            return call_method<R>(name, std::forward<Args>(args)...);
        }
        throw MaterialMethodNotImplemented(this->name(), name);
    }

  public:
    Material::Parameters params;

    PythonMaterial(const py::object& self) { this->self = self.ptr(); }

    static shared_ptr<Material> __init__(const py::tuple& args, const py::dict& kwargs);

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
            return call_method<bool>("__eq__", oother);
        }

        return *base == *theother.base &&
                oself.attr("__class__") == oother.attr("__class__") &&
                oself.attr("__dict__") == oother.attr("__dict__") &&
                doping() == theother.doping() && params.composition == theother.params.composition;
    }

    std::string name() const override {
        OmpLockGuard<OmpNestLock> lock(python_omp_lock);
        py::object cls = py::object(py::borrowed(self)).attr("__class__");
        py::object oname;
        std::string name;
        try {
            oname = cls.attr("__dict__")["name"];
            name = py::extract<std::string>(oname);
        } catch (py::error_already_set&) {
            PyErr_Clear();
            oname = cls.attr("__name__");
            name = py::extract<std::string>(oname);
        }
        return name;
    }

    std::string str() const override {
        OmpLockGuard<OmpNestLock> lock(python_omp_lock);
        if (overriden("__str__")) {
            return call_method<std::string>("__str__");
        }
        else {
            return params.str();
        }
    }

    double doping() const override {
        if (isnan(params.doping)) {
            if (base)
                return base->doping();
            else
                return 0.;
        } else
            return params.doping;
    }

    Composition composition() const override {
        return params.composition;
    }

    Material::ConductivityType condtype() const override {
        OmpLockGuard<OmpNestLock> lock(python_omp_lock);
        py::object cls = py::object(py::borrowed(self)).attr("__class__");
        py::object octype;
        try {
            octype = cls.attr("__dict__")["condtype"];
        } catch (py::error_already_set&) {
            PyErr_Clear();
            return base->condtype();
        }
        return py::extract<Material::ConductivityType>(octype);
    }

//     double doping() const override {
//     }

    Material::Kind kind() const override {
        OmpLockGuard<OmpNestLock> lock(python_omp_lock);
        py::object cls = py::object(py::borrowed(self)).attr("__class__");
        py::object okind;
        try {
            okind = cls.attr("__dict__")["kind"];
        } catch (py::error_already_set&) {
            PyErr_Clear();
            return base->kind();
        }
        return py::extract<Material::Kind>(okind);
    }

    double lattC(double T, char x) const override { return call<double>("lattC", &Material::lattC, cache->lattC, T, x); }
    double Eg(double T, double e, char point) const override {
        try { return call_override<double>("Eg", cache->Eg, T, e, point); }
        catch (NotImplemented&) {
            if ((cache->VB || overriden("VB")) && (cache->CB || overriden("CB"))) {
                try {
                    return call_override<double>("CB", cache->CB, T, e, point) -
                           call_override<double>("VB", cache->VB, T, e, point, "H");
                }
                catch (NotImplemented&) {}
            }
            return base->Eg(T, e, point);
        }
    }
    double CB(double T, double e, char point) const override {
        try { return call_override<double>("CB", cache->CB, T, e, point); }
        catch (NotImplemented&) {
            if (cache->VB || cache->Eg || overriden("VB") || overriden("Eg")) {
                try { return VB(T, e, point, 'H') + Eg(T, e, point); }
                catch (NotImplemented&) {}
            }
            return base->CB(T, e, point);
        }
    }
    double VB(double T, double e, char point, char hole) const override {
        try { return call_override<double>("VB", cache->VB, T, e, point, hole); }
        catch (NotImplemented&) {
            if (cache->CB || overriden("CB")) {
                try { return call_override<double>("CB", cache->CB, T, e, point) - Eg(T, e, point); }
                catch (NotImplemented&) {}
            }
            return base->VB(T, e, point, hole);
        }
    }
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
    double Nd() const override { return call<double>("Nd", &Material::Nd, cache->Nd); }
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
        try { return call_override<double>("D", cache->D, T); }
        catch (NotImplemented&) {
            try { return mob(T).c00 * T * 8.6173423e-5; }   // D = µ kB T / e
            catch (NotImplemented&) { return base->D(T); }
        }
    }
    Tensor2<double> thermk(double T, double t) const override { return call<Tensor2<double>>("thermk", &Material::thermk, cache->thermk, T, t); }
    double dens(double T) const override { return call<double>("dens", &Material::dens, cache->dens, T); }
    double cp(double T) const override { return call<double>("cp", &Material::cp, cache->cp, T); }
    double nr(double lam, double T, double n) const override { return call<double>("nr", &Material::nr, cache->nr, lam, T, n); }
    double absp(double lam, double T) const override { return call<double>("absp", &Material::absp, cache->absp, lam, T); }
    dcomplex Nr(double lam, double T, double n) const override {
        try { return call_override<dcomplex>("Nr", cache->Nr, lam, T, n); }
        catch (NotImplemented&) {
            try { return dcomplex(                     call<double>("nr", &Material::nr, cache->nr, lam, T, n),
                                  -7.95774715459e-09 * call<double>("absp", &Material::absp, cache->absp, lam,T) * lam
                                 ); }
            catch (NotImplemented&) { return base->Nr(lam, T, n); }
        }
    }
    Tensor3<dcomplex> NR(double lam, double T, double n) const override {
        try { return call_override<Tensor3<dcomplex>>("NR", cache->NR, lam, T, n); }
        catch (NotImplemented&) {
            try { dcomplex nr = Nr(lam, T, n); return Tensor3<dcomplex>(nr, nr, nr, 0.); }
            catch (NotImplemented&) { return base->NR(lam, T, n); }
        }
    }
    Tensor2<double> mobe(double T) const override { return call<Tensor2<double>>("mobe", &Material::mobe, cache->mobe, T); }
    Tensor2<double> mobh(double T) const override { return call<Tensor2<double>>("mobh", &Material::mobh, cache->mobh, T); }
    double taue(double T) const override { return call<double>("taue", &Material::taue, cache->taue, T); }
    double tauh(double T) const override { return call<double>("tauh", &Material::tauh, cache->tauh, T); }
    double Ce(double T) const override { return call<double>("Ce", &Material::Ce, cache->Ce, T); }
    double Ch(double T) const override { return call<double>("Ch", &Material::Ch, cache->Ch, T); }
    double e13(double T) const override { return call<double>("e13", &Material::e13, cache->e13, T); }
    double e15(double T) const override { return call<double>("e15", &Material::e15, cache->e15, T); }
    double e33(double T) const override { return call<double>("e33", &Material::e33, cache->e33, T); }
    double c13(double T) const override { return call<double>("c13", &Material::c13, cache->c13, T); }
    double c33(double T) const override { return call<double>("c33", &Material::c33, cache->c33, T); }
    double Psp(double T) const override { return call<double>("Psp", &Material::Psp, cache->Psp, T); }
    double y1() const override { return call<double>("y1", &Material::y1, cache->y1); }
    double y2() const override { return call<double>("y2", &Material::y2, cache->y2); }
    double y3() const override { return call<double>("y3", &Material::y3, cache->y3); }

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
    const bool alloy;

    PythonMaterialConstructor(const std::string& name, const py::object& cls, const py::object& base, bool alloy):
        MaterialsDB::MaterialConstructor(name), material_class(cls), alloy(alloy)
    {
        if (base.is_none()) return;

        py::extract<std::string> base_str(base);
        if (base_str.check()) {
            base_constructor = MaterialsDB::ProxyMaterialConstructor(base_str);
        } else {
            base_constructor = MaterialsDB::ProxyMaterialConstructor(py::extract<shared_ptr<Material>>(base));
        }
    }

    shared_ptr<Material> operator()(const Material::Composition& composition, double doping) const override
    {
        OmpLockGuard<OmpNestLock> lock(python_omp_lock);
        py::tuple args;
        py::dict kwargs;
        // Composition
        for (auto c : composition) kwargs[c.first] = c.second;
        // Doping information
        if (!isnan(doping) && (doping != 0 || materialName.find(':') != std::string::npos)) kwargs["doping"] = doping;
        py::object omaterial = material_class(*args, **kwargs);;
        return py::extract<shared_ptr<Material>>(omaterial);
    }

    bool isAlloy() const override { return alloy; }
};

static void setMaterialInfo(const std::string& material_name, PyObject* class_object)
{
    MaterialInfo& info = MaterialsDB::getDefault().info.add(material_name);

    #define UPDATE_INFO(meth) if (PyType_Check(class_object) && reinterpret_cast<PyTypeObject*>(class_object)->tp_dict) { \
        if (PyObject* method_object = PyDict_GetItemString(reinterpret_cast<PyTypeObject*>(class_object)->tp_dict, BOOST_PP_STRINGIZE(meth))) { \
            if (PyObject* docstring_object = PyObject_GetAttrString(method_object, "__doc__")) { \
                py::extract<std::string> docstring(docstring_object); \
                if (docstring.check()) info(plask::MaterialInfo::meth).set(docstring()); \
            } \
        } \
    }

    UPDATE_INFO(lattC);
    UPDATE_INFO(Eg);
    UPDATE_INFO(CB);
    UPDATE_INFO(VB);
    UPDATE_INFO(Dso);
    UPDATE_INFO(Mso);
    UPDATE_INFO(Me);
    UPDATE_INFO(Mhh);
    UPDATE_INFO(Mlh);
    UPDATE_INFO(Mh);
    UPDATE_INFO(ac);
    UPDATE_INFO(av);
    UPDATE_INFO(b);
    UPDATE_INFO(d);
    UPDATE_INFO(c11);
    UPDATE_INFO(c12);
    UPDATE_INFO(c44);
    UPDATE_INFO(eps);
    UPDATE_INFO(chi);
    UPDATE_INFO(Na);
    UPDATE_INFO(Nd);
    UPDATE_INFO(Ni);
    UPDATE_INFO(Nf);
    UPDATE_INFO(EactD);
    UPDATE_INFO(EactA);
    UPDATE_INFO(mob);
    UPDATE_INFO(cond);
    UPDATE_INFO(A);
    UPDATE_INFO(B);
    UPDATE_INFO(C);
    UPDATE_INFO(D);
    UPDATE_INFO(thermk);
    UPDATE_INFO(dens);
    UPDATE_INFO(cp);
    UPDATE_INFO(nr);
    UPDATE_INFO(absp);
    UPDATE_INFO(Nr);
    UPDATE_INFO(NR);
    UPDATE_INFO(mobe);
    UPDATE_INFO(mobh);
    UPDATE_INFO(taue);
    UPDATE_INFO(tauh);
    UPDATE_INFO(Ce);
    UPDATE_INFO(Ch);
    UPDATE_INFO(e13);
    UPDATE_INFO(e15);
    UPDATE_INFO(e33);
    UPDATE_INFO(c13);
    UPDATE_INFO(c33);
    UPDATE_INFO(Psp);
    UPDATE_INFO(y1);
    UPDATE_INFO(y2);
    UPDATE_INFO(y3);
}

/**
 * Function registering custom simple material class to plask
 * \param name name of the material
 * \param material_class Python class object of the custom material
 * \param base base material specification
 */
void registerSimpleMaterial(const std::string& name, py::object material_class, const py::object& base)
{
    auto constructor = plask::make_shared<PythonMaterialConstructor>(name, material_class, base, false);
    MaterialsDB::getDefault().addSimple(constructor);
    material_class.attr("_factory") = py::object(constructor);
    material_class.attr("simple") = true;
    setMaterialInfo(name, material_class.ptr());
}

/**
 * Function registering custom alloy material class to plask
 * \param name name of the material
 * \param material_class Python class object of the custom material
 * \param base base material specification
 */
void registerAlloyMaterial(const std::string& name, py::object material_class, const py::object& base)
{
    auto constructor = plask::make_shared<PythonMaterialConstructor>(name, material_class, base, true);
    MaterialsDB::getDefault().addAlloy(constructor);
    material_class.attr("_factory") = py::object(constructor);
    material_class.attr("simple") = false;
    setMaterialInfo(name, material_class.ptr());
}

// Parse material parameters from full_name and extra parameters in kwargs
static Material::Parameters kwargs2MaterialComposition(const std::string& full_name, const py::dict& kwargs)
{
    Material::Parameters result(full_name, true);

    bool had_doping_key = false;
    py::object cobj;
    try {
        cobj = kwargs["doping"];
        if (result.hasDoping()) throw ValueError(u8"Doping concentrations specified in both full name and argument");
        had_doping_key = true;
    } catch (py::error_already_set&) {
        PyErr_Clear();
    }
    if (had_doping_key) {
        if (!result.hasDopantName()) throw ValueError(u8"Doping concentration given for undoped material");
        result.doping = py::extract<double>(cobj);
    } else {
        if (result.hasDopantName() && !result.hasDoping())
            throw ValueError(u8"Dopant specified, but doping concentrations not given correctly");
    }

    py::list keys = kwargs.keys();

    // Test if kwargs contains only doping information
    if (py::len(keys) == int(had_doping_key)) return result;

    if (!result.composition.empty()) throw ValueError(u8"composition specified in both full name and arguments");

    // So, kwargs contains composition
    std::vector<std::string> objects = Material::parseObjectsNames(result.name);
    py::object none;
    // test if only correct objects are given
    for (int i = 0; i < py::len(keys); ++i) {
        std::string k = py::extract<std::string>(keys[i]);
        if (k != "doping" && std::find(objects.begin(), objects.end(), k) == objects.end()) {
            std::string name = result.name;
            if (result.label != "") name += "_" + result.label;
            if (result.dopant != "") name += ":" + result.dopant;
            throw TypeError(u8"'{}' not allowed in material {}", k, name);
        }
    }
    // make composition map
    for (auto e: objects) {
        py::object v;
        try {
            v = kwargs[e];
        } catch (py::error_already_set&) {
            PyErr_Clear();
        }
        result.composition[e] = (v != none) ? py::extract<double>(v): std::numeric_limits<double>::quiet_NaN();
    }

    return result;
}

namespace detail {
    #define MAKE_CACHED_VALUE_GETTER(Type, param) \
    Type Cached##param##Getter(const py::tuple& args, const py::dict&) { \
        const PythonMaterial* self(static_cast<const PythonMaterial*>(py::extract<const Material*>(args[0])())); \
        return *self->cache->param; \
    } \
    py::object Cached##param##GetterObj;

    MAKE_CACHED_VALUE_GETTER(double, lattC)
    MAKE_CACHED_VALUE_GETTER(double, Eg)
    MAKE_CACHED_VALUE_GETTER(double, CB)
    MAKE_CACHED_VALUE_GETTER(double, VB)
    MAKE_CACHED_VALUE_GETTER(double, Dso)
    MAKE_CACHED_VALUE_GETTER(double, Mso)
    MAKE_CACHED_VALUE_GETTER(Tensor2<double>, Me)
    MAKE_CACHED_VALUE_GETTER(Tensor2<double>, Mhh)
    MAKE_CACHED_VALUE_GETTER(Tensor2<double>, Mlh)
    MAKE_CACHED_VALUE_GETTER(Tensor2<double>, Mh)
    MAKE_CACHED_VALUE_GETTER(double, ac)
    MAKE_CACHED_VALUE_GETTER(double, av)
    MAKE_CACHED_VALUE_GETTER(double, b)
    MAKE_CACHED_VALUE_GETTER(double, d)
    MAKE_CACHED_VALUE_GETTER(double, c11)
    MAKE_CACHED_VALUE_GETTER(double, c12)
    MAKE_CACHED_VALUE_GETTER(double, c44)
    MAKE_CACHED_VALUE_GETTER(double, eps)
    MAKE_CACHED_VALUE_GETTER(double, chi)
    MAKE_CACHED_VALUE_GETTER(double, Na)
    MAKE_CACHED_VALUE_GETTER(double, Nd)
    MAKE_CACHED_VALUE_GETTER(double, Ni)
    MAKE_CACHED_VALUE_GETTER(double, Nf)
    MAKE_CACHED_VALUE_GETTER(double, EactD)
    MAKE_CACHED_VALUE_GETTER(double, EactA)
    MAKE_CACHED_VALUE_GETTER(Tensor2<double>, mob)
    MAKE_CACHED_VALUE_GETTER(Tensor2<double>, cond)
    MAKE_CACHED_VALUE_GETTER(double, A)
    MAKE_CACHED_VALUE_GETTER(double, B)
    MAKE_CACHED_VALUE_GETTER(double, C)
    MAKE_CACHED_VALUE_GETTER(double, D)
    MAKE_CACHED_VALUE_GETTER(Tensor2<double>, thermk)
    MAKE_CACHED_VALUE_GETTER(double, dens)
    MAKE_CACHED_VALUE_GETTER(double, cp)
    MAKE_CACHED_VALUE_GETTER(double, nr)
    MAKE_CACHED_VALUE_GETTER(double, absp)
    MAKE_CACHED_VALUE_GETTER(dcomplex, Nr)
    MAKE_CACHED_VALUE_GETTER(Tensor3<dcomplex>, NR)
    MAKE_CACHED_VALUE_GETTER(Tensor2<double>, mobe)
    MAKE_CACHED_VALUE_GETTER(Tensor2<double>, mobh)
    MAKE_CACHED_VALUE_GETTER(double, taue)
    MAKE_CACHED_VALUE_GETTER(double, tauh)
    MAKE_CACHED_VALUE_GETTER(double, Ce)
    MAKE_CACHED_VALUE_GETTER(double, Ch)
    MAKE_CACHED_VALUE_GETTER(double, e13)
    MAKE_CACHED_VALUE_GETTER(double, e15)
    MAKE_CACHED_VALUE_GETTER(double, e33)
    MAKE_CACHED_VALUE_GETTER(double, c13)
    MAKE_CACHED_VALUE_GETTER(double, c33)
    MAKE_CACHED_VALUE_GETTER(double, Psp)
    MAKE_CACHED_VALUE_GETTER(double, y1)
    MAKE_CACHED_VALUE_GETTER(double, y2)
    MAKE_CACHED_VALUE_GETTER(double, y3)
}

#if PY_VERSION_HEX >= 0x030b0000
    #define IF_CO_VARNAMES \
        PyObject* co_varnames = PyCode_GetVarnames(code); \
        bool __co_varnames = co_varnames && PyTuple_GET_SIZE(co_varnames) == 0; \
        Py_XDECREF(co_varnames); \
        if (__co_varnames)
#else
    #define IF_CO_VARNAMES if (code->co_varnames && PyTuple_GET_SIZE(code->co_varnames) == 0)
#endif


shared_ptr<Material> PythonMaterial::__init__(const py::tuple& args, const py::dict& kwargs)
{
    auto len = py::len(args);

    py::object self(args[0]);
    py::object cls = self.attr("__class__");

    shared_ptr<PythonMaterial> ptr(new PythonMaterial(self));
    ptr->params = kwargs2MaterialComposition(ptr->name(), kwargs);
    ptr->params.composition = ptr->params.completeComposition();

    if (PyObject_HasAttrString(cls.ptr(), "_factory")) {
        if (len > 1) {
            throw TypeError(u8"__init__ for registered materials takes exactly 1 non-keyword arguments ({:d} given)", len);
        }
        shared_ptr<PythonMaterialConstructor> factory
            = py::extract<shared_ptr<PythonMaterialConstructor>>(cls.attr("_factory"));
        ptr->base = factory->base_constructor(ptr->params.composition, ptr->params.doping);
    } else {
        if (len > 2)
            throw TypeError(u8"__init__ for unregistered material takes at most 2 non-keyword arguments ({:d} given)", len);
        if (len > 1) {
            ptr->base = py::extract<shared_ptr<Material>>(args[1]);
            ptr->params.doping = ptr->base->doping();
            ptr->params.dopant = ptr->base->dopant();
            ptr->params.composition = ptr->base->composition();
        }
        else
            ptr->base.reset(new GenericMaterial());
    }

    // Update cache
    auto found = cacheMap.find(cls.ptr());
    if (found != cacheMap.end())
        ptr->cache = found->second.get();
    else {
        std::string cls_name = py::extract<std::string>(cls.attr("__name__"));
        // MaterialCache* cache = cacheMap.emplace(cls.ptr(), std::unique_ptr<MaterialCache>(new MaterialCache)).first->second.get();
        MaterialCache* cache = (cacheMap[cls.ptr()] = std::unique_ptr<MaterialCache>(new MaterialCache)).get();
        ptr->cache = cache;
        #define CHECK_CACHE(Type, param, ...) \
        if (py::object(cls.attr(BOOST_PP_STRINGIZE(param))).ptr() == detail::Cached##param##GetterObj.ptr()) { \
            py::stl_input_iterator<py::object> begin(cls.attr("__mro__")), end; \
            begin++; /* ignore self */\
            for (auto i = begin; i != end; i++) { \
                auto f = cacheMap.find(i->ptr()); \
                if (f != cacheMap.end() && f->second->param) { \
                    cache->param.reset(*f->second->param); \
                    break; \
                } \
            } \
        } else if (PyObject_HasAttrString(self.ptr(), BOOST_PP_STRINGIZE(param))) { \
            py::object method = py::object(self.attr(BOOST_PP_STRINGIZE(param))); \
            if (PyFunction_Check(method.ptr())) { \
                if (PyCodeObject* code = reinterpret_cast<PyCodeObject*>(PyFunction_GetCode(method.ptr()))) { \
                    IF_CO_VARNAMES { \
                        writelog(LOG_DEBUG, "Caching parameter '" BOOST_PP_STRINGIZE(param) "' in material class '{}'", cls_name); \
                        try { \
                            py::object value = method(); \
                            cache->param.reset(py::extract<Type>(value)); \
                        } catch (py::error_already_set&) { \
                            throw TypeError("Cannot convert return value of static method '" BOOST_PP_STRINGIZE(param) "()' in material class '{}' to correct type", cls_name); \
                        } \
                        py::setattr(cls, py::object(BOOST_PP_STRINGIZE(param)), detail::Cached##param##GetterObj); \
                    } else { \
                        throw TypeError("Method '" BOOST_PP_STRINGIZE(param) "()' in material class '{}' can be static only if it takes no parameters", cls_name); \
                    } \
                } \
            } else if (!PyMethod_Check(method.ptr())) { \
                writelog(LOG_DEBUG, "Caching parameter '" BOOST_PP_STRINGIZE(param) "' in material class '{}'", cls_name); \
                try { \
                    cache->param.reset(py::extract<Type>(method)); \
                } catch (py::error_already_set&) { \
                    throw TypeError("Cannot convert static parameter '" BOOST_PP_STRINGIZE(param) "' in material class '{}' to correct type", cls_name); \
                } \
                py::setattr(cls, py::object(BOOST_PP_STRINGIZE(param)), detail::Cached##param##GetterObj); \
            } \
        }
        CHECK_CACHE(double, lattC, 300., py::object())
        CHECK_CACHE(double, Eg, 300., 0., "*")
        CHECK_CACHE(double, CB, 300., 0., "*")
        CHECK_CACHE(double, VB, 300., 0., "*", "H")
        CHECK_CACHE(double, Dso, 300., 0.)
        CHECK_CACHE(double, Mso, 300., 0.)
        CHECK_CACHE(Tensor2<double>, Me, 300., 0.)
        CHECK_CACHE(Tensor2<double>, Mhh, 300., 0.)
        CHECK_CACHE(Tensor2<double>, Mlh, 300., 0.)
        CHECK_CACHE(Tensor2<double>, Mh, 300., 0.)
        CHECK_CACHE(double, ac, 300.)
        CHECK_CACHE(double, av, 300.)
        CHECK_CACHE(double, b, 300.)
        CHECK_CACHE(double, d, 300.)
        CHECK_CACHE(double, c11, 300.)
        CHECK_CACHE(double, c12, 300.)
        CHECK_CACHE(double, c44, 300.)
        CHECK_CACHE(double, eps, 300.)
        CHECK_CACHE(double, chi, 300., 0., "G")
        CHECK_CACHE(double, Na)
        CHECK_CACHE(double, Nd)
        CHECK_CACHE(double, Ni, 300.)
        CHECK_CACHE(double, Nf, 300.)
        CHECK_CACHE(double, EactD, 300.)
        CHECK_CACHE(double, EactA, 300.)
        CHECK_CACHE(Tensor2<double>, mob, 300.)
        CHECK_CACHE(Tensor2<double>, cond, 300.)
        CHECK_CACHE(double, A, 300.)
        CHECK_CACHE(double, B, 300.)
        CHECK_CACHE(double, C, 300.)
        CHECK_CACHE(double, D, 300.)
        CHECK_CACHE(Tensor2<double>, thermk, 300., INFINITY)
        CHECK_CACHE(double, dens, 300.)
        CHECK_CACHE(double, cp, 300.)
        CHECK_CACHE(double, nr, py::object(), 300., 0.)
        CHECK_CACHE(double, absp, py::object(), 300.)
        CHECK_CACHE(dcomplex, Nr, py::object(), 300., 0.)
        CHECK_CACHE(Tensor3<dcomplex>, NR, py::object(), 300., 0.)
        CHECK_CACHE(Tensor2<double>, mobe, 300.)
        CHECK_CACHE(Tensor2<double>, mobh, 300.)
        CHECK_CACHE(double, taue, 300.)
        CHECK_CACHE(double, tauh, 300.)
        CHECK_CACHE(double, Ce, 300.)
        CHECK_CACHE(double, Ch, 300.)
        CHECK_CACHE(double, e13, 300.)
        CHECK_CACHE(double, e15, 300.)
        CHECK_CACHE(double, e33, 300.)
        CHECK_CACHE(double, c13, 300.)
        CHECK_CACHE(double, c33, 300.)
        CHECK_CACHE(double, Psp, 300.)
        CHECK_CACHE(double, y1)
        CHECK_CACHE(double, y2)
        CHECK_CACHE(double, y3)
    }
    return ptr;
}

py::object Material_base(const Material* self) {
    const MaterialWithBase* material = dynamic_cast<const MaterialWithBase*>(self);
    if (material) return py::object(material->base);
    else return py::object();
}

/**
 * \return true is the material with given name is in the database
 */
bool MaterialsDB_contains(const MaterialsDB& DB, const std::string& name) {
    for (auto material: DB) if (material->materialName == name) return true;
    return false;
}

MaterialsDB* MaterialsDB_copy(const MaterialsDB& DB) {
    return new MaterialsDB(DB);
}


/**
 * Create material basing on its name and additional parameters
 **/
shared_ptr<Material> MaterialsDB_get(py::tuple args, py::dict kwargs) {

    if (py::len(args) != 2)
        throw ValueError(u8"MaterialsDB.get(self, name, **kwargs) takes exactly two non-keyword arguments");

    const MaterialsDB* DB = py::extract<MaterialsDB*>(args[0]);
    std::string name = py::extract<std::string>(args[1]);

    // Test if we have just a name string
    if (py::len(kwargs) == 0) return DB->get(name);

    return DB->get(kwargs2MaterialComposition(name, kwargs));
}

shared_ptr<Material> MaterialsDB_const(py::tuple args, py::dict kwargs) {
    if (py::len(args) > 2)
        throw ValueError(u8"MaterialsDB.const(self, **kwargs) takes at most two non-keyword arguments");

    shared_ptr<Material> base;
    if (py::len(args) == 2) {
        base = py::extract<shared_ptr<Material>>(args[1]);
    }

    std::map<std::string, double> params;

    py::stl_input_iterator<std::string> begin(kwargs), end, key;
    for (key = begin; key != end; ++key) {
        try {
            params[*key] = py::extract<double>(kwargs[*key]);
        } catch (py::error_already_set) {
            PyErr_Clear();
            throw MaterialParseException("Bad material parameter value '{}={}'", std::string(*key),
                                         py::extract<std::string>(py::str(kwargs[*key]))());
        }

    }

    return plask::make_shared<ConstMaterial>(base, params);
}

// py::dict Material__completeComposition(const Material& self, py::dict src) {
//     py::list keys = src.keys();
//     Material::Composition comp;
//     py::object none;
//     for(int i = 0; i < py::len(keys); ++i) {
//         std::string k = py::extract<std::string>(keys[i]);
//         if (k != "doping") {
//             py::object s = src[keys[i]];
//             comp[py::extract<std::string>(keys[i])] = (s != none) ? py::extract<double>(s): std::numeric_limits<double>::quiet_NaN();
//         }
//     }
//     std::string name = self.name();
//     if (name != "") {
//         std::string basename = splitString2(name, ':').first;
//         std::vector<std::string> objects = Material::parseObjectsNames(basename);
//         for (auto c: comp) {
//             if (std::find(objects.begin(), objects.end(), c.first) == objects.end()) {
//                 throw TypeError(u8"'{}' not allowed in material {}", c.first, name);
//             }
//         }
//     }
//
//     comp = Material::completeComposition(comp);
//
//     py::dict result;
//     for (auto c: comp) result[c.first] = c.second;
//     return result;
// }

std::string Material__str__(const Material& self) {
    return self.str();
}

std::string Material__repr__(const Material& self) {
    return format("plask.material.Material('{0}')", Material__str__(self));
}

py::dict Material_composition(const Material& self) {
    py::dict result;
    for (const auto& item: self.composition()) {
        result[item.first] = item.second;
    }
    return result;
}

py::dict Material__minimal_composition(const Material& self) {
    py::dict result;
    for (const auto& item: Material::minimalComposition(self.composition())) {
        result[item.first] = item.second;
    }
    return result;
}

double Material__getattr__composition(const Material& self, const std::string& attr) {
    auto composition = self.composition();
    auto found = composition.find(attr);
    if (found == composition.end()) throw AttributeError("'Material' object has no attribute '{}'", attr);
    return found->second;
}


namespace detail {
    inline bool getRanges(const MaterialInfo::PropertyInfo&, py::dict&) { return false; }

    template <typename... Args>
    inline bool getRanges(const MaterialInfo::PropertyInfo& info, py::dict& ranges, MaterialInfo::ARGUMENT_NAME arg, Args... args) {
        auto range = info.getArgumentRange(arg);
        if (!isnan(range.first) ||   !isnan(range.second)) {
            ranges[MaterialInfo::ARGUMENT_NAME_STRING[size_t(arg)]] = py::make_tuple(range.first, range.second);
            getRanges(info, ranges, args...);
            return true;
        }
        return getRanges(info, ranges, args...);
    }

    static inline void setDictIfNotEmpty(py::dict& data, const char* key, const std::string& val) {
        if (!val.empty()) data[key] = val;
    }

    template <typename... Args>
    void getPropertyInfo(py::dict& result, const MaterialInfo& minfo, MaterialInfo::PROPERTY_NAME prop, Args... args) {
        if (plask::optional<plask::MaterialInfo::PropertyInfo> info = minfo.getPropertyInfo(prop)) {
            py::dict data;
            setDictIfNotEmpty(data, "source", info->getSource());
            setDictIfNotEmpty(data, "note", info->getNote());
            py::list links;
            for (const auto& link: info->getLinks()) {
                if (link.note.empty())
                    links.append(py::make_tuple(link.className, MaterialInfo::PROPERTY_NAME_STRING[size_t(link.property)]));
                else
                    links.append(py::make_tuple(link.className, MaterialInfo::PROPERTY_NAME_STRING[size_t(link.property)],
                                                link.note));
            }
            if (links) data["seealso"] = links;
            py::dict ranges;
            if (getRanges(*info, ranges, MaterialInfo::doping, args...)) data["ranges"] = ranges;
            result[MaterialInfo::PROPERTY_NAME_STRING[size_t(prop)]] = data;
        }
    }

}

py::dict getMaterialInfoForDB(const MaterialsDB& materialsdb, const std::string& name) {
    plask::optional<MaterialInfo> minfo = materialsdb.info.get(name);
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
    detail::getPropertyInfo(result, *minfo, MaterialInfo::Na);
    detail::getPropertyInfo(result, *minfo, MaterialInfo::Nd);
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
    detail::getPropertyInfo(result, *minfo, MaterialInfo::nr, MaterialInfo::lam, MaterialInfo::T, MaterialInfo::n);
    detail::getPropertyInfo(result, *minfo, MaterialInfo::absp, MaterialInfo::lam, MaterialInfo::T, MaterialInfo::n);
    detail::getPropertyInfo(result, *minfo, MaterialInfo::Nr, MaterialInfo::lam, MaterialInfo::T, MaterialInfo::n);
    detail::getPropertyInfo(result, *minfo, MaterialInfo::NR, MaterialInfo::lam, MaterialInfo::T, MaterialInfo::n);
    detail::getPropertyInfo(result, *minfo, MaterialInfo::mobe, MaterialInfo::T);
    detail::getPropertyInfo(result, *minfo, MaterialInfo::mobh, MaterialInfo::T);
    detail::getPropertyInfo(result, *minfo, MaterialInfo::taue, MaterialInfo::T);
    detail::getPropertyInfo(result, *minfo, MaterialInfo::tauh, MaterialInfo::T);
    detail::getPropertyInfo(result, *minfo, MaterialInfo::Ce, MaterialInfo::T);
    detail::getPropertyInfo(result, *minfo, MaterialInfo::Ch, MaterialInfo::T);
    detail::getPropertyInfo(result, *minfo, MaterialInfo::e13, MaterialInfo::T);
    detail::getPropertyInfo(result, *minfo, MaterialInfo::e15, MaterialInfo::T);
    detail::getPropertyInfo(result, *minfo, MaterialInfo::e33, MaterialInfo::T);
    detail::getPropertyInfo(result, *minfo, MaterialInfo::c13, MaterialInfo::T);
    detail::getPropertyInfo(result, *minfo, MaterialInfo::c33, MaterialInfo::T);
    detail::getPropertyInfo(result, *minfo, MaterialInfo::Psp, MaterialInfo::T);
    detail::getPropertyInfo(result, *minfo, MaterialInfo::y1);
    detail::getPropertyInfo(result, *minfo, MaterialInfo::y2);
    detail::getPropertyInfo(result, *minfo, MaterialInfo::y3);
    return result;
}

py::dict getMaterialInfo(const std::string& name) {
    return getMaterialInfoForDB(MaterialsDB::getDefault(), name);
}

bool MaterialIsAlloy(const std::string& material_name) {
    return MaterialsDB::getDefault().isAlloy(material_name);
}

/**
 * Iterator over material database
 */
struct MaterialsDBIterator {
    MaterialsDB::const_iterator iterator;
    const MaterialsDB::const_iterator end;

    MaterialsDBIterator(const MaterialsDB& self): iterator(self.begin()), end(self.end()) {}

    std::string next() {
        if (iterator == end) {
            PyErr_SetString(PyExc_StopIteration, "");
            py::throw_error_already_set();
        }
        return (*(iterator++))->materialName;
    }

    static MaterialsDBIterator* iter(const MaterialsDB& self) {
        return new MaterialsDBIterator(self);
    }

};

/**
 * Python Context manager for temporaty database
 */
struct TemporaryMaterialDatabase {

    std::unique_ptr<MaterialsDB::TemporaryReplaceDefault> temporary;
    bool copy;

    TemporaryMaterialDatabase(bool copy): copy(copy) {}

    MaterialsDB& enter() {
        temporary.reset(new MaterialsDB::TemporaryReplaceDefault(copy? MaterialsDB::getDefault() : MaterialsDB()));
        return temporary->toRevert;
    }

    void exit(const py::object&, const py::object&, const py::object&) {
        temporary.reset();
    }

};

TemporaryMaterialDatabase* saveMaterialDatabase(bool copy) {
    return new TemporaryMaterialDatabase(copy);
};


void setMaterialDatabase(const MaterialsDB& src) {
    MaterialsDB::getDefault() = src;
}


void register_PythonEvalMaterial_super();


void initMaterials() {

    py::object materials_module { py::handle<>(py::borrowed(PyImport_AddModule("plask._material"))) };
    py::scope().attr("_material") = materials_module;
    py::scope scope = materials_module;

    py::def("getdb", &MaterialsDB::getDefault, "Get default database.", py::return_value_policy<py::reference_existing_object>());

    py::def("setdb", &setMaterialDatabase, py::arg("src"),
            u8"Set new material database.\n\n"
            u8"This function sets a material database to a copy of the provided one.\n"
            u8"It clears the old database, so use it with care. To temporarily switch\n"
            u8"the database, use :func:`~plask.material.savedb`.\n\n"
            u8"Args:\n"
            u8"    src: New material database.\n");

    py::def("savedb", &saveMaterialDatabase, py::arg("copy")=true, py::return_value_policy<py::manage_new_object>(),
            u8"Save existing material database.\n\n"
            u8"This function returns context manager used to save the existing database.\n"
            u8"On entering the context, the old saved database is returned.\n\n"
            u8"Args:\n"
            u8"    copy (bool): if True, the current database is copied to the temporary one.\n\n"
            u8"Example:\n"
            u8"    >>> with plask.material.savedb() as saved:\n"
            u8"    >>>     plask.material.load('some_other_lib')\n");

    py::def("load_library", &MaterialsDB::loadToDefault, py::arg("lib"),
            u8"Load materials from library ``lib`` to default database.\n\n"
            u8"This method can be used to extend the database with custom materials provided\n"
            u8"in a binary library.\n\n"
            u8"Mind that this function will load each library only once (even if\n"
            u8"the database was cleared).\n\n"
            u8"Args:\n"
            u8"    lib (str): Library name to load (without an extension).\n");

    py::def("load_all_libraries", &MaterialsDB::loadAllToDefault, py::arg("dir")=plaskMaterialsPath(),
            u8"Load all materials from specified directory to default database.\n\n"
            u8"This method can be used to extend the database with custom materials provided\n"
            u8"in binary libraries.\n\n"
            u8"Mind that this function will load each library only once (even if\n"
            u8"the database was cleared).\n\n"
            u8"Args:\n"
            u8"    dir (str): Directory name to load materials from.\n");

    py::def("info", &getMaterialInfo, py::arg("name"),
            u8"Get information dictionary on built-in material.\n\n"
            u8"Args:\n"
            u8"    name (str): material name without doping amount and composition.\n"
            u8"                (e.g. 'GaAs:Si', 'AlGaAs').");

    py::def("is_alloy", &MaterialIsAlloy, py::arg("name"),
            u8"Return ``True`` if the specified material is an alloy one.\n\n"
            u8"Args:\n"
            u8"    name (str): material name without doping amount and composition.\n"
            u8"                (e.g. 'GaAs:Si', 'AlGaAs').");

    py::class_<MaterialsDB, boost::noncopyable> materialsDB("MaterialsDB",
        u8"Container of all materials"); materialsDB
        .def("get", py::raw_function(&MaterialsDB_get),
             u8"Get material of given name and doping.\n\n"
             u8":rtype: Material\n" )
        .def("is_alloy", &MaterialsDB::isAlloy, py::arg("name"),
             u8"Return ``True`` if the specified material is an alloy one.\n\n"
             u8"Args:\n"
             u8"    name (str): material name without doping amount and composition.\n"
             u8"                (e.g. 'GaAs:Si', 'AlGaAs').")
        .def("info", &getMaterialInfoForDB, py::arg("name"),
             u8"Get information dictionary on built-in material.\n\n"
             u8"Args:\n"
             u8"    name (str): material name without doping amount and composition.\n"
             u8"                (e.g. 'GaAs:Si', 'AlGaAs').")
        .def("clear", &MaterialsDB::clear, u8"Clear the database.")
        .def("update", &MaterialsDB::update, py::arg("src"),
             u8"Update the database  from a different one.\n\n"
             u8"Args:\n"
             u8"    src: Source database.\n")
        .def("__call__", py::raw_function(&MaterialsDB_get), ":rtype: Material\n")
        .def("__iter__", &MaterialsDBIterator::iter, py::return_value_policy<py::manage_new_object>())
        .def("__contains__", &MaterialsDB_contains)
        .def("__copy__", &MaterialsDB_copy, py::return_value_policy<py::manage_new_object>())
        .def("material_with_params", py::raw_function(&MaterialsDB_const),
             u8"Get material with constant parameters specified as kwargs\n\n"
             u8":rtype: Material\n")
        ;

    {
        py::scope scope(materialsDB);
        py::class_<MaterialsDBIterator, boost::noncopyable>("_Iterator", py::no_init)
            .def("__next__", &MaterialsDBIterator::next)
            .def("__iter__", pass_through)
        ;
    }

    py::class_<TemporaryMaterialDatabase, boost::noncopyable>("_SavedMaterialsContext", py::no_init)
        .def("__enter__", &TemporaryMaterialDatabase::enter, py::return_value_policy<py::reference_existing_object>())
        .def("__exit__", &TemporaryMaterialDatabase::exit)
    ;

    // Common material interface
    py::class_<Material, shared_ptr<Material>, boost::noncopyable> MaterialClass("Material", u8"Base class for all materials.", py::no_init);
    MaterialClass
        .def("__init__", raw_constructor(&PythonMaterial::__init__))

        // .def("complete_composition", &Material__completeComposition, (py::arg("composition")),
        //         u8"Fix incomplete material composition basing on pattern.\n\n"
        //         u8"Args:\n"
        //         u8"    composition (dict): Dictionary with incomplete composition (i.e. the one\n"
        //         u8"                        missing some elements).\n"
        //         u8"Return:\n"
        //         u8"    dict: Dictionary with completed composition.")

        .add_property("name", &Material::name, u8"Material name (without composition and doping amounts).")

        .add_property("dopant", &Material::dopant, u8"Dopant material name (part of name after ':', possibly empty).")

        .add_property("name_without_dopant", &Material::nameWithoutDopant, u8"Material name without dopant (without ':' and part of name after it).")

        .add_property("kind", &Material::kind, u8"Material kind.")

        .add_property("composition", &Material_composition, u8"Material composition.")

        .def("__getattr__", &Material__getattr__composition)

        .add_property("_minimal_composition", &Material__minimal_composition, u8"Material composition.")

        .add_property("doping", &Material::doping, u8"Doping concentration.")

        .add_property("base", &Material_base, u8"Base material.\n\nThis a base material specified for Python and XPL custom materials.")

        .def("__str__", &Material__str__)

        .def("__repr__", &Material__repr__)

        .def("__eq__", (bool(Material::*)(const Material&)const)&Material::operator==)

        .add_property("alloy", &Material::isAlloy)

        .def("lattC", &Material::lattC, (py::arg("T")=300., py::arg("x")="a"),
             u8"Get lattice constant (Å).\n\n"
             u8"Args:\n"
             u8"    T (float): Temperature (K).\n"
             u8"    x (char): lattice parameter (-).\n")

        .def("Eg", &Material::Eg, (py::arg("T")=300., py::arg("e")=0, py::arg("point")="*"),
             u8"Get energy gap Eg (eV).\n\n"
             u8"Args:\n"
             u8"    T (float): Temperature (K).\n"
             u8"    e (float): Lateral strain (-).\n"
             u8"    point (char): Point in the Brillouin zone ('*' means minimum bandgap).\n")

        .def("CB", &Material::CB, (py::arg("T")=300., py::arg("e")=0, py::arg("point")="*"),
             u8"Get conduction band level CB (eV).\n\n"
             u8"Args:\n"
             u8"    T (float): Temperature (K).\n"
             u8"    e (float): Lateral strain (-).\n"
             u8"    point (char): Point in the Brillouin zone ('*' means minimum bandgap).\n")

        .def("VB", &Material::VB, (py::arg("T")=300., py::arg("e")=0, py::arg("point")="*", py::arg("hole")='H'),
             u8"Get valance band level VB (eV).\n\n"
             u8"Args:\n"
             u8"    T (float): Temperature (K).\n"
             u8"    e (float): Lateral strain (-).\n"
             u8"    point (char): Point in the Brillouin zone ('*' means minimum bandgap).\n"
             u8"    hole (char): Hole type ('H' or 'L').\n")

        .def("Dso", &Material::Dso, (py::arg("T")=300., py::arg("e")=0),
             u8"Get split-off energy Dso (eV).\n\n"
             u8"Args:\n"
             u8"    T (float): Temperature (K).\n"
             u8"    e (float): Lateral strain (-).\n")

        .def("Mso", &Material::Mso, (py::arg("T")=300., py::arg("e")=0),
             u8"Get split-off mass Mso (m₀).\n\n"
             u8"Args:\n"
             u8"    T (float): Temperature (K).\n"
             u8"    e (float): Lateral strain (-).\n")

        .def("Me", &Material::Me, (py::arg("T")=300., py::arg("e")=0, py::arg("point")="*"),
             u8"Get electron effective mass Me (m₀).\n\n"
             u8"Args:\n"
             u8"    T (float): Temperature (K).\n"
             u8"    e (float): Lateral strain (-).\n"
             u8"    point (char): Point in the Brillouin zone ('*' means minimum bandgap).\n")

        .def("Mhh", &Material::Mhh, (py::arg("T")=300., py::arg("e")=0),
             u8"Get heavy hole effective mass Mhh (m₀).\n\n"
             u8"Args:\n"
             u8"    T (float): Temperature (K).\n"
             u8"    e (float): Lateral strain (-).\n")

        .def("Mlh", &Material::Mlh, (py::arg("T")=300., py::arg("e")=0),
             u8"Get light hole effective mass Mlh (m₀).\n\n"
             u8"Args:\n"
             u8"    T (float): Temperature (K).\n"
             u8"    e (float): Lateral strain (-).\n")

        .def("Mh", &Material::Mh, (py::arg("T")=300., py::arg("e")=0),
             u8"Get hole effective mass Mh (m₀).\n\n"
             u8"Args:\n"
             u8"    T (float): Temperature (K).\n"
             u8"    e (float): Lateral strain (-).\n")

        .def("ac", &Material::ac, (py::arg("T")=300.),
             u8"Get hydrostatic deformation potential for the conduction band ac (eV).\n\n"
             u8"Args:\n"
             u8"    T (float): Temperature (K).\n")

        .def("av", &Material::av, (py::arg("T")=300.),
             u8"Get hydrostatic deformation potential for the valence band av (eV).\n\n"
             u8"Args:\n"
             u8"    T (float): Temperature (K).\n")

        .def("b", &Material::b, (py::arg("T")=300.),
             u8"Get shear deformation potential b (eV).\n\n"
             u8"Args:\n"
             u8"    T (float): Temperature (K).\n")

        .def("d", &Material::d, (py::arg("T")=300.),
             u8"Get shear deformation potential d (eV).\n\n"
             u8"Args:\n"
             u8"    T (float): Temperature (K).\n")

        .def("c11", &Material::c11, (py::arg("T")=300.),
             u8"Get elastic constant c₁₁ (GPa).\n\n"
             u8"Args:\n"
             u8"    T (float): Temperature (K).\n")

        .def("c12", &Material::c12, (py::arg("T")=300.),
             u8"Get elastic constant c₁₂ (GPa).\n\n"
             u8"Args:\n"
             u8"    T (float): Temperature (K).\n")

        .def("c13", &Material::c13, (py::arg("T")=300.),
             u8"Get elastic constant c₁₃ (GPa).\n\n"
             u8"Args:\n"
             u8"    T (float): Temperature (K).\n")

        .def("c33", &Material::c33, (py::arg("T")=300.),
             u8"Get elastic constant c₃₃ (GPa).\n\n"
             u8"Args:\n"
             u8"    T (float): Temperature (K).\n")

        .def("c44", &Material::c44, (py::arg("T")=300.),
             u8"Get elastic constant c₄₄ (GPa).\n\n"
             u8"Args:\n"
             u8"    T (float): Temperature (K).\n")

        .def("e13", &Material::c44, (py::arg("T")=300.),
             u8"Get piezoelectric constant e₁₃ (C/m²).\n\n"
             u8"Args:\n"
             u8"    T (float): Temperature (K).\n")

        .def("e15", &Material::c44, (py::arg("T")=300.),
             u8"Get piezoelectric constant e₁₅ (C/m²).\n\n"
             u8"Args:\n"
             u8"    T (float): Temperature (K).\n")

        .def("e33", &Material::c44, (py::arg("T")=300.),
             u8"Get piezoelectric constant e₃₃ (C/m²).\n\n"
             u8"Args:\n"
             u8"    T (float): Temperature (K).\n")

        .def("Psp", &Material::Psp, (py::arg("T")=300.),
             u8"Get Spontaneous polarization P (C/m²).\n\n"
             u8"Args:\n"
             u8"    T (float): Temperature (K).\n")

        .def("eps", &Material::eps, (py::arg("T")=300.),
             u8"Get dielectric constant ε (-).\n\n"
             u8"Args:\n"
             u8"    T (float): Temperature (K).\n")

        .def("chi", &Material::chi, (py::arg("T")=300., py::arg("e")=0, py::arg("point")="*"),
             u8"Get electron affinity Chi (eV).\n\n"
             u8"Args:\n"
             u8"    T (float): Temperature (K).\n"
             u8"    e (float): Lateral strain (-).\n"
             u8"    point (char): Point in the Brillouin zone ('*' means minimum bandgap).\n")

        .def("Na", &Material::Na,
                 u8"Get acceptor concentration Na (1/m³).\n\n"
                 u8"Args:-\n")

        .def("Nd", &Material::Nd,
             u8"Get donor concentration Nd (1/m³).\n\n"
             u8"Args:-\n")

        .def("Ni", &Material::Ni, (py::arg("T")=300.),
             u8"Get intrinsic carrier concentration Ni (1/m³).\n\n"
             u8"Args:\n"
             u8"    T (float): Temperature (K).\n")

        .def("Nf", &Material::Nf, (py::arg("T")=300.),
             u8"Get free carrier concentration N (1/m³).\n\n"
             u8"Args:\n"
             u8"    T (float): Temperature (K).\n")

        .def("EactD", &Material::EactD, (py::arg("T")=300.),
             u8"Get donor ionisation energy EactD (eV).\n\n"
             u8"Args:\n"
             u8"    T (float): Temperature (K).\n")

        .def("EactA", &Material::EactA, (py::arg("T")=300.),
             u8"Get acceptor ionisation energy EactA (eV).\n\n"
             u8"Args:\n"
             u8"    T (float): Temperature (K).\n")

        .def("mob", &Material::mob, (py::arg("T")=300.),
             u8"Get majority carriers mobility (cm²/Vs).\n\n"
             u8"Args:\n"
             u8"    T (float): Temperature (K).\n")

        .def("mobe", &Material::mobe, (py::arg("T")=300.),
             u8"Get electron mobility (cm²/Vs).\n\n"
             u8"Args:\n"
             u8"    T (float): Temperature (K).\n")

        .def("mobh", &Material::mobh, (py::arg("T")=300.),
             u8"Get hole mobility (cm²/Vs).\n\n"
             u8"Args:\n"
             u8"    T (float): Temperature (K).\n")

        .def("cond", &Material::cond, (py::arg("T")=300.),
             u8"Get electrical conductivity Sigma (S/m).\n\n"
             u8"Args:\n"
             u8"    T (float): Temperature (K).\n")

        .add_property("condtype", &Material::condtype,
             u8"Electrical conductivity type.")

        .def("A", &Material::A, (py::arg("T")=300.),
             u8"Get monomolecular recombination coefficient A (1/s).\n\n"
             u8"Args:\n"
             u8"    T (float): Temperature (K).\n")

        .def("taue", &Material::taue, (py::arg("T")=300.),
             u8"Get monomolecular electrons lifetime (ns).\n\n"
             u8"Args:\n"
             u8"    T (float): Temperature (K).\n")

        .def("tauh", &Material::tauh, (py::arg("T")=300.),
             u8"Get monomolecular holes lifetime (ns).\n\n"
             u8"Args:\n"
             u8"    T (float): Temperature (K).\n")

        .def("B", &Material::B, (py::arg("T")=300.),
             u8"Get radiative recombination coefficient B (cm³/s).\n\n"
             u8"Args:\n"
             u8"    T (float): Temperature (K).\n")

        .def("C", &Material::C, (py::arg("T")=300.),
             u8"Get Auger recombination coefficient C (cm⁶/s).\n\n"
             u8"Args:\n"
             u8"    T (float): Temperature (K).\n")

        .def("Ce", &Material::Ce, (py::arg("T")=300.),
             u8"Get Auger recombination coefficient C (cm⁶/s) for electrons.\n\n"
             u8"Args:\n"
             u8"    T (float): Temperature (K).\n")

        .def("Ch", &Material::Ch, (py::arg("T")=300.),
             u8"Get Auger recombination coefficient C (cm⁶/s) for holes.\n\n"
             u8"Args:\n"
             u8"    T (float): Temperature (K).\n")

        .def("D", &Material::D, (py::arg("T")=300.),
             u8"Get ambipolar diffusion coefficient D (cm²/s).\n\n"
             u8"Args:\n"
             u8"    T (float): Temperature (K).\n")

        .def("thermk", &Material::thermk, (py::arg("T")=300., py::arg("h")=INFINITY),
             u8"Get thermal conductivity [W/(m K)].\n\n"
             u8"Args:\n"
             u8"    T (float): Temperature (K).\n"
             u8"    h (float): Layer thickness (µm) (-).\n")

        .def("dens", &Material::dens, (py::arg("T")=300.),
             u8"Get density (kg/m³).\n\n"
             u8"Args:\n"
             u8"    T (float): Temperature (K).\n")

        .def("cp", &Material::cp, (py::arg("T")=300.),
             u8"Get specific heat at constant pressure (J/(kg K)).\n\n"
             u8"Args:\n"
             u8"    T (float): Temperature (K).\n")

        .def("nr", &Material::nr, (py::arg("lam"), py::arg("T")=300., py::arg("n")=0.),
             u8"Get refractive index nr (-).\n\n"
             u8"Args:\n"
             u8"    lam (float): Wavelength (nm).\n"
             u8"    T (float): Temperature (K).\n"
             u8"    n (float): Injected carriers concentration (1/cm³).\n")

        .def("absp", &Material::absp, (py::arg("lam"), py::arg("T")=300.),
             u8"Get absorption coefficient alpha (1/cm).\n\n"
             u8"Args:\n"
             u8"    lam (float): Wavelength (nm).\n"
             u8"    T (float): Temperature (K).\n")

        .def("Nr", &Material::Nr, (py::arg("lam"), py::arg("T")=300., py::arg("n")=0.),
             u8"Get complex refractive index Nr (-).\n\n"
             u8"Args:\n"
             u8"    lam (float): Wavelength (nm).\n"
             u8"    T (float): Temperature (K).\n"
             u8"    n (float): Injected carriers concentration (1/cm³).\n")

        .def("NR", &Material::NR, (py::arg("lam"), py::arg("T")=300., py::arg("n")=0.),
             u8"Get complex refractive index tensor Nr (-).\n\n"
             u8"Args:\n"
             u8"    lam (float): Wavelength (nm).\n"
             u8"    T (float): Temperature (K).\n"
             u8"    n (float): Injected carriers concentration (1/cm³).\n\n"
             u8".. warning::\n"
             u8"   This parameter is used only by solvers that can consider refractive index\n"
             u8"   anisotropy properly. It is stronly advised to also define\n"
             u8"   :meth:`~plask.material.Material.Nr`.\n")

        .def("y1", &Material::y1, u8"Get Luttinger parameter γ₁ (-).\n")

        .def("y2", &Material::y2, u8"Get Luttinger parameter γ₂ (-).\n")

        .def("y3", &Material::y3, u8"Get Luttinger parameter γ₃ (-).\n")
    ;

    MaterialFromPythonString();
    register_exception<NoSuchMaterial>(PyExc_ValueError);

    py_enum<Material::Kind>()
        .value("GENERIC", Material::GENERIC)
        .value("EMPTY", Material::EMPTY)
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
            u8"Register new simple material class to the database");

    py::def("_register_material_alloy", &registerAlloyMaterial,
            (py::arg("name"), "material", "base"),
            u8"Register new alloy material class to the database");

    #define REGISTER_CACHED_VALUE_GETTER(param) \
        detail::Cached##param##GetterObj = py::raw_function(&detail::Cached##param##Getter); \
//        detail::Cached##param##GetterObj.attr("__doc__") = MaterialClass.attr(BOOST_PP_STRINGIZE(param)).attr("__doc__");

    REGISTER_CACHED_VALUE_GETTER(lattC);
    REGISTER_CACHED_VALUE_GETTER(Eg);
    REGISTER_CACHED_VALUE_GETTER(CB);
    REGISTER_CACHED_VALUE_GETTER(VB);
    REGISTER_CACHED_VALUE_GETTER(Dso);
    REGISTER_CACHED_VALUE_GETTER(Mso);
    REGISTER_CACHED_VALUE_GETTER(Me);
    REGISTER_CACHED_VALUE_GETTER(Mhh);
    REGISTER_CACHED_VALUE_GETTER(Mlh);
    REGISTER_CACHED_VALUE_GETTER(Mh);
    REGISTER_CACHED_VALUE_GETTER(ac);
    REGISTER_CACHED_VALUE_GETTER(av);
    REGISTER_CACHED_VALUE_GETTER(b);
    REGISTER_CACHED_VALUE_GETTER(d);
    REGISTER_CACHED_VALUE_GETTER(c11);
    REGISTER_CACHED_VALUE_GETTER(c12);
    REGISTER_CACHED_VALUE_GETTER(c44);
    REGISTER_CACHED_VALUE_GETTER(eps);
    REGISTER_CACHED_VALUE_GETTER(chi);
    REGISTER_CACHED_VALUE_GETTER(Na);
    REGISTER_CACHED_VALUE_GETTER(Nd);
    REGISTER_CACHED_VALUE_GETTER(Ni);
    REGISTER_CACHED_VALUE_GETTER(Nf);
    REGISTER_CACHED_VALUE_GETTER(EactD);
    REGISTER_CACHED_VALUE_GETTER(EactA);
    REGISTER_CACHED_VALUE_GETTER(mob);
    REGISTER_CACHED_VALUE_GETTER(cond);
    REGISTER_CACHED_VALUE_GETTER(A);
    REGISTER_CACHED_VALUE_GETTER(B);
    REGISTER_CACHED_VALUE_GETTER(C);
    REGISTER_CACHED_VALUE_GETTER(D);
    REGISTER_CACHED_VALUE_GETTER(thermk);
    REGISTER_CACHED_VALUE_GETTER(dens);
    REGISTER_CACHED_VALUE_GETTER(cp);
    REGISTER_CACHED_VALUE_GETTER(nr);
    REGISTER_CACHED_VALUE_GETTER(absp);
    REGISTER_CACHED_VALUE_GETTER(Nr);
    REGISTER_CACHED_VALUE_GETTER(NR);
    REGISTER_CACHED_VALUE_GETTER(mobe);
    REGISTER_CACHED_VALUE_GETTER(mobh);
    REGISTER_CACHED_VALUE_GETTER(taue);
    REGISTER_CACHED_VALUE_GETTER(tauh);
    REGISTER_CACHED_VALUE_GETTER(Ce);
    REGISTER_CACHED_VALUE_GETTER(Ch);
    REGISTER_CACHED_VALUE_GETTER(e13);
    REGISTER_CACHED_VALUE_GETTER(e15);
    REGISTER_CACHED_VALUE_GETTER(e33);
    REGISTER_CACHED_VALUE_GETTER(c13);
    REGISTER_CACHED_VALUE_GETTER(c33);
    REGISTER_CACHED_VALUE_GETTER(Psp);
    REGISTER_CACHED_VALUE_GETTER(y1);
    REGISTER_CACHED_VALUE_GETTER(y2);
    REGISTER_CACHED_VALUE_GETTER(y3);

    register_PythonEvalMaterial_super();
}

}} // namespace plask::python

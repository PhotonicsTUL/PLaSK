#include "python_globals.h"
#include <boost/python/raw_function.hpp>
#include <boost/python/stl_iterator.hpp>
#include <algorithm>

#include <plask/config.h>
#include <plask/utils/string.h>
#include <plask/utils/xml/reader.h>
#include <plask/exceptions.h>
#include <plask/material/db.h>

#include "../util/raw_constructor.h"

namespace plask { namespace python {

#if PY_VERSION_HEX >= 0x03000000
#       define PLASK_PyCodeObject PyObject
#else
#       define PLASK_PyCodeObject PyCodeObject
#endif

struct EmptyBase : public Material {
    virtual std::string name() const { return ""; }
    virtual Material::Kind kind() const { return Material::NONE; }
};

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
        if (overriden(name)) return py::call_method<R>(self, name, args...);
        return ((*base).*f)(args...);
    }

  public:
    MaterialWrap () : base(new EmptyBase) {}
    MaterialWrap (shared_ptr<Material> base) : base(base) {}

    static shared_ptr<Material> __init__(py::tuple args, py::dict kwargs) {
        if (py::len(args) > 2 || py::len(kwargs) != 0) {
            throw TypeError("wrong number of arguments");
        }
        MaterialWrap* ptr;
        int len = py::len(args);
        if (len == 2) {
            shared_ptr<Material> base = py::extract<shared_ptr<Material>>(args[1]);
            ptr = new MaterialWrap(base);
        } else if (len == 1) {
            ptr = new MaterialWrap();
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

    virtual Material::Kind kind() const { return attr<Material::Kind>("kind"); }
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
    virtual double condT(double T) const { return override<double>("condT", (double (Material::*)(double) const) &Material::condT, T); }
    virtual double condT(double T, double t) const { return override<double>("condT", (double (Material::*)(double, double) const) &Material::condT, T, t); }
    virtual double condT_v(double T, double t) const { return override<double>("condT_v", &Material::condT_v, T, t); }
    virtual double condT_l(double T, double t) const { return override<double>("condT_l", &Material::condT_l, T, t); }
    virtual double dens(double T) const { return override<double>("dens", &Material::dens, T); }
    virtual double specHeat(double T) const { return override<double>("specHeat", &Material::specHeat, T); }
    virtual double nr(double wl, double T) const { return override<double>("nr", &Material::nr, wl, T); }
    virtual double absp(double wl, double T) const { return override<double>("absp", &Material::absp, wl, T); }
    virtual dcomplex Nr(double wl, double T) const {
        if (overriden("Nr")) return py::call_method<dcomplex>(self, "Nr", wl, T);
        else return dcomplex(override<double>("nr", &Material::nr, wl, T),
                             -7.95774715459e-09*override<double>("absp", &Material::absp, wl,T)*wl);
    }

    // End of overriden methods

};


/**
 * Wrapper for Material read from XML of type eval
 * For all virtual functions it calls Python derivatives
 */
class PythonEvalMaterial;

struct PythonEvalMaterialConstructor: public MaterialsDB::MaterialConstructor {

    weak_ptr<PythonEvalMaterialConstructor> self;
    MaterialsDB* db;

    std::string base;
    Material::Kind kind;

    PLASK_PyCodeObject
        *lattC, *Eg, *CBO, *VBO, *Dso, *Mso, *Me, *Me_v, *Me_l, *Mhh, *Mhh_v, *Mhh_l, *Mlh, *Mlh_v, *Mlh_l, *Mh, *Mh_v, *Mh_l,
        *eps, *chi, *Nc, *Ni, *Nf, *EactD, *EactA, *mob, *cond, *cond_v, *cond_l, *res, *res_v, *res_l, *A, *B, *C, *D,
        *condT, *condT_t, *condT_v, *condT_l, *dens, *specHeat, *nr, *absp, *Nr;

    PythonEvalMaterialConstructor(const std::string& name, Material::Kind kind) :
        MaterialsDB::MaterialConstructor(name), base(""), kind(kind),
        lattC(NULL), Eg(NULL), CBO(NULL), VBO(NULL), Dso(NULL), Mso(NULL), Me(NULL), Me_v(NULL), Me_l(NULL),
        Mhh(NULL), Mhh_v(NULL), Mhh_l(NULL), Mlh(NULL), Mlh_v(NULL), Mlh_l(NULL), Mh(NULL), Mh_v(NULL), Mh_l(NULL),
        eps(NULL), chi(NULL), Nc(NULL), Ni(NULL), Nf(NULL), EactD(NULL), EactA(NULL), mob(NULL), cond(NULL),
        cond_v(NULL), cond_l(NULL), res(NULL), res_v(NULL), res_l(NULL), A(NULL), B(NULL), C(NULL), D(NULL),
        condT(NULL), condT_t(NULL), condT_v(NULL), condT_l(NULL), dens(NULL), specHeat(NULL), nr(NULL), absp(NULL),
        Nr(NULL) {}

    PythonEvalMaterialConstructor(const std::string& name, const std::string& base) :
        MaterialsDB::MaterialConstructor(name), base(base), kind(Material::NONE),
        lattC(NULL), Eg(NULL), CBO(NULL), VBO(NULL), Dso(NULL), Mso(NULL), Me(NULL), Me_v(NULL), Me_l(NULL),
        Mhh(NULL), Mhh_v(NULL), Mhh_l(NULL), Mlh(NULL), Mlh_v(NULL), Mlh_l(NULL), Mh(NULL), Mh_v(NULL), Mh_l(NULL),
        eps(NULL), chi(NULL), Nc(NULL), Ni(NULL), Nf(NULL), EactD(NULL), EactA(NULL), mob(NULL), cond(NULL),
        cond_v(NULL), cond_l(NULL), res(NULL), res_v(NULL), res_l(NULL), A(NULL), B(NULL), C(NULL), D(NULL),
        condT(NULL), condT_t(NULL), condT_v(NULL), condT_l(NULL), dens(NULL), specHeat(NULL), nr(NULL), absp(NULL),
        Nr(NULL) {}


    virtual ~PythonEvalMaterialConstructor() {
        Py_XDECREF(lattC); Py_XDECREF(Eg); Py_XDECREF(CBO); Py_XDECREF(VBO); Py_XDECREF(Dso); Py_XDECREF(Mso); Py_XDECREF(Me);
        Py_XDECREF(Me_v); Py_XDECREF(Me_l); Py_XDECREF(Mhh); Py_XDECREF(Mhh_v); Py_XDECREF(Mhh_l); Py_XDECREF(Mlh); Py_XDECREF(Mlh_v);
        Py_XDECREF(Mlh_l); Py_XDECREF(Mh); Py_XDECREF(Mh_v); Py_XDECREF(Mh_l); Py_XDECREF(eps); Py_XDECREF(chi); Py_XDECREF(Nc);
        Py_XDECREF(Ni); Py_XDECREF(Nf); Py_XDECREF(EactD); Py_XDECREF(EactA); Py_XDECREF(mob); Py_XDECREF(cond); Py_XDECREF(cond_v);
        Py_XDECREF(cond_l); Py_XDECREF(res); Py_XDECREF(res_v); Py_XDECREF(res_l); Py_XDECREF(A); Py_XDECREF(B); Py_XDECREF(C);
        Py_XDECREF(D); Py_XDECREF(condT); Py_XDECREF(condT_t); Py_XDECREF(condT_v); Py_XDECREF(condT_l); Py_XDECREF(dens);
        Py_XDECREF(specHeat); Py_XDECREF(nr); Py_XDECREF(absp); Py_XDECREF(Nr);
    }

    inline shared_ptr<Material> operator()(const Material::Composition& composition, Material::DopingAmountType doping_amount_type, double doping_amount) const;
};

class PythonEvalMaterial : public Material
{
    shared_ptr<PythonEvalMaterialConstructor> cls;
    shared_ptr<Material> base;

    py::dict globals;

    template <typename RETURN>
    RETURN call(PLASK_PyCodeObject *fun, const py::dict& locals) const {
        PyObject* result = PyEval_EvalCode(fun, globals.ptr(), locals.ptr());
        if (!result) throw py::error_already_set();
        return py::extract<RETURN>(result);
    }

  public:

    PythonEvalMaterial(const shared_ptr<PythonEvalMaterialConstructor>& constructor, const shared_ptr<Material>& base, const py::dict& params) :
        cls(constructor), base(base) {
        globals = py::dict(py::import("__main__").attr("__dict__")).copy(); // should be numpy instead on __main__?
        globals.update(params);
    }

    // Here there are overridden methods from Material class

    virtual std::string name() const { return cls->materialName; }
    virtual Material::Kind kind() const { return (cls->kind == Material::NONE)? base->kind() : cls->kind; }

#   define PYTHON_EVAL_CALL_1(rtype, fun, arg1) \
        if (cls->fun == NULL) return base->fun(arg1); \
        py::dict locals; locals[BOOST_PP_STRINGIZE(arg1)] = arg1; \
        return call<rtype>(cls->fun, locals);

#   define PYTHON_EVAL_CALL_2(rtype, fun, arg1, arg2) \
        if (cls->fun == NULL) return base->fun(arg1, arg2); \
        py::dict locals; locals[BOOST_PP_STRINGIZE(arg1)] = arg1; locals[BOOST_PP_STRINGIZE(arg2)] = arg2; \
        return call<rtype>(cls->fun, locals);

    virtual double lattC(double T, char x) const { PYTHON_EVAL_CALL_2(double, lattC, T, x) }
    virtual double Eg(double T, const char Point) const { PYTHON_EVAL_CALL_2(double, Eg, T, Point) }
    virtual double CBO(double T, const char Point) const { PYTHON_EVAL_CALL_2(double, CBO, T, Point) }
    virtual double VBO(double T) const { PYTHON_EVAL_CALL_1(double, VBO, T) }
    virtual double Dso(double T) const { PYTHON_EVAL_CALL_1(double, Dso, T) }
    virtual double Mso(double T) const { PYTHON_EVAL_CALL_1(double, Mso, T) }
    virtual double Me(double T, const char Point) const { PYTHON_EVAL_CALL_2(double, Me, T, Point) }
    virtual double Me_v(double T, const char Point) const { PYTHON_EVAL_CALL_2(double, Me_v, T, Point) }
    virtual double Me_l(double T, const char Point) const { PYTHON_EVAL_CALL_2(double, Me_l, T, Point) }
    virtual double Mhh(double T, const char Point) const { PYTHON_EVAL_CALL_2(double, Mhh, T, Point) }
    virtual double Mhh_v(double T, const char Point) const { PYTHON_EVAL_CALL_2(double, Mhh_v, T, Point) }
    virtual double Mhh_l(double T, const char Point) const { PYTHON_EVAL_CALL_2(double, Mhh_l, T, Point) }
    virtual double Mlh(double T, const char Point) const { PYTHON_EVAL_CALL_2(double, Mlh, T, Point) }
    virtual double Mlh_v(double T, const char Point) const { PYTHON_EVAL_CALL_2(double, Mlh_v, T, Point) }
    virtual double Mlh_l(double T, const char Point) const { PYTHON_EVAL_CALL_2(double, Mlh_l, T, Point) }
    virtual double Mh(double T, char EqType) const { PYTHON_EVAL_CALL_2(double, Mh, T, EqType) }
    virtual double Mh_v(double T, const char Point) const { PYTHON_EVAL_CALL_2(double, Mh_v, T, Point) }
    virtual double Mh_l(double T, const char Point) const { PYTHON_EVAL_CALL_2(double, Mh_l, T, Point) }
    virtual double eps(double T) const { PYTHON_EVAL_CALL_1(double, eps, T) }
    virtual double chi(double T, const char Point) const { PYTHON_EVAL_CALL_2(double, chi, T, Point) }
    virtual double Nc(double T, const char Point) const { PYTHON_EVAL_CALL_2(double, Nc, T, Point) }
    virtual double Ni(double T) const { PYTHON_EVAL_CALL_1(double, Ni, T) }
    virtual double Nf(double T) const { PYTHON_EVAL_CALL_1(double, Nf, T) }
    virtual double EactD(double T) const { PYTHON_EVAL_CALL_1(double, EactD, T) }
    virtual double EactA(double T) const { PYTHON_EVAL_CALL_1(double, EactA, T) }
    virtual double mob(double T) const { PYTHON_EVAL_CALL_1(double, mob, T) }
    virtual double cond(double T) const { PYTHON_EVAL_CALL_1(double, cond, T) }
    virtual double cond_v(double T) const { PYTHON_EVAL_CALL_1(double, cond_v, T) }
    virtual double cond_l(double T) const { PYTHON_EVAL_CALL_1(double, cond_l, T) }
    virtual double res(double T) const { PYTHON_EVAL_CALL_1(double, res, T) }
    virtual double res_v(double T) const { PYTHON_EVAL_CALL_1(double, res_v, T) }
    virtual double res_l(double T) const { PYTHON_EVAL_CALL_1(double, res_l, T) }
    virtual double A(double T) const { PYTHON_EVAL_CALL_1(double, A, T) }
    virtual double B(double T) const { PYTHON_EVAL_CALL_1(double, B, T) }
    virtual double C(double T) const { PYTHON_EVAL_CALL_1(double, C, T) }
    virtual double D(double T) const { PYTHON_EVAL_CALL_1(double, D, T) }
    virtual double condT(double T) const { PYTHON_EVAL_CALL_1(double, condT, T) }
    virtual double condT(double T, double t)  const {
        if (cls->condT == NULL) return base->condT(T, t);
        py::dict locals; locals["T"] = T; locals["t"] = t;
        return call<double>(cls->condT_t, locals);
    }
    virtual double condT_v(double T, double t) const { PYTHON_EVAL_CALL_2(double, condT_v, T, t) }
    virtual double condT_l(double T, double t) const { PYTHON_EVAL_CALL_2(double, condT_l, T, t) }
    virtual double dens(double T) const { PYTHON_EVAL_CALL_1(double, dens, T) }
    virtual double specHeat(double T) const { PYTHON_EVAL_CALL_1(double, specHeat, T) }
    virtual double nr(double wl, double T) const { PYTHON_EVAL_CALL_2(double, nr, wl, T) }
    virtual double absp(double wl, double T) const { PYTHON_EVAL_CALL_2(double, absp, wl, T) }
    virtual dcomplex Nr(double wl, double T) const {
        py::dict locals; locals["wl"] = wl; locals["T"] = T;
        py::dict globals = py::dict(py::import("__main__").attr("__dict__"));
        if (cls->Nr != NULL) {
            PyObject* result = PyEval_EvalCode(cls->Nr, globals.ptr(), locals.ptr());
            if (!result) throw py::error_already_set();
            return py::extract<dcomplex>(result);
        }
        else return dcomplex(nr(wl, T), -7.95774715459e-09 * absp(wl, T));
    }

    // End of overridden methods

};

inline shared_ptr<Material> PythonEvalMaterialConstructor::operator()(const Material::Composition& composition, Material::DopingAmountType doping_amount_type, double doping_amount) const {
    shared_ptr<Material> base_obj;
    if (base != "") base_obj = this->db->get(base, doping_amount_type, doping_amount);
    else base_obj = make_shared<EmptyBase>();
    py::dict params;
    if (doping_amount_type == Material::DOPANT_CONCENTRATION) params["dc"] = doping_amount;
    else if (doping_amount_type == Material::CARRIER_CONCENTRATION) params["cc"] = doping_amount;
    return make_shared<PythonEvalMaterial>(self.lock(), base_obj, params);
}

void PythonEvalMaterialLoadFromXML(XMLReader& reader, MaterialsDB& materialsDB) {
    shared_ptr<PythonEvalMaterialConstructor> constructor;
    std::string name = reader.requireAttribute("name");
    auto base = reader.getAttribute("base");
    if (base)
        constructor = make_shared<PythonEvalMaterialConstructor>(name, *base);
    else {
        std::string kindname = reader.requireAttribute("kind");
        Material::Kind kind =  (kindname == "semiconductor" || kindname == "SEMICONDUCTOR")? Material::SEMICONDUCTOR :
                                (kindname == "oxide" || kindname == "OXIDE")? Material::OXIDE :
                                (kindname == "dielectric" || kindname == "DIELECTRIC")? Material::DIELECTRIC :
                                (kindname == "metal" || kindname == "METAL")? Material::METAL :
                                (kindname == "liquid crystal" || kindname == "LIQUID_CRYSTAL" || kindname == "LC")? Material::LIQUID_CRYSTAL :
                                Material::NONE;
        if (kind == Material::NONE) throw XMLBadAttrException(reader, "kind", kindname);
        constructor = make_shared<PythonEvalMaterialConstructor>(name, kind);
    }
    constructor->self = constructor;
    constructor->db = &materialsDB;

    auto trim = [](const char* s) -> const char* { for(; *s != 0 && std::isspace(*s); ++s); return s; };

#   if PY_VERSION_HEX >= 0x03000000

#       define COMPILE_PYTHON_MATERIAL_FUNCTION(func) \
        else if (reader.getNodeName() == BOOST_PP_STRINGIZE(func)) \
            constructor->func = (PLASK_PyCodeObject*)Py_CompileString(trim(reader.requireText().c_str()), BOOST_PP_STRINGIZE(func), Py_eval_input);

#       define COMPILE_PYTHON_MATERIAL_FUNCTION2(name, func) \
        else if (reader.getNodeName() == name) \
            constructor->func = (PLASK_PyCodeObject*)Py_CompileString(trim(reader.requireText().c_str()), BOOST_PP_STRINGIZE(func), Py_eval_input);

#   else
        PyCompilerFlags flags { CO_FUTURE_DIVISION };

#       define COMPILE_PYTHON_MATERIAL_FUNCTION(func) \
        else if (reader.getNodeName() == BOOST_PP_STRINGIZE(func)) \
            constructor->func = (PLASK_PyCodeObject*)Py_CompileStringFlags(trim(reader.requireText().c_str()), BOOST_PP_STRINGIZE(func), Py_eval_input, &flags);

#       define COMPILE_PYTHON_MATERIAL_FUNCTION2(name, func) \
        else if (reader.getNodeName() == name) \
            constructor->func = (PLASK_PyCodeObject*)Py_CompileStringFlags(trim(reader.requireText().c_str()), BOOST_PP_STRINGIZE(func), Py_eval_input, &flags);

#   endif
    while (reader.requireTagOrEnd()) {
        if (false);
        COMPILE_PYTHON_MATERIAL_FUNCTION(lattC)
        COMPILE_PYTHON_MATERIAL_FUNCTION(Eg)
        COMPILE_PYTHON_MATERIAL_FUNCTION(CBO)
        COMPILE_PYTHON_MATERIAL_FUNCTION(VBO)
        COMPILE_PYTHON_MATERIAL_FUNCTION(Dso)
        COMPILE_PYTHON_MATERIAL_FUNCTION(Mso)
        COMPILE_PYTHON_MATERIAL_FUNCTION(Me)
        COMPILE_PYTHON_MATERIAL_FUNCTION(Me_v)
        COMPILE_PYTHON_MATERIAL_FUNCTION(Me_l)
        COMPILE_PYTHON_MATERIAL_FUNCTION(Mhh)
        COMPILE_PYTHON_MATERIAL_FUNCTION(Mhh_v)
        COMPILE_PYTHON_MATERIAL_FUNCTION(Mhh_l)
        COMPILE_PYTHON_MATERIAL_FUNCTION(Mlh)
        COMPILE_PYTHON_MATERIAL_FUNCTION(Mlh_v)
        COMPILE_PYTHON_MATERIAL_FUNCTION(Mlh_l)
        COMPILE_PYTHON_MATERIAL_FUNCTION(Mh)
        COMPILE_PYTHON_MATERIAL_FUNCTION(Mh_v)
        COMPILE_PYTHON_MATERIAL_FUNCTION(Mh_l)
        COMPILE_PYTHON_MATERIAL_FUNCTION(eps)
        COMPILE_PYTHON_MATERIAL_FUNCTION(chi)
        COMPILE_PYTHON_MATERIAL_FUNCTION(Nc)
        COMPILE_PYTHON_MATERIAL_FUNCTION(Ni)
        COMPILE_PYTHON_MATERIAL_FUNCTION(Nf)
        COMPILE_PYTHON_MATERIAL_FUNCTION(EactD)
        COMPILE_PYTHON_MATERIAL_FUNCTION(EactA)
        COMPILE_PYTHON_MATERIAL_FUNCTION(mob)
        COMPILE_PYTHON_MATERIAL_FUNCTION(cond)
        COMPILE_PYTHON_MATERIAL_FUNCTION(cond_v)
        COMPILE_PYTHON_MATERIAL_FUNCTION(cond_l)
        COMPILE_PYTHON_MATERIAL_FUNCTION(res)
        COMPILE_PYTHON_MATERIAL_FUNCTION(res_v)
        COMPILE_PYTHON_MATERIAL_FUNCTION(res_l)
        COMPILE_PYTHON_MATERIAL_FUNCTION(A)
        COMPILE_PYTHON_MATERIAL_FUNCTION(B)
        COMPILE_PYTHON_MATERIAL_FUNCTION(C)
        COMPILE_PYTHON_MATERIAL_FUNCTION(D)
        COMPILE_PYTHON_MATERIAL_FUNCTION(condT)
        COMPILE_PYTHON_MATERIAL_FUNCTION(condT_t)
        COMPILE_PYTHON_MATERIAL_FUNCTION(condT_v)
        COMPILE_PYTHON_MATERIAL_FUNCTION(condT_l)
        COMPILE_PYTHON_MATERIAL_FUNCTION(dens)
        COMPILE_PYTHON_MATERIAL_FUNCTION(specHeat)
        COMPILE_PYTHON_MATERIAL_FUNCTION(nr)
        COMPILE_PYTHON_MATERIAL_FUNCTION(absp)
        COMPILE_PYTHON_MATERIAL_FUNCTION(Nr)
        else throw XMLUnexpectedElementException(reader, "material parameter tag");

        reader.requireTagEnd();
    }

    materialsDB.addSimple(constructor);
}



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
    db.addSimple(new PythonSimpleMaterialConstructor(name, material_class, dopant));
}

/**
 * Function registering custom complex material class to plask
 * \param name name of the material
 * \param material_class Python class object of the custom material
 */
void registerComplexMaterial(const std::string& name, py::object material_class, MaterialsDB& db)
{
    std::string dopant = splitString2(name, ':').second;
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
    std::vector<std::string> elements = Material::parseElementsNames(name);
    py::object none;
    // test if only correct elements are given
    for (int i = 0; i < py::len(keys); ++i) {
        std::string k = py::extract<std::string>(keys[i]);
        if (k != "dp" && k != "dc" && k != "cc" && std::find(elements.begin(), elements.end(), k) == elements.end()) {
            throw TypeError("'%s' not allowed in material %s", k, name);
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

    return DB->get(composition, dopant, doping_type, concentration);
}

/**
 * Converter for Python string to material using default database.
 * Allows to create geometry elements as \c rectange(2,1,"GaAs")
 */
struct Material_from_Python_string {

    Material_from_Python_string() {
        boost::python::converter::registry::push_back(&convertible, &construct, boost::python::type_id<shared_ptr<Material>>());
    }

    // Determine if obj_ptr can be converted into Material
    static void* convertible(PyObject* obj_ptr) {
        if (!PyString_Check(obj_ptr)) return 0;
        return obj_ptr;
    }

    static void construct(PyObject* obj_ptr, boost::python::converter::rvalue_from_python_stage1_data* data) {
        std::string value = PyString_AsString(obj_ptr);

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
        std::vector<std::string> elements = Material::parseElementsNames(basename);
        for (auto c: comp) {
            if (std::find(elements.begin(), elements.end(), c.first) == elements.end()) {
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
        .def("getDefault", &MaterialsDB::getDefault, "Get default database", py::return_value_policy<py::reference_existing_object>())
        .staticmethod("getDefault")
        .def("load", &MaterialsDB::loadToDefault, "Load materials from library lib to default database", py::arg("lib"))
        .staticmethod("load") // TODO make it non-static
        .def("loadAll", &MaterialsDB::loadAllToDefault, "Load all materials from specified directory to default database", py::arg("directory")=plaskMaterialsPath())
        .staticmethod("loadAll") // TODO make it non-static
        .def("get", py::raw_function(&MaterialsDB_get), "Get material of given name and doping")
        .add_property("all", &MaterialsDB_list, "List of all materials in database")
        .def("__iter__", &MaterialsDB_iter)
        .def("__contains__", &MaterialsDB_contains)
    ;

    // Common material interface
    py::class_<Material, shared_ptr<Material>, boost::noncopyable> MaterialClass("Material", "Base class for all materials.", py::no_init);
    MaterialClass
        .def("__init__", raw_constructor(&MaterialWrap::__init__))
        .def("_completeComposition", &Material__completeComposition, (py::arg("composition"), py::arg("name")=""),
             "Fix incomplete material composition basing on pattern")
        .staticmethod("_completeComposition")
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
        .def("condT", (double (Material::*)(double) const)&Material::condT, (py::arg("T")=300.), "Get thermal conductivity [W/(m*K)]")
        .def("condT", (double (Material::*)(double, double) const)&Material::condT, (py::arg("T")=300., py::arg("thickness")), "Get thermal conductivity [W/(m*K)]")
        .def("condT_v", &Material::condT_v, (py::arg("T")=300., py::arg("thickness")), "Get thermal conductivity in vertical direction k [W/(m*K)]")
        .def("condT_l", &Material::condT_l, (py::arg("T")=300., py::arg("thickness")), "Get thermal conductivity in lateral direction k [W/(m*K)]")
        .def("dens", &Material::dens, (py::arg("T")=300.), "Get density [kg/m**3]")
        .def("specHeat", &Material::specHeat, (py::arg("T")=300.), "Get specific heat at constant pressure [J/(kg*K)]")
        .def("nr", &Material::nr, (py::arg("wl"), py::arg("T")=300.), "Get refractive index nR")
        .def("absp", &Material::absp, (py::arg("wl"), py::arg("T")=300.), "Get absorption coefficient alpha")
        .def("Nr", &Material::Nr, (py::arg("wl"), py::arg("T")=300.), "Get refractive index nR")

    ;

    Material_from_Python_string();
    register_exception<NoSuchMaterial>(PyExc_ValueError);
    register_exception<MaterialMethodNotApplicable>(PyExc_TypeError);

    py::enum_<Material::Kind> MaterialKind("kind"); MaterialKind
        .value("NONE", Material::NONE)
        .value("SEMICONDUCTOR", Material::SEMICONDUCTOR)
        .value("OXIDE", Material::OXIDE)
        .value("DIELECTRIC", Material::DIELECTRIC)
        .value("METAL", Material::METAL)
        .value("LIQUID_CRYSTAL", Material::LIQUID_CRYSTAL)
        .value("MIXED", Material::MIXED)
    ;

    py::def("_register_material_simple", &registerSimpleMaterial, (py::arg("name"), py::arg("material"), py::arg("database")=MaterialsDB::getDefault()),
            "Register new simple material class to the database");

    py::def("_register_material_complex", &registerComplexMaterial, (py::arg("name"), py::arg("material"), py::arg("database")=MaterialsDB::getDefault()),
            "Register new complex material class to the database");
}

}} // namespace plask::python

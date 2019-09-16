#include "python_globals.h"

#include <plask/utils/string.h>
#include <plask/utils/xml/reader.h>
#include <plask/material/db.h>

#include "python_material.h"
#include "python_manager.h"
#include "python_ptr.h"

namespace plask { namespace python {

/**
 * Wrapper for Material read from XML of type eval
 * For all virtual functions it calls Python derivatives
 */
class PythonEvalMaterial;

extern PLASK_PYTHON_API py::dict* xml_globals;

struct PythonEvalMaterialConstructor: public MaterialsDB::MaterialConstructor {

    MaterialsDB::ProxyMaterialConstructor base;

    weak_ptr<PythonEvalMaterialConstructor> self;

    MaterialCache cache;

    Material::Kind kind;
    Material::ConductivityType condtype;

    bool alloy;

    PyHandle<PyCodeObject>
        lattC, Eg, CB, VB, Dso, Mso, Me, Mhh, Mlh, Mh, ac, av, b, d, c11, c12, c44, eps, chi,
        Na, Nd, Ni, Nf, EactD, EactA, mob, cond, A, B, C, D,
        thermk, dens, cp, nr, absp, Nr, NR,
        mobe, mobh, taue, tauh, Ce, Ch, e13, e15, e33, c13, c33, Psp,
        y1, y2, y3;

    PythonEvalMaterialConstructor(const std::string& name, const std::string& base, bool alloy) :
        MaterialsDB::MaterialConstructor(name),
        base(base),
        kind(Material::GENERIC), condtype(Material::CONDUCTIVITY_UNDETERMINED),
        alloy(alloy)
    {}

    inline shared_ptr<Material> operator()(const Material::Composition& composition, double doping) const override;

    bool isAlloy() const override { return alloy; }
};

class PythonEvalMaterial: public MaterialWithBase
{
    shared_ptr<PythonEvalMaterialConstructor> cls;

    py::object self;

    Parameters params;

    friend struct PythonEvalMaterialConstructor;

    static inline PyObject* py_eval(PyCodeObject *fun, const py::dict& locals) {
        return
#if PY_VERSION_HEX >= 0x03000000
            PyEval_EvalCode((PyObject*)fun, xml_globals->ptr(), locals.ptr());
#else
            PyEval_EvalCode(fun, xml_globals->ptr(), locals.ptr());
#endif
    }

    template <typename RETURN>
    inline RETURN call(PyCodeObject *fun, const py::dict& locals, const char* funname) const {
        try {
            return py::extract<RETURN>(py::handle<>(py_eval(fun, locals)).get());
        } catch (py::error_already_set&) {
            PyObject *ptype, *pvalue, *ptraceback;
            PyErr_Fetch(&ptype, &pvalue, &ptraceback);
            Py_XDECREF(ptraceback);
            std::string type, value;
            if (ptype) { type = py::extract<std::string>(py::object(py::handle<>(ptype)).attr("__name__")); type = ": " + type; }
            if (pvalue) { value = py::extract<std::string>(py::str(py::handle<>(pvalue))); value = ": " + value; }
            throw ValueError("Error in custom material function <{1}> of '{0}'{2}{3}", this->name(), funname, type, value);
        }
    }

  public:

    PythonEvalMaterial(const shared_ptr<PythonEvalMaterialConstructor>& constructor, const shared_ptr<Material>& base) :
        MaterialWithBase(base), cls(constructor)
    {
        // This is an ugly hack using aliasing shared_ptr constructor. However, this is the only way to make a Python
        // object out of an existing normal pointer. Luckily there is very little chance anyone will ever store this,
        // as this object is accessible only to expression specified in <materials> section. The downside is that
        // if anyone uses some trick to store it and `this` gets deleted, there is no way to prevent the crash...
        self = py::object(shared_ptr<Material>(shared_ptr<Material>(), this));
    }

    // Here there are overridden methods from Material class

    OmpLockGuard<OmpNestLock> lock() const override {
        return OmpLockGuard<OmpNestLock>(python_omp_lock);
    }

    bool isEqual(const Material& other) const override {
        auto theother = static_cast<const PythonEvalMaterial&>(other);
        OmpLockGuard<OmpNestLock> lock(python_omp_lock);
        return cls == theother.cls &&
               bool(base) == bool(theother.base) && (!base || base->str() == theother.base->str()) &&
               self.attr("__dict__") == theother.self.attr("__dict__");
    }

    std::string name() const override { return cls->materialName; }
    Material::Kind kind() const override { return (cls->kind == Material::EMPTY)? base->kind() : cls->kind; }
    Material::ConductivityType condtype() const override { return (cls->condtype == Material::CONDUCTIVITY_UNDETERMINED)? base->condtype() : cls->condtype; }

    std::string str() const override {
        return params.str();
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


#   define PYTHON_EVAL_CALL_0(rtype, fun) \
        if (cls->cache.fun) return *cls->cache.fun;\
        if (cls->fun == NULL) return base->fun(); \
        OmpLockGuard<OmpNestLock> lock(python_omp_lock); \
        py::dict locals; locals["self"] = self; \
        return call<rtype>(cls->fun, locals, BOOST_PP_STRINGIZE(fun));

#   define PYTHON_EVAL_CALL_1(rtype, fun, arg1) \
        if (cls->cache.fun) return *cls->cache.fun;\
        if (cls->fun == NULL) return base->fun(arg1); \
        OmpLockGuard<OmpNestLock> lock(python_omp_lock); \
        py::dict locals; locals["self"] = self; locals[BOOST_PP_STRINGIZE(arg1)] = arg1; \
        return call<rtype>(cls->fun, locals, BOOST_PP_STRINGIZE(fun));

#   define PYTHON_EVAL_CALL_2(rtype, fun, arg1, arg2) \
        if (cls->cache.fun) return *cls->cache.fun;\
        if (cls->fun == NULL) return base->fun(arg1, arg2); \
        OmpLockGuard<OmpNestLock> lock(python_omp_lock); \
        py::dict locals; locals["self"] = self; locals[BOOST_PP_STRINGIZE(arg1)] = arg1; locals[BOOST_PP_STRINGIZE(arg2)] = arg2; \
        return call<rtype>(cls->fun, locals, BOOST_PP_STRINGIZE(fun));

#   define PYTHON_EVAL_CALL_3(rtype, fun, arg1, arg2, arg3) \
        if (cls->cache.fun) return *cls->cache.fun; \
        if (cls->fun == NULL) return base->fun(arg1, arg2, arg3); \
        OmpLockGuard<OmpNestLock> lock(python_omp_lock); \
        py::dict locals; locals["self"] = self; locals[BOOST_PP_STRINGIZE(arg1)] = arg1; locals[BOOST_PP_STRINGIZE(arg2)] = arg2; \
        locals[BOOST_PP_STRINGIZE(arg3)] = arg3; \
        return call<rtype>(cls->fun, locals, BOOST_PP_STRINGIZE(fun));

#   define PYTHON_EVAL_CALL_4(rtype, fun, arg1, arg2, arg3, arg4) \
        if (cls->cache.fun) return *cls->cache.fun;\
        if (cls->fun == NULL) return base->fun(arg1, arg2, arg3, arg4); \
        OmpLockGuard<OmpNestLock> lock(python_omp_lock); \
        py::dict locals; locals["self"] = self; locals[BOOST_PP_STRINGIZE(arg1)] = arg1; locals[BOOST_PP_STRINGIZE(arg2)] = arg2; \
        locals[BOOST_PP_STRINGIZE(arg3)] = arg3; locals[BOOST_PP_STRINGIZE(arg4)] = arg4; \
        return call<rtype>(cls->fun, locals, BOOST_PP_STRINGIZE(fun));

    double lattC(double T, char x) const override { PYTHON_EVAL_CALL_2(double, lattC, T, x) }
    double Eg(double T, double e, char point) const override { PYTHON_EVAL_CALL_3(double, Eg, T, e, point) }
    double CB(double T, double e, char point) const override {
        if (cls->cache.CB) return *cls->cache.CB;
        if (cls->CB != NULL) {
            OmpLockGuard<OmpNestLock> lock(python_omp_lock);
            py::dict locals; locals["self"] = self; locals["T"] = T; locals["e"] = e; locals["point"] = point;
            return py::extract<double>(py::handle<>(py_eval(cls->CB, locals)).get());
        }
        if (cls->VB != NULL || cls->Eg != NULL || cls->cache.VB || cls->cache.Eg)
            return VB(T, e, point, 'H') + Eg(T, e, point);
        return base->CB(T, e, point);
    }
    double VB(double T, double e, char point, char hole) const override { PYTHON_EVAL_CALL_4(double, VB, T, e, point, hole) }
    double Dso(double T, double e) const override { PYTHON_EVAL_CALL_2(double, Dso, T, e) }
    double Mso(double T, double e) const override { PYTHON_EVAL_CALL_2(double, Mso, T, e) }
    Tensor2<double> Me(double T, double e, char point) const override { PYTHON_EVAL_CALL_3(Tensor2<double>, Me, T, e, point) }
    Tensor2<double> Mhh(double T, double e) const override { PYTHON_EVAL_CALL_2(Tensor2<double>, Mhh, T, e) }
    Tensor2<double> Mlh(double T, double e) const override { PYTHON_EVAL_CALL_2(Tensor2<double>, Mlh, T, e) }
    Tensor2<double> Mh(double T, double e) const override { PYTHON_EVAL_CALL_2(Tensor2<double>, Mh, T, e) }
    double ac(double T) const override { PYTHON_EVAL_CALL_1(double, ac, T) }
    double av(double T) const override { PYTHON_EVAL_CALL_1(double, av, T) }
    double b(double T) const override { PYTHON_EVAL_CALL_1(double, b, T) }
    double d(double T) const override { PYTHON_EVAL_CALL_1(double, d, T) }
    double c11(double T) const override { PYTHON_EVAL_CALL_1(double, c11, T) }
    double c12(double T) const override { PYTHON_EVAL_CALL_1(double, c12, T) }
    double c44(double T) const override { PYTHON_EVAL_CALL_1(double, c44, T) }
    double eps(double T) const override { PYTHON_EVAL_CALL_1(double, eps, T) }
    double chi(double T, double e, char point) const override { PYTHON_EVAL_CALL_3(double, chi, T, e, point) }
    double Na() const override { PYTHON_EVAL_CALL_0(double, Na) }
    double Nd() const override { PYTHON_EVAL_CALL_0(double, Nd) }
    double Ni(double T) const override { PYTHON_EVAL_CALL_1(double, Ni, T) }
    double Nf(double T) const override { PYTHON_EVAL_CALL_1(double, Nf, T) }
    double EactD(double T) const override { PYTHON_EVAL_CALL_1(double, EactD, T) }
    double EactA(double T) const override { PYTHON_EVAL_CALL_1(double, EactA, T) }
    Tensor2<double> mob(double T) const override { PYTHON_EVAL_CALL_1(Tensor2<double>, mob, T) }
    Tensor2<double> cond(double T) const override { PYTHON_EVAL_CALL_1(Tensor2<double>, cond, T) }
    double A(double T) const override { PYTHON_EVAL_CALL_1(double, A, T) }
    double B(double T) const override { PYTHON_EVAL_CALL_1(double, B, T) }
    double C(double T) const override { PYTHON_EVAL_CALL_1(double, C, T) }
    double D(double T) const override {
        if (cls->cache.D) { return *cls->cache.D; }
        if (cls->D != NULL) {
            OmpLockGuard<OmpNestLock> lock(python_omp_lock);
            py::dict locals; locals["self"] = self; locals["T"] = T;
            return call<double>(cls->D, locals, "D");
        }
        // D = µ kB T / e
        if (cls->cache.mob || cls->mob != NULL) {
            return mob(T).c00 * T * 8.6173423e-5;
        }
        return base->D(T);
    }
    Tensor2<double> thermk(double T, double h) const override { PYTHON_EVAL_CALL_2(Tensor2<double>, thermk, T, h) }
    double dens(double T) const override { PYTHON_EVAL_CALL_1(double, dens, T) }
    double cp(double T) const override { PYTHON_EVAL_CALL_1(double, cp, T) }
    double nr(double lam, double T, double n = .0) const override {
        if (cls->cache.nr) return *cls->cache.nr;
        if (cls->nr == NULL) return base->nr(lam, T, n);
        OmpLockGuard<OmpNestLock> lock(python_omp_lock);
        py::dict locals; locals["self"] = self; locals["lam"] = locals["wl"] = lam; locals["T"] = T; locals["n"] = n;
        return call<double>(cls->nr, locals, "nr");
    }
    double absp(double lam, double T) const override {
        if (cls->cache.absp) return *cls->cache.absp;
        if (cls->absp == NULL) return base->absp(lam, T);
        OmpLockGuard<OmpNestLock> lock(python_omp_lock);
        py::dict locals; locals["self"] = self; locals["lam"] = locals["wl"] = lam; locals["T"] = T;
        return call<double>(cls->absp, locals, "absp");
    }
    dcomplex Nr(double lam, double T, double n = .0) const override {
        if (cls->cache.Nr) return *cls->cache.Nr;
        if (cls->Nr != NULL) {
            OmpLockGuard<OmpNestLock> lock(python_omp_lock);
            py::dict locals; locals["self"] = self; locals["lam"] = locals["wl"] = lam; locals["T"] = T; locals["n"] = n;
            return call<dcomplex>(cls->Nr, locals, "Nr");
        }
        if (cls->nr != NULL || cls->absp != NULL || cls->cache.nr || cls->cache.absp)
            return dcomplex(nr(lam, T, n), -7.95774715459e-09 * absp(lam, T)*lam);
        return base->Nr(lam, T, n);
    }
    Tensor3<dcomplex> NR(double lam, double T, double n = .0) const override {
        if (cls->cache.NR) return *cls->cache.NR;
        if (cls->NR != NULL) {
            OmpLockGuard<OmpNestLock> lock(python_omp_lock);
            py::dict locals; locals["self"] = self; locals["lam"] = locals["wl"] = lam; locals["T"] = T; locals["n"] = n;
            return call<Tensor3<dcomplex>>(cls->NR, locals, "NR");
        }
        if (cls->cache.Nr) {
            dcomplex nc = *cls->cache.Nr;
            return Tensor3<dcomplex>(nc, nc, nc, 0.);
        }
        if (cls->Nr != NULL) {
            OmpLockGuard<OmpNestLock> lock(python_omp_lock);
            py::dict locals; locals["self"] = self; locals["lam"] = locals["wl"] = lam; locals["T"] = T; locals["n"] = n;
            dcomplex nc = call<dcomplex>(cls->Nr, locals, "Nr");
            return Tensor3<dcomplex>(nc, nc, nc, 0.);
        }
        if (cls->nr != NULL || cls->absp != NULL || cls->cache.nr || cls->cache.absp) {
            OmpLockGuard<OmpNestLock> lock(python_omp_lock);
            dcomplex nc(nr(lam, T, n), -7.95774715459e-09 * absp(lam, T)*lam);
            return Tensor3<dcomplex>(nc, nc, nc, 0.);
        }
        return base->NR(lam, T, n);
    }

    Tensor2<double> mobe(double T) const override { PYTHON_EVAL_CALL_1(Tensor2<double>, mobe, T) }
    Tensor2<double> mobh(double T) const override { PYTHON_EVAL_CALL_1(Tensor2<double>, mobh, T) }
    double taue(double T) const override { PYTHON_EVAL_CALL_1(double, taue, T) }
    double tauh(double T) const override { PYTHON_EVAL_CALL_1(double, tauh, T) }
    double Ce(double T) const override { PYTHON_EVAL_CALL_1(double, Ce, T) }
    double Ch(double T) const override { PYTHON_EVAL_CALL_1(double, Ch, T) }
    double e13(double T) const override { PYTHON_EVAL_CALL_1(double, e13, T) }
    double e15(double T) const override { PYTHON_EVAL_CALL_1(double, e15, T) }
    double e33(double T) const override { PYTHON_EVAL_CALL_1(double, e33, T) }
    double c13(double T) const override { PYTHON_EVAL_CALL_1(double, c13, T) }
    double c33(double T) const override { PYTHON_EVAL_CALL_1(double, c33, T) }
    double Psp(double T) const override { PYTHON_EVAL_CALL_1(double, Psp, T) }

    double y1() const override { PYTHON_EVAL_CALL_0(double, y1) }
    double y2() const override { PYTHON_EVAL_CALL_0(double, y2) }
    double y3() const override { PYTHON_EVAL_CALL_0(double, y3) }

    // End of overridden methods
};

inline shared_ptr<Material> PythonEvalMaterialConstructor::operator()(const Material::Composition& composition, double doping) const {
    OmpLockGuard<OmpNestLock> lock(python_omp_lock);
    auto material = plask::make_shared<PythonEvalMaterial>(self.lock(), base(composition, doping));
    material->params = Material::Parameters(materialName, true);
    if (alloy) material->params.composition = Material::completeComposition(composition);
    material->params.doping = doping;
    return material;
}

void PythonManager::loadMaterial(XMLReader& reader) {
    try {
        std::string material_name = reader.requireAttribute("name");
        std::string base_name = reader.requireAttribute("base");
        bool alloy = reader.getAttribute<bool>("alloy", false);

        shared_ptr<PythonEvalMaterialConstructor> constructor = plask::make_shared<PythonEvalMaterialConstructor>(material_name, base_name, alloy);
        constructor->self = constructor;

        auto trim = [](const char* s) -> const char* {
            while (std::isspace(*s)) ++s;
            return s;
        };

    #   if PY_VERSION_HEX >= 0x03000000
    #       define COMPILE_PYTHON_MATERIAL_FUNCTION2(funcname, func) \
            else if (reader.getNodeName() == funcname) { \
                constructor->func = (PyCodeObject*)Py_CompileString(trim(reader.requireTextInCurrentTag().c_str()), funcname, Py_eval_input); \
                if (constructor->func == nullptr) \
                    throw XMLException(format("XML line {0} in <" funcname ">", reader.getLineNr()), "Material parameter syntax error"); \
                try { \
                    py::dict locals; \
                    constructor->cache.func.reset( \
                        py::extract<typename std::remove_reference<decltype(*constructor->cache.func)>::type>( \
                            py::handle<>(PyEval_EvalCode(constructor->func.ptr_cast<PyObject>(), xml_globals->ptr(), locals.ptr())).get() \
                        ) \
                    ); \
                    writelog(LOG_DEBUG, "Cached parameter '" funcname "' in material '{0}'", material_name); \
                } catch (py::error_already_set&) { \
                    PyErr_Clear(); \
                } \
            }
    #   else
            PyCompilerFlags flags { CO_FUTURE_DIVISION };
    #       define COMPILE_PYTHON_MATERIAL_FUNCTION2(funcname, func) \
            else if (reader.getNodeName() == funcname) { \
                constructor->func = (PyCodeObject*)Py_CompileStringFlags(trim(reader.requireTextInCurrentTag().c_str()), funcname, Py_eval_input, &flags); \
                if (constructor->func == nullptr) \
                    throw XMLException(format("XML line {0} in <" funcname ">", reader.getLineNr()), "Material parameter syntax error"); \
                try { \
                    py::dict locals; \
                    constructor->cache.func.reset( \
                        py::extract<typename std::remove_reference<decltype(*constructor->cache.func)>::type>( \
                            py::handle<>(PyEval_EvalCode(constructor->func, xml_globals->ptr(), locals.ptr())).get() \
                        ) \
                    ); \
                    writelog(LOG_DEBUG, "Cached parameter '" funcname "' in material '{0}'", material_name); \
                } catch (py::error_already_set) { \
                    PyErr_Clear(); \
                } \
            }
    #   endif

    #   define COMPILE_PYTHON_MATERIAL_FUNCTION(func) COMPILE_PYTHON_MATERIAL_FUNCTION2(BOOST_PP_STRINGIZE(func), func)

        while (reader.requireTagOrEnd()) {
            if (reader.getNodeName() == "condtype") {
                auto condname = reader.requireTextInCurrentTag();
                Material::ConductivityType condtype = (condname == "n" || condname == "N")? Material::CONDUCTIVITY_N :
                                (condname == "i" || condname == "I")? Material::CONDUCTIVITY_I :
                                (condname == "p" || condname == "P")? Material::CONDUCTIVITY_P :
                                (condname == "other" || condname == "OTHER")? Material::CONDUCTIVITY_OTHER :
                                 Material::CONDUCTIVITY_UNDETERMINED;
                if (condtype == Material::CONDUCTIVITY_UNDETERMINED)
                    throw XMLException(format("XML line {0} in <{1}>", reader.getLineNr(), "condtype"), "Material parameter syntax error, condtype must be given as one of: n, i, p, other (or: N, I, P, OTHER)");
                constructor->condtype = condtype;
            } //else if
            COMPILE_PYTHON_MATERIAL_FUNCTION(lattC)
            COMPILE_PYTHON_MATERIAL_FUNCTION(Eg)
            COMPILE_PYTHON_MATERIAL_FUNCTION(CB)
            COMPILE_PYTHON_MATERIAL_FUNCTION(VB)
            COMPILE_PYTHON_MATERIAL_FUNCTION(Dso)
            COMPILE_PYTHON_MATERIAL_FUNCTION(Mso)
            COMPILE_PYTHON_MATERIAL_FUNCTION(Me)
            COMPILE_PYTHON_MATERIAL_FUNCTION(Mhh)
            COMPILE_PYTHON_MATERIAL_FUNCTION(Mlh)
            COMPILE_PYTHON_MATERIAL_FUNCTION(Mh)
            COMPILE_PYTHON_MATERIAL_FUNCTION(ac)
            COMPILE_PYTHON_MATERIAL_FUNCTION(av)
            COMPILE_PYTHON_MATERIAL_FUNCTION(b)
            COMPILE_PYTHON_MATERIAL_FUNCTION(d)
            COMPILE_PYTHON_MATERIAL_FUNCTION(c11)
            COMPILE_PYTHON_MATERIAL_FUNCTION(c12)
            COMPILE_PYTHON_MATERIAL_FUNCTION(c44)
            COMPILE_PYTHON_MATERIAL_FUNCTION(eps)
            COMPILE_PYTHON_MATERIAL_FUNCTION(chi)
            COMPILE_PYTHON_MATERIAL_FUNCTION(Na)
            COMPILE_PYTHON_MATERIAL_FUNCTION(Nd)
            COMPILE_PYTHON_MATERIAL_FUNCTION(Ni)
            COMPILE_PYTHON_MATERIAL_FUNCTION(Nf)
            COMPILE_PYTHON_MATERIAL_FUNCTION(EactD)
            COMPILE_PYTHON_MATERIAL_FUNCTION(EactA)
            COMPILE_PYTHON_MATERIAL_FUNCTION(mob)
            COMPILE_PYTHON_MATERIAL_FUNCTION(cond)
            COMPILE_PYTHON_MATERIAL_FUNCTION(A)
            COMPILE_PYTHON_MATERIAL_FUNCTION(B)
            COMPILE_PYTHON_MATERIAL_FUNCTION(C)
            COMPILE_PYTHON_MATERIAL_FUNCTION(D)
            COMPILE_PYTHON_MATERIAL_FUNCTION(thermk)
            COMPILE_PYTHON_MATERIAL_FUNCTION(dens)
            COMPILE_PYTHON_MATERIAL_FUNCTION(cp)
            COMPILE_PYTHON_MATERIAL_FUNCTION(nr)
            COMPILE_PYTHON_MATERIAL_FUNCTION(absp)
            COMPILE_PYTHON_MATERIAL_FUNCTION(Nr)
            COMPILE_PYTHON_MATERIAL_FUNCTION(NR)

            COMPILE_PYTHON_MATERIAL_FUNCTION(mobe)
            COMPILE_PYTHON_MATERIAL_FUNCTION(mobh)
            COMPILE_PYTHON_MATERIAL_FUNCTION(taue)
            COMPILE_PYTHON_MATERIAL_FUNCTION(tauh)
            COMPILE_PYTHON_MATERIAL_FUNCTION(Ce)
            COMPILE_PYTHON_MATERIAL_FUNCTION(Ch)
            COMPILE_PYTHON_MATERIAL_FUNCTION(e13)
            COMPILE_PYTHON_MATERIAL_FUNCTION(e15)
            COMPILE_PYTHON_MATERIAL_FUNCTION(e33)
            COMPILE_PYTHON_MATERIAL_FUNCTION(c13)
            COMPILE_PYTHON_MATERIAL_FUNCTION(c33)
            COMPILE_PYTHON_MATERIAL_FUNCTION(Psp)

            COMPILE_PYTHON_MATERIAL_FUNCTION(y1)
            COMPILE_PYTHON_MATERIAL_FUNCTION(y2)
            COMPILE_PYTHON_MATERIAL_FUNCTION(y3)

            else throw XMLUnexpectedElementException(reader, "material parameter tag");
        }

        if (alloy)
            MaterialsDB::getDefault().addAlloy(constructor);
        else
            MaterialsDB::getDefault().addSimple(constructor);
    } catch (py::error_already_set&) {
        if (draft) PyErr_Clear();
        else throw;
    } catch (...) {
        if (!draft) throw;
    }
}

}} // namespace plask::python

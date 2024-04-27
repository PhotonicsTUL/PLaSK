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

#include "plask/material/db.hpp"
#include "plask/utils/string.hpp"
#include "plask/utils/xml/reader.hpp"

#include "python_manager.hpp"
#include "python_ptr.hpp"

namespace plask { namespace python {

/**
 * Wrapper for Material read from XML of type eval
 * For all virtual functions it calls Python derivatives
 */
class PythonEvalMaterial;

struct PythonEvalMaterialConstructor : public MaterialsDB::MaterialConstructor {
    MaterialsDB::ProxyMaterialConstructor base;

    weak_ptr<PythonEvalMaterialConstructor> self;

    MaterialCache cache;

    Material::Kind kind;
    Material::ConductivityType condtype;

    bool alloy;

    py::object globals;

    PyHandle<> lattC, Eg, CB, VB, Dso, Mso, Me, Mhh, Mlh, Mh, ac, av, b, d, c11, c12, c44, eps, chi, Na, Nd, Ni, Nf, EactD, EactA,
        mob, cond, A, B, C, D, thermk, dens, cp, nr, absp, Nr, Eps, mobe, mobh, taue, tauh, Ce, Ch, e13, e15, e33, c13, c33, Psp, y1,
        y2, y3;

    template <typename BaseT>
    PythonEvalMaterialConstructor(const std::string& name, const BaseT& base, bool alloy, const py::object& globals)
        : MaterialsDB::MaterialConstructor(name),
          base(base),
          kind(Material::GENERIC),
          condtype(Material::CONDUCTIVITY_UNDETERMINED),
          alloy(alloy),
          globals(globals) {}

    inline shared_ptr<Material> operator()(const Material::Composition& composition, double doping) const override;

    bool isAlloy() const override { return alloy; }
};

struct MaterialSuper {
    const shared_ptr<Material>& base;
    MaterialSuper(const shared_ptr<Material>& base) : base(base) {}
    shared_ptr<Material> __call__0() { return base; }
    shared_ptr<Material> __call__2(const PyObject*, const PyObject*) { return base; }
};

void register_PythonEvalMaterial_super() {
    py::class_<MaterialSuper>("super", py::no_init)
        .def("__call__", &MaterialSuper::__call__0)
        .def("__call__", &MaterialSuper::__call__2);
    py::delattr(py::scope(), "super");
}

class PythonEvalMaterial : public MaterialWithBase {
    shared_ptr<PythonEvalMaterialConstructor> cls;

    py::object self;

    Parameters params;

    friend struct PythonEvalMaterialConstructor;

    template <typename RETURN> inline RETURN call(PyObject* fun, py::dict& locals, const char* funname) const {
        locals["self"] = self;
        locals["super"] = MaterialSuper(base);
        try {
            if (PyCode_Check(fun))
                return py::extract<RETURN>(py::handle<>(PyEval_EvalCode(fun, cls->globals.ptr(), locals.ptr())).get());
            else {
                py::tuple args;
                return py::extract<RETURN>(py::handle<>(PyObject_Call(fun, args.ptr(), locals.ptr())).get());
            }
        } catch (py::error_already_set&) {
            PyObject *ptype, *pvalue, *ptraceback;
            PyErr_Fetch(&ptype, &pvalue, &ptraceback);
            Py_XDECREF(ptraceback);
            std::string type, value;
            if (ptype) {
                type = py::extract<std::string>(py::object(py::handle<>(ptype)).attr("__name__"));
                type = ": " + type;
            }
            if (pvalue) {
                value = py::extract<std::string>(py::str(py::handle<>(pvalue)));
                value = ": " + value;
            }
            throw ValueError("error in custom material function <{1}> of '{0}'{2}{3}", this->name(), funname, type, value);
        }
    }

  public:
    PythonEvalMaterial(const shared_ptr<PythonEvalMaterialConstructor>& constructor, const shared_ptr<Material>& base)
        : MaterialWithBase(base), cls(constructor) {
        // This is an ugly hack using aliasing shared_ptr constructor. However, this is the only way to make a Python
        // object out of an existing normal pointer. Luckily there is very little chance anyone will ever store this,
        // as this object is accessible only to expression specified in <materials> section. The downside is that
        // if anyone uses some trick to store it and `this` gets deleted, there is no way to prevent the crash...
        self = py::object(shared_ptr<Material>(shared_ptr<Material>(), this));
    }

    // Here there are overridden methods from Material class

    OmpLockGuard<OmpNestLock> lock() const override { return OmpLockGuard<OmpNestLock>(python_omp_lock); }

    bool isEqual(const Material& other) const override {
        auto theother = static_cast<const PythonEvalMaterial&>(other);
        OmpLockGuard<OmpNestLock> lock(python_omp_lock);
        return cls == theother.cls && bool(base) == bool(theother.base) && (!base || base->str() == theother.base->str()) &&
               params == theother.params && self.attr("__dict__") == theother.self.attr("__dict__");
    }

    std::string name() const override { return cls->materialName; }
    Material::Kind kind() const override { return (cls->kind == Material::EMPTY) ? base->kind() : cls->kind; }
    Material::ConductivityType condtype() const override {
        return (cls->condtype == Material::CONDUCTIVITY_UNDETERMINED) ? base->condtype() : cls->condtype;
    }

    std::string str() const override { return params.str(); }

    double doping() const override {
        if (isnan(params.doping)) {
            if (base)
                return base->doping();
            else
                return 0.;
        } else
            return params.doping;
    }

    Composition composition() const override { return params.composition; }

#define PYTHON_EVAL_CALL_0(rtype, fun)               \
    if (cls->cache.fun) return *cls->cache.fun;      \
    if (cls->fun == NULL) return base->fun();        \
    OmpLockGuard<OmpNestLock> lock(python_omp_lock); \
    py::dict locals;                                 \
    return call<rtype>(cls->fun, locals, BOOST_PP_STRINGIZE(fun));

#define PYTHON_EVAL_CALL_1(rtype, fun, arg1)         \
    if (cls->cache.fun) return *cls->cache.fun;      \
    if (cls->fun == NULL) return base->fun(arg1);    \
    OmpLockGuard<OmpNestLock> lock(python_omp_lock); \
    py::dict locals;                                 \
    locals[BOOST_PP_STRINGIZE(arg1)] = arg1;         \
    return call<rtype>(cls->fun, locals, BOOST_PP_STRINGIZE(fun));

#define PYTHON_EVAL_CALL_2(rtype, fun, arg1, arg2)      \
    if (cls->cache.fun) return *cls->cache.fun;         \
    if (cls->fun == NULL) return base->fun(arg1, arg2); \
    OmpLockGuard<OmpNestLock> lock(python_omp_lock);    \
    py::dict locals;                                    \
    locals[BOOST_PP_STRINGIZE(arg1)] = arg1;            \
    locals[BOOST_PP_STRINGIZE(arg2)] = arg2;            \
    return call<rtype>(cls->fun, locals, BOOST_PP_STRINGIZE(fun));

#define PYTHON_EVAL_CALL_3(rtype, fun, arg1, arg2, arg3)      \
    if (cls->cache.fun) return *cls->cache.fun;               \
    if (cls->fun == NULL) return base->fun(arg1, arg2, arg3); \
    OmpLockGuard<OmpNestLock> lock(python_omp_lock);          \
    py::dict locals;                                          \
    locals[BOOST_PP_STRINGIZE(arg1)] = arg1;                  \
    locals[BOOST_PP_STRINGIZE(arg2)] = arg2;                  \
    locals[BOOST_PP_STRINGIZE(arg3)] = arg3;                  \
    return call<rtype>(cls->fun, locals, BOOST_PP_STRINGIZE(fun));

#define PYTHON_EVAL_CALL_4(rtype, fun, arg1, arg2, arg3, arg4)      \
    if (cls->cache.fun) return *cls->cache.fun;                     \
    if (cls->fun == NULL) return base->fun(arg1, arg2, arg3, arg4); \
    OmpLockGuard<OmpNestLock> lock(python_omp_lock);                \
    py::dict locals;                                                \
    locals[BOOST_PP_STRINGIZE(arg1)] = arg1;                        \
    locals[BOOST_PP_STRINGIZE(arg2)] = arg2;                        \
    locals[BOOST_PP_STRINGIZE(arg3)] = arg3;                        \
    locals[BOOST_PP_STRINGIZE(arg4)] = arg4;                        \
    return call<rtype>(cls->fun, locals, BOOST_PP_STRINGIZE(fun));

    double lattC(double T, char x) const override { PYTHON_EVAL_CALL_2(double, lattC, T, x) }
    double Eg(double T, double e, char point) const override {
        if (cls->cache.Eg) return *cls->cache.Eg;
        if (cls->Eg != NULL) {
            OmpLockGuard<OmpNestLock> lock(python_omp_lock);
            py::dict locals;
            locals["T"] = T;
            locals["e"] = e;
            locals["point"] = point;
            return call<double>(cls->Eg, locals, "Eg");
        }
        if ((cls->VB != NULL || cls->cache.VB) && (cls->CB != NULL || cls->cache.CB)) return CB(T, e, point) - VB(T, e, point, 'H');
        return base->Eg(T, e, point);
    }
    double CB(double T, double e, char point) const override {
        if (cls->cache.CB) return *cls->cache.CB;
        if (cls->CB != NULL) {
            OmpLockGuard<OmpNestLock> lock(python_omp_lock);
            py::dict locals;
            locals["T"] = T;
            locals["e"] = e;
            locals["point"] = point;
            return call<double>(cls->CB, locals, "CB");
        }
        if (cls->VB != NULL || cls->cache.VB || cls->Eg != NULL || cls->cache.Eg) return VB(T, e, point, 'H') + Eg(T, e, point);
        return base->CB(T, e, point);
    }
    double VB(double T, double e, char point, char hole) const override {
        if (cls->cache.VB) return *cls->cache.VB;
        if (cls->VB != NULL) {
            OmpLockGuard<OmpNestLock> lock(python_omp_lock);
            py::dict locals;
            locals["T"] = T;
            locals["e"] = e;
            locals["point"] = point;
            locals["hole"] = hole;
            return call<double>(cls->VB, locals, "VB");
        }
        if (cls->CB != NULL || cls->cache.CB) return CB(T, e, point) - Eg(T, e, point);
        return base->VB(T, e, point, hole);
    }
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
        if (cls->cache.D) {
            return *cls->cache.D;
        }
        if (cls->D != NULL) {
            OmpLockGuard<OmpNestLock> lock(python_omp_lock);
            py::dict locals;
            locals["T"] = T;
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
        py::dict locals;
        locals["lam"] = lam;
        locals["T"] = T;
        locals["n"] = n;
        return call<double>(cls->nr, locals, "nr");
    }
    double absp(double lam, double T) const override {
        if (cls->cache.absp) return *cls->cache.absp;
        if (cls->absp == NULL) return base->absp(lam, T);
        OmpLockGuard<OmpNestLock> lock(python_omp_lock);
        py::dict locals;
        locals["lam"] = lam;
        locals["T"] = T;
        return call<double>(cls->absp, locals, "absp");
    }
    dcomplex Nr(double lam, double T, double n = .0) const override {
        if (cls->cache.Nr) return *cls->cache.Nr;
        if (cls->Nr != NULL) {
            OmpLockGuard<OmpNestLock> lock(python_omp_lock);
            py::dict locals;
            locals["lam"] = lam;
            locals["T"] = T;
            locals["n"] = n;
            return call<dcomplex>(cls->Nr, locals, "Nr");
        }
        if (cls->nr != NULL || cls->absp != NULL || cls->cache.nr || cls->cache.absp)
            return dcomplex(nr(lam, T, n), -7.95774715459e-09 * absp(lam, T) * lam);
        return base->Nr(lam, T, n);
    }
    Tensor3<dcomplex> Eps(double lam, double T, double n = .0) const override {
        if (cls->cache.Eps) return *cls->cache.Eps;
        if (cls->Eps != NULL) {
            OmpLockGuard<OmpNestLock> lock(python_omp_lock);
            py::dict locals; locals["lam"] = lam; locals["T"] = T; locals["n"] = n;
            return call<Tensor3<dcomplex>>(cls->Eps, locals, "Eps");
        }
        if (cls->cache.Nr) {
            dcomplex nc = *cls->cache.Nr;
            nc *= nc;
            return Tensor3<dcomplex>(nc, nc, nc, 0.);
        }
        if (cls->Nr != NULL) {
            OmpLockGuard<OmpNestLock> lock(python_omp_lock);
            py::dict locals;
            locals["lam"] = lam;
            locals["T"] = T;
            locals["n"] = n;
            dcomplex nc = call<dcomplex>(cls->Nr, locals, "Nr");
            nc *= nc;
            return Tensor3<dcomplex>(nc, nc, nc, 0.);
        }
        if (cls->nr != NULL || cls->absp != NULL || cls->cache.nr || cls->cache.absp) {
            OmpLockGuard<OmpNestLock> lock(python_omp_lock);
            dcomplex nc(nr(lam, T, n), -7.95774715459e-09 * absp(lam, T) * lam);
            nc *= nc;
            return Tensor3<dcomplex>(nc, nc, nc, 0.);
        }
        return base->Eps(lam, T, n);
    }

    Tensor2<double> mobe(double T) const override{
        PYTHON_EVAL_CALL_1(Tensor2<double>, mobe, T)} Tensor2<double> mobh(double T) const override {
        PYTHON_EVAL_CALL_1(Tensor2<double>, mobh, T)
    }
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

inline shared_ptr<Material> PythonEvalMaterialConstructor::operator()(const Material::Composition& composition,
                                                                      double doping) const {
    OmpLockGuard<OmpNestLock> lock(python_omp_lock);
    auto material = plask::make_shared<PythonEvalMaterial>(self.lock(), base(composition, doping));
    material->params = Material::Parameters(materialName, true);
    if (alloy) material->params.composition = Material::completeComposition(composition);
    material->params.doping = doping;
    return material;
}

void PythonManager::loadMaterial(XMLReader& reader) {
    std::string material_name, base_name;
    bool alloy;

    shared_ptr<PythonEvalMaterialConstructor> constructor;

    try {
        material_name = reader.requireAttribute("name");
        base_name = reader.requireAttribute("base");
        alloy = reader.getAttribute<bool>("alloy", false);
        constructor = plask::make_shared<PythonEvalMaterialConstructor>(material_name, base_name, alloy, globals);
        constructor->self = constructor;
    } catch (py::error_already_set&) {
        if (draft)
            PyErr_Clear();
        else
            throw;
    } catch (const std::runtime_error& err) {
        throwErrorIfNotDraft(XMLException(reader, err.what()));
    }
    if (!constructor)
        constructor = plask::make_shared<PythonEvalMaterialConstructor>(
            material_name, shared_ptr<Material>(new DummyMaterial(base_name)), alloy, globals);
    constructor->self = constructor;

#define COMPILE_PYTHON_MATERIAL_FUNCTION_(funcname, func, args)                                                      \
    else if (reader.getNodeName() == funcname) {                                                                     \
        constructor->func = compilePythonFromXml(reader, *this, args, globals);                                      \
        if (PyCode_Check(constructor->func.ref())) {                                                                 \
            try {                                                                                                    \
                py::dict locals;                                                                                     \
                constructor->cache.func.reset(                                                                       \
                    py::extract<typename std::remove_reference<decltype(*constructor->cache.func)>::type>(           \
                        py::handle<>(PyEval_EvalCode(constructor->func.ref(), globals.ptr(), locals.ptr())).get())); \
                writelog(LOG_DEBUG, "Cached parameter '" funcname "' in material '{0}'", material_name);             \
            } catch (py::error_already_set&) {                                                                       \
                PyErr_Clear();                                                                                       \
            }                                                                                                        \
        }                                                                                                            \
    }

#define COMPILE_PYTHON_MATERIAL_FUNCTION(func, args) \
    COMPILE_PYTHON_MATERIAL_FUNCTION_(BOOST_PP_STRINGIZE(func), func, "self, " args ", super")

    try {
        while (reader.requireTagOrEnd()) {
            if (reader.getNodeName() == "condtype") {
                auto condname = reader.requireTextInCurrentTag();
                Material::ConductivityType condtype = (condname == "n" || condname == "N")   ? Material::CONDUCTIVITY_N
                                                      : (condname == "i" || condname == "I") ? Material::CONDUCTIVITY_I
                                                      : (condname == "p" || condname == "P") ? Material::CONDUCTIVITY_P
                                                      : (condname == "other" || condname == "OTHER")
                                                          ? Material::CONDUCTIVITY_OTHER
                                                          : Material::CONDUCTIVITY_UNDETERMINED;
                if (condtype == Material::CONDUCTIVITY_UNDETERMINED)
                    throw XMLException(
                        format("XML line {0} in <{1}>", reader.getLineNr(), "condtype"),
                        "Material parameter syntax error, condtype must be given as one of: n, i, p, other (or: N, I, P, OTHER)");
                constructor->condtype = condtype;
            }  // else if
            COMPILE_PYTHON_MATERIAL_FUNCTION(lattC, "T, x")
            COMPILE_PYTHON_MATERIAL_FUNCTION(Eg, "T, e, point")
            COMPILE_PYTHON_MATERIAL_FUNCTION(CB, "T, e, point")
            COMPILE_PYTHON_MATERIAL_FUNCTION(VB, "T, e, point, hole")
            COMPILE_PYTHON_MATERIAL_FUNCTION(Dso, "T, e")
            COMPILE_PYTHON_MATERIAL_FUNCTION(Mso, "T, e")
            COMPILE_PYTHON_MATERIAL_FUNCTION(Me, "T, e, point")
            COMPILE_PYTHON_MATERIAL_FUNCTION(Mhh, "T, e")
            COMPILE_PYTHON_MATERIAL_FUNCTION(Mlh, "T, e")
            COMPILE_PYTHON_MATERIAL_FUNCTION(Mh, "T, e")
            COMPILE_PYTHON_MATERIAL_FUNCTION(ac, "T")
            COMPILE_PYTHON_MATERIAL_FUNCTION(av, "T")
            COMPILE_PYTHON_MATERIAL_FUNCTION(b, "T")
            COMPILE_PYTHON_MATERIAL_FUNCTION(d, "T")
            COMPILE_PYTHON_MATERIAL_FUNCTION(c11, "T")
            COMPILE_PYTHON_MATERIAL_FUNCTION(c12, "T")
            COMPILE_PYTHON_MATERIAL_FUNCTION(c44, "T")
            COMPILE_PYTHON_MATERIAL_FUNCTION(eps, "T")
            COMPILE_PYTHON_MATERIAL_FUNCTION(chi, "T, e, point")
            COMPILE_PYTHON_MATERIAL_FUNCTION_("Na", Na, "self, super")
            COMPILE_PYTHON_MATERIAL_FUNCTION_("Nd", Nd, "self, super")
            COMPILE_PYTHON_MATERIAL_FUNCTION(Ni, "T")
            COMPILE_PYTHON_MATERIAL_FUNCTION(Nf, "T")
            COMPILE_PYTHON_MATERIAL_FUNCTION(EactD, "T")
            COMPILE_PYTHON_MATERIAL_FUNCTION(EactA, "T")
            COMPILE_PYTHON_MATERIAL_FUNCTION(mob, "T")
            COMPILE_PYTHON_MATERIAL_FUNCTION(cond, "T")
            COMPILE_PYTHON_MATERIAL_FUNCTION(A, "T")
            COMPILE_PYTHON_MATERIAL_FUNCTION(B, "T")
            COMPILE_PYTHON_MATERIAL_FUNCTION(C, "T")
            COMPILE_PYTHON_MATERIAL_FUNCTION(D, "T")
            COMPILE_PYTHON_MATERIAL_FUNCTION(thermk, "T, h")
            COMPILE_PYTHON_MATERIAL_FUNCTION(dens, "T")
            COMPILE_PYTHON_MATERIAL_FUNCTION(cp, "T")
            COMPILE_PYTHON_MATERIAL_FUNCTION(nr, "lam, T, n")
            COMPILE_PYTHON_MATERIAL_FUNCTION(absp, "lam, T")
            COMPILE_PYTHON_MATERIAL_FUNCTION(Nr, "lam, T, n")
            COMPILE_PYTHON_MATERIAL_FUNCTION(Eps, "lam, T, n")
            COMPILE_PYTHON_MATERIAL_FUNCTION(mobe, "T")
            COMPILE_PYTHON_MATERIAL_FUNCTION(mobh, "T")
            COMPILE_PYTHON_MATERIAL_FUNCTION(taue, "T")
            COMPILE_PYTHON_MATERIAL_FUNCTION(tauh, "T")
            COMPILE_PYTHON_MATERIAL_FUNCTION(Ce, "T")
            COMPILE_PYTHON_MATERIAL_FUNCTION(Ch, "T")
            COMPILE_PYTHON_MATERIAL_FUNCTION(e13, "T")
            COMPILE_PYTHON_MATERIAL_FUNCTION(e15, "T")
            COMPILE_PYTHON_MATERIAL_FUNCTION(e33, "T")
            COMPILE_PYTHON_MATERIAL_FUNCTION(c13, "T")
            COMPILE_PYTHON_MATERIAL_FUNCTION(c33, "T")
            COMPILE_PYTHON_MATERIAL_FUNCTION(Psp, "T")
            COMPILE_PYTHON_MATERIAL_FUNCTION_("y1", y1, "self, super")
            COMPILE_PYTHON_MATERIAL_FUNCTION_("y2", y2, "self, super")
            COMPILE_PYTHON_MATERIAL_FUNCTION_("y3", y3, "self, super")

            else throw XMLUnexpectedElementException(reader, "material parameter tag");
        }

        if (alloy)
            MaterialsDB::getDefault().addAlloy(constructor);
        else
            MaterialsDB::getDefault().addSimple(constructor);
    } catch (py::error_already_set&) {
        if (draft)
            PyErr_Clear();
        else
            throw;
    } catch (const std::runtime_error& err) {
        throwErrorIfNotDraft(XMLException(reader, err.what()));
    }
}

}}  // namespace plask::python

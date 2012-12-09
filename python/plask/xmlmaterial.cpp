#include "python_globals.h"

#include <plask/utils/string.h>
#include <plask/utils/xml/reader.h>
#include <plask/material/db.h>

namespace plask { namespace python {

/**
 * Wrapper for Material read from XML of type eval
 * For all virtual functions it calls Python derivatives
 */
class PythonEvalMaterial;

extern py::dict xml_globals;

struct PythonEvalMaterialConstructor: public MaterialsDB::MaterialConstructor {

    weak_ptr<PythonEvalMaterialConstructor> self;
    MaterialsDB* db;

    std::string base;
    Material::Kind kind;
    Material::ConductivityType condtype;

    PyCodeObject
        *lattC, *Eg, *CBO, *VBO, *Dso, *Mso, *Me, *Mhh, *Mlh, *Mh, *ac, *av, *b, *c11, *c12, *eps, *chi,
        *Nc, *Nv, *Ni, *Nf, *EactD, *EactA, *mob, *cond, *A, *B, *C, *D,
        *thermk, *dens, *cp, *nr, *absp, *nR, *nR_tensor;

    PythonEvalMaterialConstructor(const std::string& name) :
        MaterialsDB::MaterialConstructor(name), base(""), kind(Material::NONE), condtype(Material::CONDUCTIVITY_UNDETERMINED),
        lattC(NULL), Eg(NULL), CBO(NULL), VBO(NULL), Dso(NULL), Mso(NULL), Me(NULL),
        Mhh(NULL), Mlh(NULL), Mh(NULL), ac(NULL), av(NULL), b(NULL), c11(NULL), c12(NULL), eps(NULL), chi(NULL), Nc(NULL), Nv(NULL), Ni(NULL), Nf(NULL),
        EactD(NULL), EactA(NULL), mob(NULL), cond(NULL), A(NULL), B(NULL), C(NULL), D(NULL),
        thermk(NULL), dens(NULL), cp(NULL), nr(NULL), absp(NULL), nR(NULL), nR_tensor(NULL) {}

    PythonEvalMaterialConstructor(const std::string& name, const std::string& base) :
        MaterialsDB::MaterialConstructor(name), base(base), kind(Material::NONE), condtype(Material::CONDUCTIVITY_UNDETERMINED),
        lattC(NULL), Eg(NULL), CBO(NULL), VBO(NULL), Dso(NULL), Mso(NULL), Me(NULL),
        Mhh(NULL), Mlh(NULL), Mh(NULL), ac(NULL), av(NULL), b(NULL), c11(NULL), c12(NULL), eps(NULL), chi(NULL), Nc(NULL), Nv(NULL), Ni(NULL), Nf(NULL),
        EactD(NULL), EactA(NULL), mob(NULL), cond(NULL), A(NULL), B(NULL), C(NULL), D(NULL),
        thermk(NULL), dens(NULL), cp(NULL), nr(NULL), absp(NULL), nR(NULL), nR_tensor(NULL) {}

    virtual ~PythonEvalMaterialConstructor() {
        Py_XDECREF(lattC); Py_XDECREF(Eg); Py_XDECREF(CBO); Py_XDECREF(VBO); Py_XDECREF(Dso); Py_XDECREF(Mso); Py_XDECREF(Me);
        Py_XDECREF(Mhh); Py_XDECREF(Mlh); Py_XDECREF(Mh); Py_XDECREF(ac); Py_XDECREF(av); Py_XDECREF(b); Py_XDECREF(c11); Py_XDECREF(c12); Py_XDECREF(eps); Py_XDECREF(chi);
        Py_XDECREF(Nc); Py_XDECREF(Nv); Py_XDECREF(Ni); Py_XDECREF(Nf); Py_XDECREF(EactD); Py_XDECREF(EactA);
        Py_XDECREF(mob); Py_XDECREF(cond); Py_XDECREF(A); Py_XDECREF(B); Py_XDECREF(C); Py_XDECREF(D);
        Py_XDECREF(thermk); Py_XDECREF(dens); Py_XDECREF(cp);
        Py_XDECREF(nr); Py_XDECREF(absp); Py_XDECREF(nR); Py_XDECREF(nR_tensor);
    }

    inline shared_ptr<Material> operator()(const Material::Composition& composition, Material::DopingAmountType doping_amount_type, double doping_amount) const;
};

class PythonEvalMaterial : public Material
{
    shared_ptr<PythonEvalMaterialConstructor> cls;
    shared_ptr<Material> base;

    py::object self;

    template <typename RETURN>
    RETURN call(PyCodeObject *fun, const py::dict& locals) const {
#       if PY_VERSION_HEX >= 0x03000000
            PyObject* result = PyEval_EvalCode((PyObject*)fun, xml_globals.ptr(), locals.ptr());
#       else
            PyObject* result = PyEval_EvalCode(fun, xml_globals.ptr(), locals.ptr());
#       endif
        if (!result) throw py::error_already_set();
        return py::extract<RETURN>(result);
    }

  public:

    PythonEvalMaterial(const shared_ptr<PythonEvalMaterialConstructor>& constructor, const shared_ptr<Material>& base,
                       const Material::Composition& composition, Material::DopingAmountType doping_amount_type, double doping_amount) :
        cls(constructor), base(base) {
        self = py::object(base);
        if (doping_amount_type == Material::DOPANT_CONCENTRATION) self.attr("dc") = doping_amount;
        else if (doping_amount_type == Material::CARRIER_CONCENTRATION) self.attr("cc") = doping_amount;
    }

    // Here there are overridden methods from Material class

    virtual std::string name() const { return cls->materialName; }
    virtual Material::Kind kind() const { return (cls->kind == Material::NONE)? base->kind() : cls->kind; }
    virtual Material::ConductivityType condtype() const { return (cls->condtype == Material::CONDUCTIVITY_UNDETERMINED)? base->condtype() : cls->condtype; }

#   define PYTHON_EVAL_CALL_1(rtype, fun, arg1) \
        if (cls->fun == NULL) return base->fun(arg1); \
        py::dict locals; locals["self"] = self; locals[BOOST_PP_STRINGIZE(arg1)] = arg1; \
        return call<rtype>(cls->fun, locals);

#   define PYTHON_EVAL_CALL_2(rtype, fun, arg1, arg2) \
        if (cls->fun == NULL) return base->fun(arg1, arg2); \
        py::dict locals; locals["self"] = self; locals[BOOST_PP_STRINGIZE(arg1)] = arg1; locals[BOOST_PP_STRINGIZE(arg2)] = arg2; \
        return call<rtype>(cls->fun, locals);

    virtual double lattC(double T, char x) const { PYTHON_EVAL_CALL_2(double, lattC, T, x) }
    virtual double Eg(double T, const char Point) const { PYTHON_EVAL_CALL_2(double, Eg, T, Point) }
    virtual double CBO(double T, const char Point) const { PYTHON_EVAL_CALL_2(double, CBO, T, Point) }
    virtual double VBO(double T) const { PYTHON_EVAL_CALL_1(double, VBO, T) }
    virtual double Dso(double T) const { PYTHON_EVAL_CALL_1(double, Dso, T) }
    virtual double Mso(double T) const { PYTHON_EVAL_CALL_1(double, Mso, T) }
    virtual Tensor2<double> Me(double T, const char Point) const { PYTHON_EVAL_CALL_2(Tensor2<double>, Me, T, Point) }
    virtual Tensor2<double> Mhh(double T, const char Point) const { PYTHON_EVAL_CALL_2(Tensor2<double>, Mhh, T, Point) }
    virtual Tensor2<double> Mlh(double T, const char Point) const { PYTHON_EVAL_CALL_2(Tensor2<double>, Mlh, T, Point) }
    virtual Tensor2<double> Mh(double T, char EqType) const { PYTHON_EVAL_CALL_2(Tensor2<double>, Mh, T, EqType) }
    virtual double ac(double T) const { PYTHON_EVAL_CALL_1(double, ac, T) }
    virtual double av(double T) const { PYTHON_EVAL_CALL_1(double, av, T) }
    virtual double b(double T) const { PYTHON_EVAL_CALL_1(double, b, T) }
    virtual double c11(double T) const { PYTHON_EVAL_CALL_1(double, c11, T) }
    virtual double c12(double T) const { PYTHON_EVAL_CALL_1(double, c12, T) }
    virtual double eps(double T) const { PYTHON_EVAL_CALL_1(double, eps, T) }
    virtual double chi(double T, const char Point) const { PYTHON_EVAL_CALL_2(double, chi, T, Point) }
    virtual double Nc(double T, const char Point) const { PYTHON_EVAL_CALL_2(double, Nc, T, Point) }
    virtual double Nv(double T) const { PYTHON_EVAL_CALL_1(double, Nv, T) }
    virtual double Ni(double T) const { PYTHON_EVAL_CALL_1(double, Ni, T) }
    virtual double Nf(double T) const { PYTHON_EVAL_CALL_1(double, Nf, T) }
    virtual double EactD(double T) const { PYTHON_EVAL_CALL_1(double, EactD, T) }
    virtual double EactA(double T) const { PYTHON_EVAL_CALL_1(double, EactA, T) }
    virtual Tensor2<double> mob(double T) const { PYTHON_EVAL_CALL_1(Tensor2<double>, mob, T) }
    virtual Tensor2<double> cond(double T) const { PYTHON_EVAL_CALL_1(Tensor2<double>, cond, T) }
    virtual double A(double T) const { PYTHON_EVAL_CALL_1(double, A, T) }
    virtual double B(double T) const { PYTHON_EVAL_CALL_1(double, B, T) }
    virtual double C(double T) const { PYTHON_EVAL_CALL_1(double, C, T) }
    virtual double D(double T) const { PYTHON_EVAL_CALL_1(double, D, T) }
    virtual Tensor2<double> thermk(double T) const { const double t = INFINITY; PYTHON_EVAL_CALL_2(Tensor2<double>, thermk, T, t) }
    virtual Tensor2<double> thermk(double T, double t)  const { PYTHON_EVAL_CALL_2(Tensor2<double>, thermk, T, t) }
    virtual double dens(double T) const { PYTHON_EVAL_CALL_1(double, dens, T) }
    virtual double cp(double T) const { PYTHON_EVAL_CALL_1(double, cp, T) }
    virtual double nr(double wl, double T) const { PYTHON_EVAL_CALL_2(double, nr, wl, T) }
    virtual double absp(double wl, double T) const { PYTHON_EVAL_CALL_2(double, absp, wl, T) }
    virtual dcomplex nR(double wl, double T) const {
        py::dict locals; locals["self"] = self; locals["wl"] = wl; locals["T"] = T;
        if (cls->nR != NULL) {
#           if PY_VERSION_HEX >= 0x03000000
                PyObject* result = PyEval_EvalCode((PyObject*)(cls->nR), xml_globals.ptr(), locals.ptr());
#           else
                PyObject* result = PyEval_EvalCode(cls->nR, xml_globals.ptr(), locals.ptr());
#           endif
            if (!result) throw py::error_already_set();
            return py::extract<dcomplex>(result);
        }
        else return dcomplex(nr(wl, T), -7.95774715459e-09 * absp(wl, T)*wl);
    }
    virtual Tensor3<dcomplex> nR_tensor(double wl, double T) const {
        py::dict locals; locals["self"] = self; locals["wl"] = wl; locals["T"] = T;
        if (cls->nR != NULL) {
#           if PY_VERSION_HEX >= 0x03000000
                PyObject* result = PyEval_EvalCode((PyObject*)(cls->nR_tensor), xml_globals.ptr(), locals.ptr());
#           else
                PyObject* result = PyEval_EvalCode(cls->nR_tensor, xml_globals.ptr(), locals.ptr());
#           endif
            if (!result) throw py::error_already_set();
            return py::extract<Tensor3<dcomplex>>(result);
        } else {
            dcomplex n = nR(wl, T);
            return Tensor3<dcomplex>(n, n, n, 0., 0.);
        }
    }
    // End of overridden methods

};

inline shared_ptr<Material> PythonEvalMaterialConstructor::operator()(const Material::Composition& composition, Material::DopingAmountType doping_amount_type, double doping_amount) const {
    shared_ptr<Material> base_obj;
    if (base != "") {
        if (base.find("=") != std::string::npos)
            base_obj = this->db->get(base);
        else
            base_obj = this->db->get(base, doping_amount_type, doping_amount);
    }
    else base_obj = make_shared<EmptyMaterial>();
    return make_shared<PythonEvalMaterial>(self.lock(), base_obj, composition, doping_amount_type, doping_amount);
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

        Material::ConductivityType condtype = Material::CONDUCTIVITY_UNDETERMINED;
        auto condname = reader.getAttribute("condtype");
        if (condname) {
            condtype = (*condname == "n" || *condname == "N")? Material::CONDUCTIVITY_N :
                        (*condname == "i" || *condname == "I")? Material::CONDUCTIVITY_I :
                        (*condname == "p" || *condname == "P")? Material::CONDUCTIVITY_P :
                        (*condname == "other" || *condname == "OTHER")? Material::CONDUCTIVITY_OTHER :
                         Material::CONDUCTIVITY_UNDETERMINED;
            if (condtype == Material::CONDUCTIVITY_UNDETERMINED) throw XMLBadAttrException(reader, "condtype", *condname);
        }

        constructor = make_shared<PythonEvalMaterialConstructor>(name);
        constructor->kind = kind;
        constructor->condtype = condtype;
    }
    constructor->self = constructor;
    constructor->db = &materialsDB;

    auto trim = [](const char* s) -> const char* {
        for(; *s != 0 && std::isspace(*s); ++s)
        ;
        return s;
    };

#   if PY_VERSION_HEX >= 0x03000000

#       define COMPILE_PYTHON_MATERIAL_FUNCTION(func) \
        else if (reader.getNodeName() == BOOST_PP_STRINGIZE(func)) \
            constructor->func = (PyCodeObject*)Py_CompileString(trim(reader.requireTextInCurrentTag().c_str()), BOOST_PP_STRINGIZE(func), Py_eval_input);

#       define COMPILE_PYTHON_MATERIAL_FUNCTION2(name, func) \
        else if (reader.getNodeName() == name) \
            constructor->func = (PyCodeObject*)Py_CompileString(trim(reader.requireTextInCurrentTag().c_str()), BOOST_PP_STRINGIZE(func), Py_eval_input);

#   else
        PyCompilerFlags flags { CO_FUTURE_DIVISION };

#       define COMPILE_PYTHON_MATERIAL_FUNCTION(func) \
        else if (reader.getNodeName() == BOOST_PP_STRINGIZE(func)) { \
            constructor->func = (PyCodeObject*)Py_CompileStringFlags(trim(reader.requireTextInCurrentTag().c_str()), BOOST_PP_STRINGIZE(func), Py_eval_input, &flags); \
            if (constructor->func == nullptr)  throw XMLException(format("XML line %1% in <%2%>", reader.getLineNr(), BOOST_PP_STRINGIZE(func)), "Material parameter syntax error"); \
        }

#       define COMPILE_PYTHON_MATERIAL_FUNCTION2(name, func) \
        else if (reader.getNodeName() == name) { \
            constructor->func = (PyCodeObject*)Py_CompileStringFlags(trim(reader.requireTextInCurrentTag().c_str()), BOOST_PP_STRINGIZE(func), Py_eval_input, &flags); \
            if (constructor->func == nullptr)  throw XMLException(format("XML line %1% in <%2%>", reader.getLineNr(), name), "Material parameter syntax error"); \
        }

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
        COMPILE_PYTHON_MATERIAL_FUNCTION(Mhh)
        COMPILE_PYTHON_MATERIAL_FUNCTION(Mlh)
        COMPILE_PYTHON_MATERIAL_FUNCTION(Mh)
        COMPILE_PYTHON_MATERIAL_FUNCTION(ac)
        COMPILE_PYTHON_MATERIAL_FUNCTION(av)
        COMPILE_PYTHON_MATERIAL_FUNCTION(b)
        COMPILE_PYTHON_MATERIAL_FUNCTION(c11)
        COMPILE_PYTHON_MATERIAL_FUNCTION(c12)
        COMPILE_PYTHON_MATERIAL_FUNCTION(eps)
        COMPILE_PYTHON_MATERIAL_FUNCTION(chi)
        COMPILE_PYTHON_MATERIAL_FUNCTION(Nc)
        COMPILE_PYTHON_MATERIAL_FUNCTION(Nv)
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
        COMPILE_PYTHON_MATERIAL_FUNCTION(nR)
        COMPILE_PYTHON_MATERIAL_FUNCTION(nR_tensor)
        else throw XMLUnexpectedElementException(reader, "material parameter tag");
    }

    materialsDB.addSimple(constructor);
}

}} // namespace plask::python

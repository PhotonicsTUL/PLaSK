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
#include <complex>

#include <boost/algorithm/string.hpp>

#define PY_ARRAY_UNIQUE_SYMBOL PLASK_ARRAY_API

#include "python/globals.hpp"
#include "python/manager.hpp"
#include "python/numpy.hpp"
#include "python/util/raw_constructor.hpp"
#include <frameobject.h> // for Python traceback
#include <plask/utils/warnings.hpp>
#include "python/method_override.hpp"

#include "common/fem/python.hpp"

#include <plask/version.hpp>
#include <plask/exceptions.hpp>
#include <plask/mesh/interpolation.hpp>
#include <plask/memory.hpp>
#include <plask/solver.hpp>

using namespace plask::python;

// Declare some initialization functions
namespace plask { namespace python {

PLASK_PYTHON_API void createPythonLogger();

void initMaterials();
void initGeometry();

void register_manager();
void register_xml_writer();

void register_xml_reader();

void register_vectors();
void register_tensors();
void register_mesh();
void register_providers();

void register_data_vectors();

void register_python_log();

void register_standard_properties();

PLASK_PYTHON_API PyObject* pyXmlError;
PLASK_PYTHON_API PyObject* pyGeometryException;
PLASK_PYTHON_API PyObject* pyComputationError;

PLASK_PYTHON_API std::string getPythonExceptionMessage() {
    PyObject *value, *type, *original_traceback;
    PyErr_Fetch(&type, &value, &original_traceback);
    PyErr_NormalizeException(&type, &value, &original_traceback);
    py::handle<> value_h(value), type_h(type), original_traceback_h(py::allow_null(original_traceback));
    return py::extract<std::string>(py::str(value_h));
}

PLASK_PYTHON_API py::object py_eval(std::string string, py::object global, py::object local)
{
    // Set suitable default values for global and local dicts.
    if (global.is_none()) {
        if (PyObject *g = PyEval_GetGlobals())
            global = py::object(py::detail::borrowed_reference(g));
        else
            global = py::dict();
    }
    if (local.is_none()) local = global;
    PyObject* result = PyRun_String(string.c_str(), Py_eval_input, global.ptr(), local.ptr());
    if (!result) py::throw_error_already_set();
    return py::object(py::detail::new_reference(result));
}


// Parallel locking

#ifdef OPENMP_FOUND

struct PythonOmpEnv : public OmpEnv {
    PyThreadState* thread_state;
    virtual void enable() { thread_state = PyEval_SaveThread(); };
    virtual void disable() { PyEval_RestoreThread(thread_state); };
} python_omp_env;

#endif

PLASK_PYTHON_API OmpPythonLock python_omp_lock;

// Config
PLASK_PYTHON_API AxisNames current_axes = AxisNames::axisNamesRegister.get("ltv");
PLASK_PYTHON_API bool ufunc_ignore_errors = true;

static LoggingConfig getLoggingConfig(const Config&) {
    return LoggingConfig();
}

std::string Config::axes_name() const {
    return current_axes.str();
}
void Config::set_axes(std::string axes) {
    current_axes = AxisNames::axisNamesRegister.get(axes);
}

// return list of 3 axes names or throw exception if axes_names isn't fine
py::list axeslist_by_name(const std::string& axes_names) {
    AxisNames a = AxisNames::axisNamesRegister.get(axes_names);
    py::list l;
    l.append(a.byIndex[0]); l.append(a.byIndex[1]); l.append(a.byIndex[2]);
    return l;
}



bool Config::getUfuncIgnoreError() const { return ufunc_ignore_errors; }
void Config::setUfuncIgnoreError(bool value) { ufunc_ignore_errors = value; }


std::string Config::__str__() const {
    return  "axes:         " + axes_name()
        + "\nlog.colors:   " + str(LoggingConfig().getLoggingColor())
        + "\nlog.level:    " + str(py::object(maxLoglevel))
        + "\nlog.output:   " + str(LoggingConfig().getLoggingDest())
        + "\nnan_on_error: " + (ufunc_ignore_errors? "True" : "False");
    ;
}

std::string Config:: __repr__() const {
    return
        format("config.axes = '{}'", axes_name()) +
           + "\nconfig.log.colors = " + str(LoggingConfig().getLoggingColor())
           + "\nconfig.log.level = LOG_" + str(py::object(maxLoglevel))
           + "\nconfig.log.output = " + str(LoggingConfig().getLoggingDest())
           + "\nconfig.nan_on_error = " + (ufunc_ignore_errors? "True" : "False");
    ;
}

std::string LoggingConfig::__str__() const {
    return  "colors: " + str(getLoggingColor())
        + "\nlevel:  " + str(py::object(maxLoglevel))
        + "\noutput: " + str(getLoggingDest());
    ;
}

std::string LoggingConfig::__repr__() const {
    return "log.colors = " + str(getLoggingColor())
        + "\nlog.level = LOG_" + str(py::object(maxLoglevel))
        + "\nlog.output = " + str(getLoggingDest());
    ;
}


inline static void register_config()
{
    py::class_<Config> config_class("config",

        u8"Global PLaSK configuration.\n\n"

        u8"This class has only one instance and it contains global configuration options.\n"
        u8"The attributes of this class are config parameters that can be set using the\n"
        u8"``config`` object.\n\n"

        u8"Example:\n"
        u8"    >>> config.axes = 'xy'\n"
        u8"    >>> config.log.level = 'debug'\n"
        u8"    >>> print config\n"
        u8"    axes:        zxy\n"
        u8"    log.colors:  ansi\n"
        u8"    log.level:   DEBUG\n"
        u8"    log.output:  stdout\n"

        , py::no_init);
    config_class
        .def("__str__", &Config::__str__)
        .def("__repr__", &Config::__repr__)
        .add_property("axes", &Config::axes_name, &Config::set_axes,
                      u8"String representing axis names.\n\n"

                      u8"The accepted values are listed below. Each row shows different aliases for\n"
                      u8"the same axes:\n\n"

                      u8"================ ================ ================\n"
                      u8"`xyz`            `yz`             `z_up`\n"
                      u8"`zxy`            `xy`             `y_up`\n"
                      u8"`prz`            `rz`             `rad`\n"
                      u8"`ltv`                             `abs`\n"
                      u8"`long,tran,vert`                  `absolute`\n"
                      u8"================ ================ ================\n\n"

                      u8"The choice of the axes should depend on your structure. In Cartesian coordinates\n"
                      u8"you probably prefer `xyz` or `zxy`. In cylindrical ones the most natural choice\n"
                      u8"is `prz`. However, it is important to realize that any names can be chosen in\n"
                      u8"any geometry and they are fully independent from it.\n"
                     )
        .add_property("log", &getLoggingConfig,
                      u8"Settings of the logging system.\n\n"

                      u8"This setting has several subattributes listed below:\n\n"

                      u8"**color**\n"
                      u8"        System used for coloring the log messages depending on their level.\n"
                      u8"        This parameter can have on of the following values:\n\n"

                      u8"        :ansi:    Use ANSI codes for coloring. Works best in UNIX-like systems\n"
                      u8"                  (Linux, OSX) or with GUI launchers.\n"
                      u8"        :windows: Use Windows API for coloring. Available only on Windows.\n"
                      u8"        :none:    Do not perform coloring at all. Recommended when redirecting\n"
                      u8"                  output to a file.\n\n"

                      u8"        On its start PLaSK tries to automatically determine the best value for\n"
                      u8"        this option, so usually you will not need to change it.\n\n"

                      u8"**level**\n"
                      u8"        Maximum logging level. It can be one of:\n\n"

                      u8"        :CRITICAL_ERROR: Critical errors that result in program interruption.\n"
                      u8"        :ERROR:          Minor errors that do not break the whole program flow.\n"
                      u8"        :ERROR_DETAIL:   Details of the errors with more information on them.\n"
                      u8"        :WARNING:        Important warnings that you should investigate.\n"
                      u8"        :INFO:           General information of the executed operations.\n"
                      u8"        :RESULT:         Some intermediate computations results.\n"
                      u8"        :DATA:           Some data used for tracking the computations.\n"
                      u8"        :DETAIL:         Details of computations processes.\n"
                      u8"        :DEBUG:          Additional information useful for debugging PLaSK.\n\n"

                      u8"        Setting any of the above levels will instruct PLaSK to print only\n"
                      u8"        information of the specified level and above. It is recommended to\n"
                      u8"        always set the logging level at least to 'WARNING'. This setting is\n"
                      u8"        ignored when the plask option :option:`-l` is specified.\n\n"

                      u8"**output**\n"
                      u8"        Stream to which the log messages are printed. Can be either **stderr**\n"
                      u8"        (which is the default) or **stdout** (turned on for interactive mode).\n"
                     )
        .add_property("nan_on_error", &Config::getUfuncIgnoreError, &Config::setUfuncIgnoreError,
                      u8"Ignore error for array computations and return NAN.\n\n"

                      u8"If this parameter is ``True``, some solver methods will not report computation\n"
                      u8"errors if an array is passed as an argument. Instead, such methods will set\n"
                      u8"the corresponding value in the result array to NAN.\n"
                     );
    py::scope().attr("config") = Config();
}


/// XPL filename
PLASK_PYTHON_API std::string xplFilename;

PLASK_PYTHON_API void setCurrentAxes(const AxisNames& axes) { current_axes = axes; }

PLASK_PYTHON_API void setXplFilename(const std::string& filename) { xplFilename = filename; }

PLASK_PYTHON_API PyObject* getXmlErrorClass() { return pyXmlError; }



template <typename... Args>
static void printMultiLineLog(plask::LogLevel level, const std::string& msg, Args&&... args) {
    typedef boost::tokenizer<boost::char_separator<char> > LineTokenizer;
    std::string message = plask::format(msg, args...);
    LineTokenizer tokenizer(message, boost::char_separator<char>("\n\r"));
    for (LineTokenizer::const_iterator line = tokenizer.begin(), end = tokenizer.end(); line != end ; ++line)
        plask::writelog(level, *line);
}


// Print Python exception to PLaSK logging system
PLASK_PYTHON_API int printPythonException(PyObject* otype, PyObject* value, PyObject* otraceback, const char* scriptname, const char* top_frame, int scriptline) {

    if (otype == PyExc_SystemExit) {
        Py_INCREF(value);
        int exitcode;
        if (PyExceptionInstance_Check(value)) {
            PyObject* code = PyObject_GetAttrString(value, "code");
            if (code) {
                Py_DECREF(value);
                value = code;
            }
        }
        if (PyLong_Check(value))
            exitcode = (int)PyLong_AsLong(value);
        else {
            std::cerr.flush();
            std::cout.flush();
            PyObject_Print(value, stderr, Py_PRINT_RAW);
            PySys_WriteStderr("\n");
            exitcode = 1;
        }
        Py_DECREF(value);
        PyErr_Clear();
        return exitcode;
    }

    {
        PyObject* cause = PyException_GetCause(value);
        PyObject* context = PyException_GetContext(value);
        py::handle<> cause_h(py::allow_null(cause)), context_h(py::allow_null(context));
        if (cause && cause != Py_None) {
            printPythonException(cause, scriptname, top_frame, scriptline);
            writelog(LOG_ERROR_DETAIL, "The above exception was the direct cause of the following exception:");
        } else if (context && context != Py_None) {
            printPythonException(context, scriptname, top_frame, scriptline);
            writelog(LOG_ERROR_DETAIL, "During handling of the above exception, another exception occurred:");
        }
    }

    PyTypeObject* type = (PyTypeObject*)otype;
    PyTracebackObject* original_traceback = (PyTracebackObject*)otraceback;
    PyObject* value_str = PyObject_Str(value);
    std::string message = py::extract<std::string>(value_str);
    Py_DECREF(value_str);

    std::string error_name = type->tp_name;
    if (error_name.substr(0, 11) == "exceptions.") error_name = error_name.substr(11);

    if (original_traceback) {
        PyTracebackObject* traceback = original_traceback;
        while (traceback) {
            PyCodeObject* f_code =
                #if PY_VERSION_HEX >= 0x030900B1
                    PyFrame_GetCode(traceback->tb_frame);
                #else
                    traceback->tb_frame->f_code;
                #endif
            int lineno = traceback->tb_lineno + scriptline;
            #if PY_VERSION_HEX >= 0x030a0000
                if (lineno == -1) lineno = PyCode_Addr2Line(f_code, traceback->tb_lasti);
            #endif
            std::string filename = py::extract<std::string>(f_code->co_filename);
            std::string funcname = py::extract<std::string>(f_code->co_name);
            #if PY_VERSION_HEX >= 0x030900B1
                Py_XDECREF(f_code);
            #endif
            if (funcname == "<module>" && (traceback == original_traceback || (top_frame && traceback == original_traceback->tb_next)))
                funcname = top_frame? top_frame : "<script>";
            if (traceback->tb_next)
                plask::writelog(plask::LOG_ERROR_DETAIL, u8"{0} line {1}, function '{2}' calling:", filename, lineno, funcname);
            else {
                if ((PyObject*)type == PyExc_IndentationError || (PyObject*)type == PyExc_SyntaxError) {
                    plask::writelog(plask::LOG_ERROR_DETAIL, u8"{0} line {1}, function '{2}' calling:", filename, lineno, funcname);
                    std::string form = message;
                    std::size_t f = form.find(" (") + 2, l = form.rfind(" line ") + 6;
                    std::string msg = form.substr(0, f-2), file = form.substr(f, l-f-6);
                    try {
                        int lineno = boost::lexical_cast<int>(form.substr(l, form.length()-l-1)) + scriptline;
                        printMultiLineLog(plask::LOG_CRITICAL_ERROR, u8"{0} line {1}: {2}: {3}", file, lineno, error_name, msg);
                    } catch (boost::bad_lexical_cast&) {
                        printMultiLineLog(plask::LOG_CRITICAL_ERROR, u8"{0}: {1}", error_name, message);
                    }
                } else
                    printMultiLineLog(plask::LOG_CRITICAL_ERROR, u8"{0} line {1}, function '{2}': {3}: {4}",
                                      filename, lineno, funcname, error_name, message);
            }
            traceback = traceback->tb_next;
        }
    } else {
        if ((PyObject*)type == PyExc_IndentationError || (PyObject*)type == PyExc_SyntaxError) {
                std::string form = message;
                std::size_t f = form.find(" (") + 2, l = form.rfind(" line ") + 6;
                std::string msg = form.substr(0, f-2), file = form.substr(f, l-f-6);
                try {
                    int lineno = boost::lexical_cast<int>(form.substr(l, form.length()-l-1));
                    printMultiLineLog(plask::LOG_CRITICAL_ERROR, u8"{0} line {1}: {2}: {3}", file, lineno, error_name, msg);
                } catch (boost::bad_lexical_cast&) {
                    printMultiLineLog(plask::LOG_CRITICAL_ERROR, u8"{0}: {1}", error_name, message);
                }
        } else
            printMultiLineLog(plask::LOG_CRITICAL_ERROR, u8"{0}: {1}", error_name, message);
    }
    return 1;
}

// Default options for docstrings
py::docstring_options doc_options(
    true,   // show user defined
    true,   // show py signatures
    false   // show cpp signatures
);

// struct IntFromInt64 {
//     IntFromInt64() {
//         boost::python::converter::registry::push_back(&convertible, &construct, boost::python::type_id<int>());
//     }
//
//     static void* convertible(PyObject* obj) {
//         return (PyArray_Check(obj) PyArray_IsIntegerScalar(obj) && PyObject_TypeCheck(obj, &PyLong64ArrType_Type))? obj : NULL;
//         // return PyObject_TypeCheck(obj, &PyLong64ArrType_Type)? obj : NULL;
//     }
//
//     static void construct(PyObject* obj, boost::python::converter::rvalue_from_python_stage1_data* data) {
//         void* storage = ((boost::python::converter::rvalue_from_python_storage<int>*)data)->storage.bytes;
//         try {
//             if (PyArray_Check(obj)) {
//                 if (PyArray_NDIM((PyArrayObject*)obj) != 1 || PyArray_DIMS((PyArrayObject*)obj)[0] != dim) throw py::error_already_set();
//                 MakeVecFromNumpyImpl<dim,T>::call(storage, obj);
//             } else {
//                 auto seq = py::object(py::handle<>(py::borrowed(obj)));
//                 if (py::len(seq) != dim || (PyArray_Check(obj) && PyArray_NDIM((PyArrayObject*)obj) != 1)) throw py::error_already_set();
//                 py::stl_input_iterator<T> begin(seq);
//                 new(storage) Vec<dim,T>(Vec<dim,T>::fromIterator(begin));
//             }
//             data->convertible = storage;
//         } catch (py::error_already_set) {
//             throw TypeError("must provide either plask.vector or a sequence of length {0} of proper dtype", dim);
//         }
//     }
// };

// struct DoubleFromComplex {
//
//     static PyObject* ComplexWarning;
//
//     DoubleFromComplex() {
//         ComplexWarning = PyErr_NewExceptionWithDoc((char*)"plask.ComplexWarning",
//                                                    (char*)"The warning raised when casting a complex dtype to a real dtype.\n\n"
//                                                           "As implemented, casting a complex number to a real discards its imaginary\n"
//                                                           "part, but this behavior may not be what the user actually wants.\n",
//                                                    PyExc_Warning, NULL);
//         boost::python::converter::registry::push_back(&convertible, &construct, boost::python::type_id<double>());
//     }
//
//     // Determine if obj can be converted into an Aligner
//     static void* convertible(PyObject* obj) {
//         if (!PyComplex_Check(obj)) return NULL;
//         return obj;
//     }
//
//     static void construct(PyObject* obj, boost::python::converter::rvalue_from_python_stage1_data* data) {
//         void* storage = ((boost::python::converter::rvalue_from_python_storage<double>*)data)->storage.bytes;
//         plask::dcomplex cplx = py::extract<plask::dcomplex>(obj);
//         new(storage) double(real(cplx));
//         data->convertible = storage;
//         PyErr_Warn(ComplexWarning, "Casting complex values to real discards the imaginary part");
//     }
// };
// PyObject* DoubleFromComplex::ComplexWarning;


/// Solver wrapper to be inherited from Python
struct SolverWrap: public Solver, Overriden<Solver> {

    SolverWrap(PyObject* self, const std::string& name): Solver(name), Overriden(self) {}

    static shared_ptr<Solver> init(const py::tuple& args, const py::dict& kwargs) {
        PyObject* self;
        const char* name = nullptr;
        static const char *kwlist[] = { "self", "name", NULL };
        if (!PyArg_ParseTupleAndKeywords(args.ptr(), kwargs.ptr(), "O|s:__init__", (char**)kwlist, &self, &name))
            throw py::error_already_set();

        return plask::make_shared<SolverWrap>(self, name? name : "");
    }

    std::string getClassName() const override {
        return py::extract<std::string>(PyObject_GetAttrString(PyObject_GetAttrString(self, "__class__"), "__name__"));
    }

    void onInitialize() override {
        OmpLockGuard lock(python_omp_lock);
        if (overriden("on_initialize")) {
            py::call_method<void>(self, "on_initialize");
        }
    }

    void onInvalidate() override {
        OmpLockGuard lock(python_omp_lock);
        if (overriden("on_invalidate")) {
            py::call_method<void>(self, "on_invalidate");
        }
    }

    void loadConfiguration(XMLReader& source, Manager& manager) override {
        OmpLockGuard lock(python_omp_lock);
        if (overriden("load_xpl")) {
            py::call_method<void>(self, "load_xpl", boost::ref(source), boost::ref(manager));
        } else {
            Solver::loadConfiguration(source, manager);
        }

    }

};


/// Custom wrapper for XMLError
template <>
void register_exception<plask::XMLException>(PyObject* py_exc) {
    py::register_exception_translator<plask::XMLException>([=](const plask::XMLException& err) {
        py::object excType(py::handle<>(py::borrowed(py_exc)));
        py::object exc = excType(err.what());
        if (err.line == -1) exc.attr("line") = py::object();
        else exc.attr("line") = err.line;
        PyErr_SetObject(py_exc, exc.ptr());
    });
}


}} // namespace plask::python


#ifdef PRINT_STACKTRACE_ON_EXCEPTION
#   if defined(_WIN32) || defined(__WIN32__) || defined(WIN32) //win32 support
#       include <win_printstack.hpp>
#   else
#       include <backward.hpp>
        void printStack(void) {
            backward::StackTrace st;
            st.load_here(256);
            backward::Printer printer;
            printer.address = true;
            printer.print(st, stderr);
        }
#   endif
#endif


BOOST_PYTHON_MODULE(_plask)
{
    // Initialize numpy
    if (!plask_import_array()) throw(py::error_already_set());

    #ifdef OPENMP_FOUND
        plask::OmpEnabler::env = &python_omp_env;
    #endif

    py::scope scope; // Default scope

    // Config
    register_config();

    // Log
    register_python_log();

    // Manager
    register_manager();
    register_xml_writer();
    register_xml_reader();

    // Vectors and tensors
    register_vectors();
    register_tensors();

    register_vector_of<size_t>("unsigned_int");
    register_vector_of<int>("int");
    register_vector_of<double>("float");
    register_vector_of<std::complex<double>>("complex");

    // Materials
    initMaterials();

    // Geometry
    initGeometry();
    def("axeslist_by_name", axeslist_by_name);

    // Meshes
    register_mesh();

    // Data vector
    register_data_vectors();

    // Solvers
    py::class_<plask::Solver, plask::shared_ptr<plask::Solver>, boost::noncopyable>
    solver("Solver",
           u8"Base class for all solvers.\n\n"

           u8"Solver(name='')\n\n"

           u8"Args:\n"
           u8"    name: Solver name for its identification in logs.\n"

           u8"You should inherit this class if you are creating custom Python solvers\n"
           u8"in Python, which can read its configuration from the XPL file. Then you need to\n"
           u8"override the :meth:`load_xml` method, which reads the configuration. If you\n"
           u8"override :meth:`on_initialize` of :meth:`on_invalidate` methods, they will be\n"
           u8"called once on the solver initialization/invalidation.\n\n"

           u8"Example:\n"
           u8"  .. code-block:: python\n\n"

           u8"     class MySolver(Solver):\n\n"

           u8"         def __init__(self, name=''):\n"
           u8"             super().__init__(name)\n"
           u8"             self.param = 0.\n"
           u8"             self.geometry = None\n"
           u8"             self.mesh = None\n"
           u8"             self.workspace = None\n"
           u8"             self.bc = plask.mesh.Rectangular2D.BoundaryConditions()\n\n"

           u8"         def load_xpl(self, xpl, manager):\n"
           u8"             for tag in xpl:\n"
           u8"                 if tag == 'config':\n"
           u8"                     self.param = tag.get('param', self.param)\n"
           u8"                 elif tag == 'geometry':\n"
           u8"                     self.geometry = tag.getitem(manager.geo, 'ref')\n"
           u8"                 elif tag == 'mesh':\n"
           u8"                     self.mesh = tag.getitem(manager.msh, 'ref')\n"
           u8"                 elif tag == 'boundary':\n"
           u8"                     self.bc.read_from_xpl(tag, manager)\n\n"

           u8"         def on_initialize(self):\n"
           u8"             self.workspace = zeros(1000.)\n\n"

           u8"         def on_invalidate(self):\n"
           u8"             self.workspace = None\n\n"

           u8"         def run_computations(self):\n"
           u8"             pass\n\n"

           u8"To make your solver visible in GUI, you must write the ``solvers.yml`` file\n"
           u8"and put it in the same directory as your data file.\n\n"
           u8"Example:\n"
           u8"  .. code-block:: yaml\n\n"
           u8"     - solver: MySolver\n"
           u8"       lib: mymodule\n"
           u8"       category: local\n"
           u8"       geometry: Cartesian2D\n"
           u8"       mesh: Rectangular2D\n"
           u8"       tags:\n"
           u8"       - tag: config\n"
           u8"         label: Solver Configuration\n"
           u8"         help: Configuration of the effective model of p-n junction.\n"
           u8"         attrs:\n"
           u8"         - attr: param\n"
           u8"           label: Parameter\n"
           u8"           type: float\n"
           u8"           unit: V\n"
           u8"           help: Some voltage parameter.\n"
           u8"       - bcond: boundary\n"
           u8"         label: Something\n", py::no_init);
    solver
        .def("__init__", raw_constructor(&SolverWrap::init))
        .add_property("id", &plask::Solver::getId,
                      u8"Id of the solver object. (read only)\n\n"
                      u8"Example:\n"
                      u8"    >>> mysolver.id\n"
                      u8"    mysolver:category.type")
        .add_property("initialized", &plask::Solver::isInitialized,
                      u8"True if the solver has been initialized. (read only)\n\n"
                      u8"Solvers usually get initialized at the beginning of the computations.\n"
                      u8"You can clean the initialization state and free the memory by calling\n"
                      u8"the :meth:`invalidate` method.")
        .def("initialize", &plask::Solver::initCalculation,
             u8"Initialize solver.\n\n"
             u8"This method manually initialized the solver and sets :attr:`initialized` to\n"
             u8"*True*. Normally calling it is not necessary, as each solver automatically\n"
             u8"initializes itself when needed.\n\n"
             u8"Returns:\n"
             u8"    bool: solver :attr:`initialized` state prior to this method call.")
        .def("invalidate", &plask::Solver::invalidate,
             u8"Set the solver back to uninitialized state.\n\n"
             u8"This method frees the memory allocated by the solver and sets\n"
             u8":attr:`initialized` to *False*.")
        .def("load_xpl", &plask::Solver::loadConfiguration, (py::arg("xpl"), "manager"),
             u8"Load configuration from XPL reader.\n\n"
             u8"This method should be overridden in custom Python solvers.\n\n"
             u8"Example:\n"
             u8"  .. code-block:: python\n\n"
             u8"     def load_xpl(self, xpl, manager):\n"
             u8"         for tag in xpl:\n"
             u8"             if tag == 'config':\n"
             u8"                 self.a = tag['a']\n"
             u8"                 self.b = tag.get('b', 0)\n"
             u8"                 if 'c' in tag:\n"
             u8"                     self.c = tag['c']\n"
             u8"             if tag == 'combined':\n"
             u8"                 for subtag in tag:\n"
             u8"                     if subtag == 'withtext':\n"
             u8"                         self.data = subtag.attrs\n"
             u8"                         # Text must be read last\n"
             u8"                         self.text = subtag.text\n"
             u8"             elif tag == 'geometry':\n"
             u8"                 self.geometry = tag.getitem(manager.geo, 'ref')\n")
    ;
    solver.attr("__module__") = "plask";

    // Exceptions
    register_exception<plask::Exception>(PyExc_RuntimeError);

    py::register_exception_translator<std::string>( [=](const std::string& err) { PyErr_SetString(PyExc_RuntimeError, err.c_str()); } );
    py::register_exception_translator<const char*>( [=](const char* err) { PyErr_SetString(PyExc_RuntimeError, err); } );

    register_exception<plask::NotImplemented>(PyExc_NotImplementedError);
    register_exception<plask::OutOfBoundsException>(PyExc_IndexError);
    register_exception<plask::NoSuchMaterial>(PyExc_ValueError);
    register_exception<plask::MaterialParseException>(PyExc_ValueError);
    register_exception<plask::NoSuchGeometryObjectType>(PyExc_TypeError);
    register_exception<plask::NotUniqueObjectException>(PyExc_ValueError);
    register_exception<plask::NoSuchGeometryObject>(PyExc_ValueError);
    register_exception<plask::BadInput>(PyExc_ValueError);
    register_exception<plask::DataError>(PyExc_ValueError);
    register_exception<plask::NoValue>(PyExc_ValueError);
    register_exception<plask::NoProvider>(PyExc_TypeError);

    register_exception<plask::python::ValueError>(PyExc_ValueError);
    register_exception<plask::python::TypeError>(PyExc_TypeError);
    register_exception<plask::python::IndexError>(PyExc_IndexError);
    register_exception<plask::python::KeyError>(PyExc_KeyError);
    register_exception<plask::python::AttributeError>(PyExc_AttributeError);
    register_exception<plask::python::StopIteration>(PyExc_StopIteration);
    register_exception<plask::python::IOError>(PyExc_IOError);
    #if PY_VERSION_HEX >= 0x03050000
        register_exception<plask::python::RecursionError>(PyExc_RecursionError);
    #else
        register_exception<plask::python::RecursionError>(PyExc_RuntimeError);
    #endif

    pyXmlError = PyErr_NewExceptionWithDoc((char*)"plask.XMLError", (char*)u8"Error in XML file.", NULL, NULL);
    register_exception<plask::XMLException>(pyXmlError);
    py::scope().attr("XMLError") = py::handle<>(py::incref(pyXmlError));

    pyGeometryException = PyErr_NewExceptionWithDoc((char*)"plask.GeometryException", (char*)u8"Error in PLaSK geometry specification.",
                                                            NULL, NULL);
    register_exception<plask::GeometryException>(pyGeometryException);
    py::scope().attr("GeometryException") = py::handle<>(py::incref(pyGeometryException));

    pyComputationError = PyErr_NewExceptionWithDoc((char*)"plask.ComputationError", (char*)u8"Computational error in some PLaSK solver.",
                                                            PyExc_ArithmeticError, NULL);
    register_exception<plask::ComputationError>(pyComputationError);
    py::scope().attr("ComputationError") = py::handle<>(py::incref(pyComputationError));

    py::def("_print_exception", static_cast<int(*)(PyObject*, PyObject*, PyObject*, const char*, const char*, int)>(&printPythonException),
            u8"Print exception information to PLaSK logging system",
            (py::arg("exc_type"), "exc_value", "exc_traceback", py::arg("scriptname")="", py::arg("top_frame")="<module>", py::arg("scriptline")=0));

#   ifdef PRINT_STACKTRACE_ON_EXCEPTION
        py::def("_print_stack", &printStack, "Print C stack (for debug purposes_");
#   endif

    // Converters
    // DoubleFromComplex();

    // PLaSK version
    scope.attr("version") = PLASK_VERSION;

    scope.attr("prefix") = plask::prefixPath();
    scope.attr("lib_path") = plask::plaskLibPath();

    // Properties
    register_standard_properties();

    // Logging
    if (!plask::default_logger) plask::python::createPythonLogger();

    // Common definitions for solvers
    registerFemCommon();
}

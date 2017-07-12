#include <complex>

#include <boost/algorithm/string.hpp>

#include "python_globals.h"
#include "python_numpy.h"
#include <frameobject.h> // for Python traceback

#include <plask/version.h>
#include <plask/exceptions.h>
#include <plask/mesh/interpolation.h>
#include <plask/solver.h>
#include <plask/license/verify.h>

using namespace plask::python;

// Declare some initialization functions
namespace plask { namespace python {

PLASK_PYTHON_API void createPythonLogger();

void initMaterials();
void initGeometry();

void register_manager();
void register_xml_writer();

void register_vectors();
void register_mesh();
void register_providers();

void register_data_vectors();

void register_python_log();

void register_standard_properties();

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
#   if PY_VERSION_HEX >= 0x03000000
        PyObject* result = PyRun_String(string.c_str(), Py_eval_input, global.ptr(), local.ptr());
#   else
        PyCompilerFlags flags { CO_FUTURE_DIVISION };
        PyObject* result = PyRun_StringFlags(string.c_str(), Py_eval_input, global.ptr(), local.ptr(), &flags);
#   endif
    if (!result) py::throw_error_already_set();
    return py::object(py::detail::new_reference(result));
}


// Parallel locking
PLASK_PYTHON_API OmpNestLock python_omp_lock;

// Config
PLASK_PYTHON_API AxisNames current_axes = AxisNames::axisNamesRegister.get("zxy");

static LoggingConfig getLoggingConfig(const Config&) {
    return LoggingConfig();
}

std::string Config::axes_name() const {
    return current_axes.str();
}
void Config::set_axes(std::string axis) {
    current_axes = AxisNames::axisNamesRegister.get(axis);
}

// return list of 3 axes names or throw exception if axes_names isn't fine
py::list axeslist_by_name(const std::string& axes_names) {
    AxisNames a = AxisNames::axisNamesRegister.get(axes_names);
    py::list l;
    l.append(a.byIndex[0]); l.append(a.byIndex[1]); l.append(a.byIndex[2]);
    return l;
}

std::string Config::__str__() const {
    return  "axes:        " + axes_name()
        + "\nlog.colors:  " + std::string(py::extract<std::string>(LoggingConfig().getLoggingColor().attr("__str__")()))
        + "\nlog.level:   " + std::string(py::extract<std::string>(py::object(maxLoglevel).attr("__str__")))
        + "\nlog.output:  " + std::string(py::extract<std::string>(LoggingConfig().getLoggingDest().attr("__str__")()));
    ;
}

std::string Config:: __repr__() const {
    return
        format("config.axes = '{}'", axes_name()) +
           + "\nlog.colors = " + std::string(py::extract<std::string>(LoggingConfig().getLoggingColor().attr("__repr__")()))
           + "\nlog.level = LOG_" + std::string(py::extract<std::string>(py::object(maxLoglevel).attr("__str__")))
           + "\nlog.output = " + std::string(py::extract<std::string>(LoggingConfig().getLoggingDest().attr("__repr__")()));
    ;
}

inline static void register_config()
{
    py::class_<Config> config_class("config",

        "Global PLaSK configuration.\n\n"

        "This class has only one instance and it contains global configuration options.\n"
        "The attributes of this class are config parameters that can be set using the\n"
        "``config`` object.\n\n"

        "Example:\n"
        "    >>> config.axes = 'xy'\n"
        "    >>> config.log.level = 'debug'\n"
        "    >>> print config\n"
        "    axes:        zxy\n"
        "    log.colors:  ansi\n"
        "    log.level:   DEBUG\n"
        "    log.output:  stdout\n"

        , py::no_init);
    config_class
        .def("__str__", &Config::__str__)
        .def("__repr__", &Config::__repr__)
        .add_property("axes", &Config::axes_name, &Config::set_axes,
                      "String representing axis names.\n\n"

                      "The accepted values are listed below. Each row shows different aliases for\n"
                      "the same axes:\n\n"

                      "================ ================ ================\n"
                      "`xyz`            `yz`             `z_up`\n"
                      "`zxy`            `xy`             `y_up`\n"
                      "`prz`            `rz`             `rad`\n"
                      "`ltv`                             `abs`\n"
                      "`long,tran,vert`                  `absolute`\n"
                      "================ ================ ================\n\n"

                      "The choice of the axes should depend on your structure. In Cartesian coordinates\n"
                      "you probably prefer `xyz` or `zxy`. In cylindrical ones the most natural choice\n"
                      "is `prz`. However, it is important to realize that any names can be chosen in\n"
                      "any geometry and they are fully independent from it.\n"
                     )
        .add_property("log", &getLoggingConfig,
                      "Settings of the logging system.\n\n"

                      "This setting has several subattributes listed below:\n\n"

                      "**color**\n"
                      "        System used for coloring the log messages depending on their level.\n"
                      "        This parameter can have on of the following values:\n\n"

                      "        :ansi:    Use ANSI codes for coloring. Works best in UNIX-like systems\n"
                      "                  (Linux, OSX) or with GUI launchers.\n"
                      "        :windows: Use Windows API for coloring. Available only on Windows.\n"
                      "        :none:    Do not perform coloring at all. Recommended when redirecting\n"
                      "                  output to a file.\n\n"

                      "        On its start PLaSK tries to automatically determine the best value for\n"
                      "        this option, so usually you will not need to change it.\n\n"

                      "**level**\n"
                      "        Maximum logging level. It can be one of:\n\n"

                      "        :CRITICAL_ERROR: Critical errors that result in program interruption.\n"
                      "        :ERROR:          Minor errors that do not break the whole program flow.\n"
                      "        :ERROR_DETAIL:   Details of the errors with more information on them.\n"
                      "        :WARNING:        Important warnings that you should investigate.\n"
                      "        :INFO:           General information of the executed operations.\n"
                      "        :RESULT:         Some intermediate computations results.\n"
                      "        :DATA:           Some data used for tracking the computations.\n"
                      "        :DETAIL:         Details of computations processes.\n"
                      "        :DEBUG:          Additional information useful for debugging PLaSK.\n\n"

                      "        Setting any of the above levels will instruct PLaSK to print only\n"
                      "        information of the specified level and above. It is recommended to\n"
                      "        always set the logging level at least to 'WARNING'. This setting is\n"
                      "        ignored when the plask option :option:`-l` is specified.\n\n"

                      "**output**\n"
                      "        Stream to which the log messages are printed. Can be either **stderr**\n"
                      "        (which is the default) or **stdout** (turned on for interactive mode).\n"
                     );
    py::scope().attr("config") = Config();
}


// Globals for XML material
PLASK_PYTHON_API py::dict* xml_globals;

template <typename... Args>
static void printMultiLineLog(plask::LogLevel level, const std::string& msg, Args&&... args) {
    typedef boost::tokenizer<boost::char_separator<char> > LineTokenizer;
    std::string message = plask::format(msg, args...);
    LineTokenizer tokenizer(message, boost::char_separator<char>("\n\r"));
    for (LineTokenizer::const_iterator line = tokenizer.begin(), end = tokenizer.end(); line != end ; ++line)
        plask::writelog(level, *line);
}

// Print Python exception to PLaSK logging system
#if defined(_WIN32) || defined(__WIN32__) || defined(WIN32)
__declspec(dllexport)
#endif
int printPythonException(PyObject* otype, py::object value, PyObject* otraceback, const char* scriptname=nullptr, bool second_is_script=false) {
    PyTypeObject* type = (PyTypeObject*)otype;
    PyTracebackObject* original_traceback = (PyTracebackObject*)otraceback;

    if ((PyObject*)type == PyExc_SystemExit) {
        int exitcode;
        if (PyExceptionInstance_Check(value.ptr())) {
            PyObject* code = PyObject_GetAttrString(value.ptr(), "code");
            if (code) { value = py::object(py::handle<>(code)); }
        }
        if (PyInt_Check(value.ptr()))
            exitcode = (int)PyInt_AsLong(value.ptr());
        else {
            std::cerr.flush();
            std::cout.flush();
            PyObject_Print(value.ptr(), stderr, Py_PRINT_RAW);
            PySys_WriteStderr("\n");
            exitcode = 1;
        }
        PyErr_Clear();
        return exitcode;
    }


    std::string message = py::extract<std::string>(py::str(value));

    std::string error_name = type->tp_name;
    if (error_name.substr(0, 11) == "exceptions.") error_name = error_name.substr(11);

    if (original_traceback) {
        PyTracebackObject* traceback = original_traceback;
        while (traceback) {
            int lineno = traceback->tb_lineno;
            std::string filename = PyString_AsString(traceback->tb_frame->f_code->co_filename);
            std::string funcname = PyString_AsString(traceback->tb_frame->f_code->co_name);
            if (funcname == "<module>" && (traceback == original_traceback || (second_is_script && traceback == original_traceback->tb_next)))
                funcname = "<script>";
            if (traceback->tb_next)
                plask::writelog(plask::LOG_ERROR_DETAIL, "{0}, line {1}, function '{2}' calling:", filename, lineno, funcname);
            else {
                if ((PyObject*)type == PyExc_IndentationError || (PyObject*)type == PyExc_SyntaxError) {
                    plask::writelog(plask::LOG_ERROR_DETAIL, "{0}, line {1}, function '{2}' calling:", filename, lineno, funcname);
                    std::string form = message;
                    std::size_t f = form.find(" (") + 2, l = form.rfind(", line ") + 7;
                    std::string msg = form.substr(0, f-2), file = form.substr(f, l-f-7);
                    try {
                        int lineno = boost::lexical_cast<int>(form.substr(l, form.length()-l-1));
                        printMultiLineLog(plask::LOG_CRITICAL_ERROR, "{0}, line {1}: {2}: {3}", file, lineno, error_name, msg);
                    } catch (boost::bad_lexical_cast) {
                        printMultiLineLog(plask::LOG_CRITICAL_ERROR, "{0}: {1}", error_name, message);
                    }
                } else
                    printMultiLineLog(plask::LOG_CRITICAL_ERROR, "{0}, line {1}, function '{2}': {3}: {4}", filename, lineno, funcname, error_name, message);
            }
            traceback = traceback->tb_next;
        }
    } else {
        if ((PyObject*)type == PyExc_IndentationError || (PyObject*)type == PyExc_SyntaxError) {
                std::string form = message;
                std::size_t f = form.find(" (") + 2, l = form.rfind(", line ") + 7;
                std::string msg = form.substr(0, f-2), file = form.substr(f, l-f-7);
                try {
                    int lineno = boost::lexical_cast<int>(form.substr(l, form.length()-l-1));
                    printMultiLineLog(plask::LOG_CRITICAL_ERROR, "{0}, line {1}: {2}: {3}", file, lineno, error_name, msg);
                } catch (boost::bad_lexical_cast) {
                    printMultiLineLog(plask::LOG_CRITICAL_ERROR, "{0}: {1}", error_name, message);
                }
        } else
            printMultiLineLog(plask::LOG_CRITICAL_ERROR, "{0}: {1}", error_name, message);
    }
    return 1;
}

// Default options for docstrings
py::docstring_options doc_options(
    true,   // show user defined
    true,   // show py signatures
    false   // show cpp signatures
);

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

// struct IntFromInt64 {
//     IntFromInt64() {
//         boost::python::converter::registry::push_back(&convertible, &construct, boost::python::type_id<int>());
//     }
//
//     static void* convertible(PyObject* obj) {
//         return (PyArray_Check(obj) PyArray_IsIntegerScalar(obj) && PyObject_TypeCheck(obj, &PyInt64ArrType_Type))? obj : NULL;
//         // return PyObject_TypeCheck(obj, &PyInt64ArrType_Type)? obj : NULL;
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
//             throw TypeError("Must provide either plask.vector or a sequence of length {0} of proper dtype", dim);
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


BOOST_PYTHON_MODULE(_plask)
{
    // Initialize numpy
    if (!plask_import_array()) throw(py::error_already_set());

    py::scope scope; // Default scope

    // Config
    register_config();

    // Log
    register_python_log();

    // Manager
    register_manager();
    register_xml_writer();

    // Vectors
    register_vectors();

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
    solver("Solver", "Base class for all solvers.", py::no_init);
    solver
        .add_property("id", &plask::Solver::getId,
                      "Id of the solver object. (read only)\n\n"
                      "Example:\n"
                      "    >>> mysolver.id\n"
                      "    mysolver:category.type")
        .add_property("initialized", &plask::Solver::isInitialized,
                      "True if the solver has been initialized. (read only)\n\n"
                      "Solvers usually get initialized at the beginning of the computations.\n"
                      "You can clean the initialization state and free the memory by calling\n"
                      "the :meth:`invalidate` method.")
        .def("initialize", &plask::Solver::initCalculation,
             "Initialize solver.\n\n"
             "This method manually initialized the solver and sets :attr:`initialized` to\n"
             "*True*. Normally calling it is not necessary, as each solver automatically\n"
             "initializes itself when needed.\n\n"
             "Returns:\n"
             "    bool: solver :attr:`initialized` state prior to this method call.")
        .def("invalidate", &plask::Solver::invalidate,
             "Set the solver back to uninitialized state.\n\n"
             "This method frees the memory allocated by the solver and sets\n"
             ":attr:`initialized` to *False*.")
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

    PyObject* xml_error = PyErr_NewExceptionWithDoc((char*)"plask.XMLError", (char*)"Error in XML file.", NULL, NULL);
    register_exception<plask::XMLException>(xml_error);
    py::scope().attr("XMLError") = py::handle<>(py::incref(xml_error));

    PyObject* computation_error = PyErr_NewExceptionWithDoc((char*)"plask.ComputationError", (char*)"Computational error in some PLaSK solver.",
                                                            PyExc_ArithmeticError, NULL);
    register_exception<plask::ComputationError>(computation_error);
    py::scope().attr("ComputationError") = py::handle<>(py::incref(computation_error));

    py::def("_print_exception", &printPythonException, "Print exception information to PLaSK logging system",
            (py::arg("exc_type"), "exc_value", "exc_traceback", py::arg("scriptname")="", py::arg("second_is_script")=false));

#   ifdef PRINT_STACKTRACE_ON_EXCEPTION
        py::def("_print_stack", &printStack, "Print C stack (for debug purposes_");
#   endif

    // Converters
    // DoubleFromComplex();

    // PLaSK version
    scope.attr("version") = PLASK_VERSION;
    scope.attr("version_major") = PLASK_VERSION_MAJOR;
    scope.attr("version_minor") = PLASK_VERSION_MINOR;

    // Set global namespace for materials
    plask::python::xml_globals = new py::dict();
    py::incref(plask::python::xml_globals->ptr()); // HACK: Prevents segfault on exit. I don't know why it is needed.
    scope.attr("__xml__globals") = *plask::python::xml_globals;

    scope.attr("prefix") = plask::prefixPath();
    scope.attr("lib_path") = plask::plaskLibPath();

#ifdef LICENSE_CHECK
    py::dict license;
    license["user"] = plask::license_verifier.getUser();
    license["institution"] = plask::license_verifier.getInstitution();
    license["date"] = plask::license_verifier.getExpiration();
    license["systemid"] = plask::license_verifier.getSystemId();
    scope.attr("license") = license;
#endif

    // Properties
    register_standard_properties();

    // Logging
    if (!plask::default_logger) plask::python::createPythonLogger();
}

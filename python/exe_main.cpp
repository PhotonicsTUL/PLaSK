#include <cmath>
#include <cstdio>
#include <vector>
#include <string>
#include <stack>

#include <plask/version.h>
#include <plask/exceptions.h>
#include <plask/utils/system.h>
#include <plask/log/log.h>
#include <plask/python_globals.h>
#include <plask/python_manager.h>
#include <plask/utils/string.h>
#include <plask/utils/system.h>
#include <plask/config.h>
#include <plask/license/verify.h>

#include <boost/python.hpp>
#include <boost/python/stl_iterator.hpp>
namespace py = boost::python;

#if defined(MS_WINDOWS) || defined(__CYGWIN__)
#  include <io.h>
#  include <fcntl.h>
#endif

//******************************************************************************
#if PY_VERSION_HEX >= 0x03000000
    extern "C" PyObject* PyInit__plask(void);
#   define PLASK_MODULE PyInit__plask
    inline auto PyString_Check(PyObject* o) -> decltype(PyUnicode_Check(o)) { return PyUnicode_Check(o); }
    inline const char* PyString_AsString(PyObject* o) { return py::extract<const char*>(o); }
    inline bool PyInt_Check(PyObject* o) { return PyLong_Check(o); }
    inline long PyInt_AsLong(PyObject* o) { return PyLong_AsLong(o); }
    PyAPI_DATA(int) Py_UnbufferedStdioFlag;
#else
    extern "C" void init_plask(void);
#   define PLASK_MODULE init_plask
#endif

//******************************************************************************
py::object globals;

//******************************************************************************

// static PyThreadState* mainTS;   // state of the main thread

namespace plask { namespace python {

    PLASK_PYTHON_API int printPythonException(PyObject* otype, py::object value, PyObject* otraceback, const char* scriptname=nullptr, bool second_is_script=false);

    PLASK_PYTHON_API std::string getPythonExceptionMessage();

    PLASK_PYTHON_API void PythonManager_load(py::object self, py::object src, py::dict vars, py::object filter=py::object());

    PLASK_PYTHON_API void createPythonLogger();

    PLASK_PYTHON_API void setLoggingColor(std::string color);

    extern PLASK_PYTHON_API AxisNames current_axes;

    extern PLASK_PYTHON_API py::dict xml_globals;
}}

//******************************************************************************
static void from_import_all(const char* name, py::object& dest)
{
    py::object module = py::import(name);
    py::dict module_dict = py::dict(module.attr("__dict__"));
    py::list all;

    try {
        all = py::list(module.attr("__all__"));
    } catch (py::error_already_set) {
        PyErr_Clear();
        all = module_dict.keys();
    }
    py::stl_input_iterator<std::string> begin(all), end;
    for (auto item = begin; item != end; item++) {
        if ((*item)[0] != '_') dest[*item] = module_dict[*item];
    }
}

//******************************************************************************
// Initialize the binary modules and load the package from disk
static py::object initPlask(int argc, const char* argv[])
{
    // Initialize the plask module
    if (PyImport_AppendInittab("_plask", &PLASK_MODULE) != 0) throw plask::CriticalException("No _plask module");

    // Initialize Python
    Py_Initialize();


    // Add search paths
    py::object sys = py::import("sys");
    py::list path = py::list(sys.attr("path"));
    std::string plask_path = plask::prefixPath();
    plask_path += plask::FILE_PATH_SEPARATOR; plask_path += "lib";
    plask_path += plask::FILE_PATH_SEPARATOR; plask_path += "plask";
    std::string solvers_path = plask_path;
    plask_path += plask::FILE_PATH_SEPARATOR; plask_path += "python";
    solvers_path += plask::FILE_PATH_SEPARATOR; solvers_path += "solvers";
    path.insert(0, plask_path);
    path.insert(1, solvers_path);
    if (argc > 0) // This is correct!!! argv[0] here is argv[1] in `main`
        try {
            path.insert(0, boost::filesystem::absolute(boost::filesystem::path(argv[0])).parent_path().string());
        } catch (std::runtime_error) { // can be thrown if there is wrong locale set
            std::string file(argv[0]);
            size_t pos = file.rfind(plask::FILE_PATH_SEPARATOR);
            if (pos == std::string::npos) pos = 0;
            path.insert(0, file.substr(0, pos));
        }
    else
        path.insert(0, "");

    sys.attr("path") = path;

    sys.attr("executable") = plask::exePathAndName();

    py::object _plask = py::import("_plask");

    plask::writelog(plask::LOG_INFO, PLASK_BANNER);
    plask::writelog(plask::LOG_INFO, PLASK_COPYRIGHT);
#ifdef LICENSE_CHECKING
    std::string user = plask::license_verifier.getUser();
    if (user != "") plask::writelog(plask::LOG_INFO, "Licensed to {}.", user);
#endif

    sys.attr("modules")["plask._plask"] = _plask;

    // Add program arguments to sys.argv
    if (argc > 0) {
        py::list sys_argv;
        for (int i = 0; i < argc; i++) {
            sys_argv.append(argv[i]);
        }
        sys.attr("argv") = sys_argv;
    }

    // mainTS = PyEval_SaveThread();
    //PyEval_ReleaseLock();

    PyObject* __main__ = PyImport_AddModule("__main__");
    globals = py::object(py::handle<>(py::borrowed(PyModule_GetDict(__main__))));

    return _plask;
}


//******************************************************************************
// This functions closes all matplotlib windows in order to avoid the annoying
// 'Fatal Python error: PyEval_RestoreThread: NULL tstate' bug on Windows.
static inline void fixMatplotlibBug() {
#if defined(_WIN32) || defined(__WIN32__) || defined(WIN32)
    py::object modules = py::import("sys").attr("modules");
    if (py::dict(modules).has_key("matplotlib")) {
        try {
            py::object pylab = py::import("pylab");
            std::string backend = py::extract<std::string>(pylab.attr("get_backend")());
            if (backend == "TkAgg") pylab.attr("close")("all"); // fix bug in TkAgg backend in Windows
        } catch (py::error_already_set) {
            PyErr_Clear(); // silently ignore errors here
        }
    }
#endif
}


//******************************************************************************
int handlePythonException(const char* scriptname=nullptr) {
    PyObject* value;
    PyObject* type;
    PyObject* original_traceback;

    PyErr_Fetch(&type, &value, &original_traceback);
    PyErr_NormalizeException(&type, &value, &original_traceback);

    py::handle<> value_h(value), type_h(type), original_traceback_h(py::allow_null(original_traceback));
    return plask::python::printPythonException(type, py::object(value_h), original_traceback, scriptname);
}


//******************************************************************************
// Finalize Python interpreter
void endPlask() {
    // PyEval_RestoreThread(mainTS);
    fixMatplotlibBug();
    // Py_Finalize(); // Py_Finalize is not supported by Boost
}


//******************************************************************************
int main(int argc, const char *argv[])
{
    //setlocale(LC_ALL,""); std::locale::global(std::locale(""));    // set default locale from env (C is used when program starts), boost filesystem will do the same

    if (argc > 1 && std::string(argv[1]) == "-V") {
        printf("PLaSK " PLASK_VERSION "\n");
        return 0;
    }

#   if defined(_WIN32) || defined(__WIN32__) || defined(WIN32)
        SetDllDirectory(plask::exePath().c_str());
        DWORD procIDs[2];
        unsigned console_count = GetConsoleProcessList(procIDs, 2);
#   else
        unsigned console_count = 1;
#   endif

    // Parse commnad line
    bool force_interactive = false;
    boost::optional<plask::LogLevel> loglevel;
    const char* command = nullptr;
    const char* log_color = nullptr;
    bool python_logger = false;

    enum {
        FILE_ANY = 0,
        FILE_XML,
        FILE_PY
    } filetype = FILE_ANY;

    std::deque<const char*> defs;

    while (argc > 1) {
        std::string arg = argv[1];
        if (arg == "-i") {
            force_interactive = true;
            --argc; ++argv;
        } else if (arg.substr(0,2) == "-l") {
            const char* level = (arg.length() > 2)? argv[1]+2 : argv[2];
            try {
                loglevel.reset(plask::LogLevel(boost::lexical_cast<unsigned>(level)));
            } catch (boost::bad_lexical_cast) {
                std::string ll = level; boost::to_lower(ll);
                if (ll == "critical_error") loglevel.reset(plask::LOG_CRITICAL_ERROR);
                if (ll == "critical") loglevel.reset(plask::LOG_CRITICAL_ERROR);
                else if (ll == "error") loglevel.reset(plask::LOG_ERROR);
                else if (ll == "error_detail") loglevel.reset(plask::LOG_ERROR_DETAIL);
                else if (ll == "warning") loglevel.reset(plask::LOG_WARNING);
                else if (ll == "info") loglevel.reset(plask::LOG_INFO);
                else if (ll == "result") loglevel.reset(plask::LOG_RESULT);
                else if (ll == "data") loglevel.reset(plask::LOG_DATA);
                else if (ll == "detail") loglevel.reset(plask::LOG_DETAIL);
                else if (ll == "debug") loglevel.reset(plask::LOG_DEBUG);
                else if (ll == "python") { python_logger = true; }
                else if (ll == "ansi") { log_color = "ansi"; }
                else if (ll == "mono") { log_color = "none"; }
                else {
                    fprintf(stderr, "Bad log level specified\n");
                    return 4;
                }
            }
            if (loglevel) plask::forcedLoglevel = true;
            if (level == argv[2]) { argc -= 2; argv += 2; }
            else { --argc; ++argv; }
        } else if (arg == "-c") {
            command = argv[2];
            argv[2] = "-c";
            --argc; ++argv;
            break;
        } else if (arg == "-g") {
            python_logger = true;
#           if defined(_WIN32) || defined(__WIN32__) || defined(WIN32)
                if (console_count == 1) { // we are the only ones using the console
                    HWND hwnd = GetConsoleWindow();
                    ShowWindow(hwnd, SW_HIDE);
                    console_count = 0;
                }
#           endif
            argv[1] = "-u";
        } else if (arg == "-u") {
#           if defined(MS_WINDOWS) || defined(__CYGWIN__)
                _setmode(fileno(stderr), _O_BINARY);
                _setmode(fileno(stdout), _O_BINARY);
#           endif
            setvbuf(stdout, nullptr, _IONBF, 0);
            setvbuf(stderr, nullptr, _IONBF, 0);
            log_color = "none";
#           if PY_VERSION_HEX >= 0x03000000
                Py_UnbufferedStdioFlag = 1;
#           endif
            --argc; ++argv;
        } else if (arg == "-x") {
            if (filetype == FILE_PY)  {
                fprintf(stderr, "You cannot specify both -x and -p\n");
                return 4;
            }
            filetype = FILE_XML;
            --argc; ++argv;
        } else if (arg == "-p") {
            if (filetype == FILE_XML)  {
                fprintf(stderr, "You cannot specify both -x and -p\n");
                return 4;
            }
            filetype = FILE_PY;
            --argc; ++argv;
        } else if (arg.find('=') != std::string::npos) {
            defs.push_back(argv[1]);
            --argc; ++argv;
        } else if (arg == "--") {
            --argc; ++argv;
            break;
        } else break;
    }

    // Set the Python logger
    if (python_logger) plask::python::createPythonLogger();
    else plask::createDefaultLogger();
    if (log_color) plask::python::setLoggingColor(log_color);
    if (loglevel) plask::maxLoglevel = *loglevel;

    // Initalize python and load the plask module
    try {
        initPlask(argc-1, argv+1);
    } catch (plask::CriticalException) {
        plask::writelog(plask::LOG_CRITICAL_ERROR, "Cannot import plask builtin module.");
        endPlask();
        return 101;
    } catch (py::error_already_set) {
        handlePythonException();
        endPlask();
        return 102;
    }

    // Test if we should run command specified in the command line, use the file or start an interactive mode
    if (command) { // run command specified in the command line

        try {
            if (!defs.empty()) {
                PyErr_SetString(PyExc_RuntimeError, "Command-line defines can only be specified when running XPL file");
                throw py::error_already_set();
            }
            py::object plask = py::import("plask");
            globals["plask"] = plask;           // import plask
            from_import_all("plask", globals);  // from plask import *

            PyObject* result = NULL;
#           if PY_VERSION_HEX >= 0x03000000
                PyObject* code = Py_CompileString(command, "-c", Py_file_input);
                if (code) result = PyEval_EvalCode(code, globals.ptr(), globals.ptr());
#           else
                PyCompilerFlags flags { CO_FUTURE_DIVISION };
                PyObject* code = Py_CompileStringFlags(command, "-c", Py_file_input, &flags);
                if (code) result = PyEval_EvalCode((PyCodeObject*)code, globals.ptr(), globals.ptr());
#           endif
            Py_XDECREF(code);
            if (!result) py::throw_error_already_set();
            else Py_DECREF(result);

        } catch (py::error_already_set) {
            int exitcode = handlePythonException();
            endPlask();
            return exitcode;
        } catch (...) {
            endPlask();
            return 0;
        }

    } else if(argc > 1 && !force_interactive && argv[1][0] != 0) { // load commands from file

        // Add plask to the global namespace
        try {
            py::object plask = py::import("plask");
            globals["plask"] = plask;           // import plask
            from_import_all("plask", globals);  // from plask import *
        } catch (py::error_already_set) {
            int exitcode = handlePythonException();
            endPlask();
            return exitcode;
        }

        std::string filename = argv[1];
        try {
            bool realfile = true;
            if (filename[0] == '-' and (filename.length() == 1 or filename[1] == ':')) {
                realfile = false;
                if (filename[1] == ':' and filename.length() > 2) {
                    filename = filename.substr(2);
                } else {
                    filename = "<stdin>";
                }
            }
            globals["__file__"] = filename;

            // Detect if the file is Python script or PLaSK input
            if (realfile) {
                if (!filetype) {
                    // check file extension
                    try {
                        std::string ext = filename.substr(filename.length()-4);
                        if (ext == ".xpl") filetype = FILE_XML;
                        else if (ext == ".xml") filetype = FILE_XML;
                        else if (ext.substr(2) == ".py") filetype = FILE_PY;
                    } catch (std::out_of_range) {}
                }
                if (!filetype) {
                    // check first char (should be '<' in XML)
                    FILE* file = std::fopen(filename.c_str(), "r");
                    if (!file) throw std::invalid_argument("No such file: '" + filename + "'");
                    int c;
                    while ((c = std::getc(file))) {
                        if (!std::isspace(c) || c == EOF) break;
                    }
                    std::fclose(file);
                    if (c == '<') filetype = FILE_XML;
                    else filetype = FILE_PY;
                } else {
                    FILE* file = std::fopen(filename.c_str(), "r");
                    if (!file) throw std::invalid_argument("No such file: '" + filename + "'");
                    std::fclose(file);
                }
                assert(filetype);
            } else {
                if (!filetype) {
                    throw std::invalid_argument("Filetype must by specified (with -x or -p) when reading from <stdin>");
                }
            }

            if (filetype == FILE_XML) {

                py::dict locals;
                for (const char* def: defs) {
                    auto keyval = plask::splitString2(def, '=');
                    try {
                        locals[keyval.first] = (plask::python::py_eval(keyval.second,
                                                                       plask::python::xml_globals, locals));
                    } catch (py::error_already_set) {
                        plask::writelog(plask::LOG_WARNING,
                                        "Cannot parse command-line definition '{}' (storing it as string): {}",
                                        keyval.first, plask::python::getPythonExceptionMessage());
                        PyErr_Clear();
                        locals[keyval.first] = keyval.second;
                    }
                    plask::writelog(plask::LOG_DATA, "{} = {}", keyval.first, keyval.second);
                }

                auto manager = plask::make_shared<plask::python::PythonManager>();
                py::object omanager(manager);
                globals["__manager__"] = omanager;
                if (realfile)
                    plask::python::PythonManager_load(omanager, py::str(filename), locals);
                else {
                    py::object sys = py::import("sys");
                    plask::python::PythonManager_load(omanager, sys.attr("stdin"), locals);
                }
                if (manager->scriptline)
                    manager->script = "#coding: utf8\n" + std::string(manager->scriptline-1, '\n') + manager->script;
                PyDict_Update(globals.ptr(), manager->defs.ptr());
                plask::python::PythonManager::export_dict(omanager, globals);

                // Set default axes if all loaded geometries share the same
                boost::optional<plask::AxisNames> axes;
                for (const auto& geometry: manager->roots) {
                    if (!axes) axes.reset(geometry->axisNames);
                    else if (geometry->axisNames != *axes) {
                        axes.reset();
                        break;
                    }
                }
                if (axes) plask::python::current_axes = *axes;

                PyObject* result = NULL;
#               if PY_VERSION_HEX >= 0x03000000
                    PyObject* code = Py_CompileString(manager->script.c_str(), filename.c_str(), Py_file_input);
                    if (code)
                        result = PyEval_EvalCode(code, globals.ptr(), globals.ptr());
#               else
                    PyCompilerFlags flags { CO_FUTURE_DIVISION };
                    PyObject* code = Py_CompileStringFlags(manager->script.c_str(), filename.c_str(), Py_file_input, &flags);
                    if (code)
                        result = PyEval_EvalCode((PyCodeObject*)code, globals.ptr(), globals.ptr());
#               endif
                Py_XDECREF(code);
                if (!result) py::throw_error_already_set();
                else Py_DECREF(result);

            } else {
                if (!defs.empty()) {
                    PyErr_SetString(PyExc_RuntimeError, "Command-line defines can only be specified when running XPL file");
                    throw py::error_already_set();
                }
                PyObject* pyfile = nullptr;
                PyObject* result;
#               if PY_VERSION_HEX >= 0x03000000
                    if (realfile) {
#                       if PY_VERSION_HEX >= 0x03040000
                            FILE* file = _Py_fopen(filename.c_str(), "r");
#                       else
                            pyfile = PyUnicode_FromString(filename.c_str());
                            FILE* file = _Py_fopen(pyfile, "r");
#                       endif
                        result = PyRun_File(file, filename.c_str(), Py_file_input, globals.ptr(), globals.ptr());
                        fclose(file);
                    } else {
                        result = PyRun_File(stdin, filename.c_str(), Py_file_input, globals.ptr(), globals.ptr());
                    }
#               else
                    // We want to set "from __future__ import division" flag
                    if (realfile) {
                        pyfile = PyFile_FromString(const_cast<char*>(filename.c_str()), const_cast<char*>("r"));
                        if (!pyfile) throw std::invalid_argument("No such file: '" + filename + "'");
                        FILE* file = PyFile_AsFile(pyfile);
                        PyCompilerFlags flags { CO_FUTURE_DIVISION };
                        result = PyRun_FileFlags(file, filename.c_str(), Py_file_input, globals.ptr(), globals.ptr(), &flags);
                    } else {
                        PyCompilerFlags flags { CO_FUTURE_DIVISION };
                        result = PyRun_FileFlags(stdin, filename.c_str(), Py_file_input, globals.ptr(), globals.ptr(), &flags);
                    }
#               endif
                Py_XDECREF(pyfile);
                if (!result) py::throw_error_already_set();
                else Py_DECREF(result);
            }
        }
        // when PRINT_STACKTRACE_ON_EXCEPTION is defined, we will not catch most exceptions
        // in order to handle it by terminate handler and print a call stack
        catch (std::invalid_argument& err) {
            plask::writelog(plask::LOG_CRITICAL_ERROR, err.what());
            endPlask();
            return -1;
        }
#       ifndef PRINT_STACKTRACE_ON_EXCEPTION
            catch (plask::XMLException& err) {
                plask::writelog(plask::LOG_CRITICAL_ERROR, "{0}, {1}", filename, err.what());
                endPlask();
                return 2;
            }
            catch (plask::Exception& err) {
                plask::writelog(plask::LOG_CRITICAL_ERROR, "{0}: {1}", filename, err.what());
                endPlask();
                return 3;
            }
#       endif
        catch (py::error_already_set) {
            int exitcode = handlePythonException(filename.c_str());
            endPlask();
            return exitcode;
        }
#       ifndef PRINT_STACKTRACE_ON_EXCEPTION
            catch (std::runtime_error& err) {
                plask::writelog(plask::LOG_CRITICAL_ERROR, err.what());
                endPlask();
                return 3;
            }
            catch (...) {
                plask::writelog(plask::LOG_CRITICAL_ERROR, "Unrecognized exception");
                endPlask();
                return 3;
            }
#       endif
    } else if (console_count) { // start the interactive console

        if (!defs.empty()) {
            PyErr_SetString(PyExc_RuntimeError, "Command-line defines can only be specified when running XPL file");
            int exitcode = handlePythonException();
            endPlask();
            return exitcode;
        }

        py::object sys = py::import("sys");
        sys.attr("executable") = plask::exePathAndName();

        try
        {
            py::object interactive = py::import("plask.interactive");
            py::list sys_argv;
            if (argc == 1) sys_argv.append("");
            for (int i = 1; i < argc; i++) sys_argv.append(argv[i]);
            interactive.attr("interact")(py::object(), sys_argv);
        }  catch (py::error_already_set) { // This should not happen
            int exitcode = handlePythonException();
            endPlask();
            return exitcode;
        } catch (...) {
            return 0;
        }
    }


    // Close the Python interpreter and exit
    endPlask();
    return 0;
}

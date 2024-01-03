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
#include "exe_common.hpp"  // includes windows.h

#include <plask/config.hpp>
#include "plask/utils/system.hpp"

#if defined(MS_WINDOWS) || defined(__CYGWIN__)
#    include <fcntl.h>
#    include <io.h>
// #  include <windows.h>	// in exe_common.h
#endif

//******************************************************************************
#define PLASK_MODULE PyInit__plask
extern "C" PyObject* PLASK_MODULE(void);

#if PY_VERSION_HEX < 0x030C0000
PyAPI_DATA(int) Py_UnbufferedStdioFlag;
#else
static bool unbuffered_stdio = false;
#endif

//******************************************************************************
py::object* globals;

enum FileType { FILE_ANY = 0, FILE_XML, FILE_PY };
static FileType filetype = FILE_ANY;

//******************************************************************************

// static PyThreadState* mainTS;   // state of the main thread

namespace plask { namespace python {
PLASK_PYTHON_API std::string getPythonExceptionMessage();
PLASK_PYTHON_API void loadXpl(py::object self, py::object src, py::dict vars, py::object filter = py::object());
PLASK_PYTHON_API void createPythonLogger();
PLASK_PYTHON_API void setLoggingColor(std::string color);
PLASK_PYTHON_API void setCurrentAxes(const AxisNames& axes);
PLASK_PYTHON_API void setXplFilename(const std::string& filename);
PLASK_PYTHON_API PyObject* getXmlErrorClass();
}}  // namespace plask::python

//******************************************************************************
static void from_import_all(const py::object& module, py::object& dest) {
    py::dict module_dict = py::dict(module.attr("__dict__"));
    py::list all;

    try {
        all = py::list(module.attr("__all__"));
    } catch (py::error_already_set&) {
        PyErr_Clear();
        all = module_dict.keys();
    }
    py::stl_input_iterator<std::string> begin(all), end;
    for (auto item = begin; item != end; item++) {
        if ((*item)[0] != '_') dest[*item] = module_dict[*item];
    }
}

static void from_import_all(const char* name, py::object& dest) {
    py::object module = py::import(name);
    from_import_all(module, dest);
}

//******************************************************************************
// Initialize the binary modules and load the package from disk
static py::object initPlask(int argc, const system_char* argv[], bool banner) {
    // Initialize the plask module
    if (PyImport_AppendInittab("_plask", &PLASK_MODULE) != 0) throw plask::CriticalException("No _plask module");

        // Initialize Python
#if PY_VERSION_HEX >= 0x03080000
    PyPreConfig preconfig;
    PyPreConfig_InitPythonConfig(&preconfig);
    preconfig.utf8_mode = 1;
    PyStatus status = Py_PreInitialize(&preconfig);
    // if (PyStatus_Exception(status)) {
    //     Py_ExitStatusException(status);
    // }
#elif PY_VERSION_HEX >= 0x03070000
    Py_UTF8Mode = 1;  // use UTF-8 for all strings
#endif

#if PY_VERSION_HEX >= 0x030C0000
    PyConfig config;
    PyConfig_InitPythonConfig(&config);
    config.buffered_stdio = !unbuffered_stdio;
    Py_InitializeFromConfig(&config);
    PyConfig_Clear(&config);
#else
    Py_Initialize();
#endif

    // Add search paths
    py::object sys = py::import("sys");
    py::list path = py::list(sys.attr("path"));
    std::string plask_path = plask::prefixPath();
    plask_path += plask::FILE_PATH_SEPARATOR;
    plask_path += "lib";
    plask_path += plask::FILE_PATH_SEPARATOR;
    plask_path += "plask";
    if (const char* envPath = getenv("PLASK_PYTHON_PATH")) {
        path.insert(0, envPath);
    } else {
        plask_path += plask::FILE_PATH_SEPARATOR;
        plask_path += "python";
        path.insert(0, plask_path);
    }
    if (argc > 0)  // This is correct!!! argv[0] here is argv[1] in `main`
        try {
            boost::filesystem::path argpath = boost::filesystem::absolute(boost::filesystem::path(argv[0]));
            if (!boost::filesystem::is_directory(argpath))
                path.insert(0, system_to_utf8(argpath.parent_path().wstring()));
            else
                path.insert(0, system_to_utf8(argpath.wstring()));
        } catch (std::runtime_error&) {  // can be thrown if there is wrong locale set
            system_string file(argv[0]);
            size_t pos = file.rfind(system_char(plask::FILE_PATH_SEPARATOR));
            if (pos == std::string::npos) pos = 0;
            path.insert(0, file.substr(0, pos));
        }
    else
        path.insert(0, "");

    sys.attr("path") = path;

    sys.attr("executable") = plask::exePathAndName();

    py::object _plask = py::import("_plask");

    if (banner) {
        plask::writelog(plask::LOG_INFO, PLASK_BANNER1);
        plask::writelog(plask::LOG_INFO, PLASK_BANNER2);
        plask::writelog(plask::LOG_INFO, PLASK_BANNER3);
    }

    sys.attr("modules")["plask._plask"] = _plask;

    // Add program arguments to sys.argv
    if (argc > 0) {
        py::list sys_argv;
        for (int i = 0; i < argc; i++) {
            sys_argv.append(system_str_to_pyobject(argv[i]));
        }
        sys.attr("argv") = sys_argv;
    }

    // mainTS = PyEval_SaveThread();
    // PyEval_ReleaseLock();

    PyObject* __main__ = PyImport_AddModule("__main__");
    globals = new py::object(py::handle<>(py::borrowed(PyModule_GetDict(__main__))));

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
            if (backend == "TkAgg") pylab.attr("close")("all");  // fix bug in TkAgg backend in Windows
        } catch (py::error_already_set) {
            PyErr_Clear();  // silently ignore errors here
        }
    }
#endif
}

static inline void finalizeMPI() {
    py::dict modules(py::handle<>(PySys_GetObject("modules")));

    py::object mpi(modules.get("mpi4py.MPI"));
    if (!mpi.is_none()) {
        try {
            bool initialized = py::extract<bool>(mpi.attr("Is_initialized")());
            bool finalized = py::extract<bool>(mpi.attr("Is_finalized")());
            if (initialized && !finalized) {
                mpi.attr("Finalize")();
            }
        } catch (py::error_already_set&) {
            PyErr_Clear();
        }
        return;
    }

    mpi = modules.get("boost.mpi");
    if (!mpi.is_none()) {
        try {
            bool initialized = py::extract<bool>(mpi.attr("initialized")());
            bool finalized = py::extract<bool>(mpi.attr("finalized")());
            if (initialized && !finalized) {
                mpi.attr("finalize")();
            }
        } catch (py::error_already_set&) {
            PyErr_Clear();
        }
        return;
    }
}

//******************************************************************************
int handlePythonException(const char* scriptname = nullptr) {
    PyObject *value, *type, *traceback;
    PyErr_Fetch(&type, &value, &traceback);
    PyErr_NormalizeException(&type, &value, &traceback);
    py::handle<> value_h(value), type_h(type), traceback_h(py::allow_null(traceback));

    if (type == plask::python::getXmlErrorClass() && filetype == FILE_XML && scriptname) {
        PyObject* value_str = PyObject_Str(value);
        std::string message = py::extract<std::string>(value_str);
        Py_DECREF(value_str);
        plask::writelog(plask::LOG_CRITICAL_ERROR, "{}, {}", scriptname, message);
        return 1;
    }

    return plask::python::printPythonException(type, value, traceback, scriptname);
}

//******************************************************************************
// Finalize Python interpreter
void endPlask() {
    // PyEval_RestoreThread(mainTS);
    fixMatplotlibBug();

    // Py_Finalize is not supported by Boost, however we should call atexit hooks
    // Py_Finalize();
    try {
        py::object atexit = py::import("atexit");
        if (PyObject_HasAttrString(atexit.ptr(), "_run_exitfuncs")) atexit.attr("_run_exitfuncs")();
    } catch (py::error_already_set) {
        handlePythonException();
    }
    finalizeMPI();

    // Flush buffers
    try {
        py::object(py::handle<>(PySys_GetObject("stderr"))).attr("flush")();
    } catch (py::error_already_set) {
        PyErr_Clear();
    }
    try {
        py::object(py::handle<>(PySys_GetObject("stdout"))).attr("flush")();
    } catch (py::error_already_set) {
        PyErr_Clear();
    }
}

//******************************************************************************
#if defined(_WIN32) || defined(__WIN32__) || defined(WIN32)
extern "C" int __declspec(dllexport) __stdcall
#else
int
#endif
    system_main(int argc, const system_char* argv[]) {
    // setlocale(LC_ALL,""); std::locale::global(std::locale(""));    // set default locale from env (C is used when program
    // starts), boost filesystem will do the same

    system_string basename = argv[0];
    system_string::size_type last_sep = basename.find_last_of(plask::FILE_PATH_SEPARATOR);
    if (last_sep != std::string::npos) basename = basename.substr(last_sep + 1);

    if (argc > 1) {
        system_string arg(argv[1]);
        if (arg == CSTR(-V) || arg == CSTR(--version)) {
            if (basename.size() >= 6 && basename.substr(0, 6) == CSTR(python)) {
                printf("Python %d.%d.%d\n", PY_MAJOR_VERSION, PY_MINOR_VERSION, PY_MICRO_VERSION);
            } else {
                printf("PLaSK " PLASK_VERSION "\n");
            }
            return 0;
        } else if (arg == CSTR(-h) || arg == CSTR(--help) || arg == CSTR(-?)) {
            printf(
                // "usage: plask [option]... [def=val]... [-i | -c cmd | -m mod | file | -] [args]\n\n"
                "usage: plask [option]... [-i | -c cmd | -m module | file | -] [args]\n\n"

                "Options and arguments:\n"
                "-c cmd         program passed in as string (terminates option list)\n"
                "-D def=val     define 'def' to the value 'val'; this can be used only when\n"
                "               running XPL file (the value defined in the file is ignored)\n"
#if defined(_WIN32) || defined(__WIN32__) || defined(WIN32)
                "-g             run in graphical mode; do not show console window\n"
#endif
                "-h, --help     print this help message and exit\n"
                "-i             force interactive shell\n"
                "-l arg         force logging level (error, error_detail, warning, important,\n"
                "               info, result, data, detail, debug) or force colored (ansi) or\n"
                "               monochromatic (mono) log\n"
                "-m module      run python module as a script (terminates option list)\n"
                "-p             treat provided file as Python script regardless of its\n"
                "               extension (cannot be used together with -x)\n"
                "-s             print hardware system ID for licensing and exit\n"
                "-u             use unbuffered binary stdout and stderr\n"
                "-V, --version  print the PLaSK version number and exit\n"
                "-x             treat provided file as XPL regardless of its\n"
                "               extension (cannot be used together with -p)\n"

                // "\ndef=val        define 'def' to the value 'val'; this can be used only when\n"
                // "               running XPL file (the value defined in the file is ignored)\n"

            );
            return 0;
        }
    }

#if defined(_WIN32) || defined(__WIN32__) || defined(WIN32)
    SetDllDirectory(plask::exePath().c_str());
    DWORD procIDs[2];
    unsigned console_count = GetConsoleProcessList(procIDs, 2);
#else
    unsigned console_count = 1;
#endif

    // Parse command line
    bool force_interactive = false;
    plask::optional<plask::LogLevel> loglevel;
    const system_char* command = nullptr;
    const system_char* runmodule = nullptr;
    const char* log_color = nullptr;
    bool python_logger = true;

    std::deque<std::string> defs;

    while (argc > 1) {
        system_string arg = argv[1];
        if (arg == CSTR(-i)) {
            force_interactive = true;
            --argc;
            ++argv;
        } else if (arg.substr(0, 2) == CSTR(-l)) {
            const system_char* level;
            int drop = 1;
            if (arg.length() > 2)
                level = argv[1] + 2;
            else if (argc > 2) {
                level = argv[2];
                ++drop;
            } else {
                fprintf(stderr, "No log level specified\n");
                return 4;
            }
            try {
                loglevel.reset(plask::LogLevel(boost::lexical_cast<unsigned>(level)));
            } catch (boost::bad_lexical_cast&) {
                system_string ll = level;
                boost::to_lower(ll);
                if (ll == CSTR(critical_error)) loglevel.reset(plask::LOG_CRITICAL_ERROR);
                if (ll == CSTR(critical))
                    loglevel.reset(plask::LOG_CRITICAL_ERROR);
                else if (ll == CSTR(error))
                    loglevel.reset(plask::LOG_ERROR);
                else if (ll == CSTR(error_detail))
                    loglevel.reset(plask::LOG_ERROR_DETAIL);
                else if (ll == CSTR(warning))
                    loglevel.reset(plask::LOG_WARNING);
                else if (ll == CSTR(important))
                    loglevel.reset(plask::LOG_IMPORTANT);
                else if (ll == CSTR(info))
                    loglevel.reset(plask::LOG_INFO);
                else if (ll == CSTR(result))
                    loglevel.reset(plask::LOG_RESULT);
                else if (ll == CSTR(data))
                    loglevel.reset(plask::LOG_DATA);
                else if (ll == CSTR(detail))
                    loglevel.reset(plask::LOG_DETAIL);
                else if (ll == CSTR(debug))
                    loglevel.reset(plask::LOG_DEBUG);
                else if (ll == CSTR(nopython) || ll == CSTR(nopy)) {
                    python_logger = false;
                } else if (ll == CSTR(ansi)) {
                    log_color = "ansi";
                } else if (ll == CSTR(mono)) {
                    log_color = "none";
                } else {
                    fprintf(stderr, "Bad log level specified\n");
                    return 4;
                }
            }
            if (loglevel) plask::forcedLoglevel = true;
            argc -= drop;
            argv += drop;
        } else if (arg.substr(0, 2) == CSTR(-c)) {
            int drop = 0;
            if (arg.length() > 2)
                command = argv[1] + 2;
            else if (argc > 2) {
                command = argv[2];
                ++drop;
            } else {
                fprintf(stderr, "No command specified for the -c option\n");
                return 4;
            }
            argc -= drop;
            argv += drop;
            argv[1] = CSTR(-c);
            break;
        } else if (arg.substr(0, 2) == CSTR(-m)) {
            int drop = 0;
            if (arg.length() > 2)
                runmodule = argv[1] + 2;
            else if (argc > 2) {
                runmodule = argv[2];
                ++drop;
            } else {
                fprintf(stderr, "No module specified for the -m option\n");
                return 4;
            }
            argc -= drop;
            argv += drop;
            argv[1] = CSTR(-m);
            break;
        } else if (arg == CSTR(-g)) {
#if defined(_WIN32) || defined(__WIN32__) || defined(WIN32)
            if (console_count == 1) {  // we are the only ones using the console
                HWND hwnd = GetConsoleWindow();
                ShowWindow(hwnd, SW_HIDE);
                console_count = 0;
            }
#endif
            argv[1] = CSTR(-u);
        } else if (arg == CSTR(-u)) {
#if defined(MS_WINDOWS) || defined(__CYGWIN__)
            _setmode(_fileno(stderr), _O_BINARY);
            _setmode(_fileno(stdout), _O_BINARY);
#endif
            setvbuf(stdout, nullptr, _IONBF, 0);
            setvbuf(stderr, nullptr, _IONBF, 0);
            log_color = "none";
#if PY_VERSION_HEX < 0x030C0000
            Py_UnbufferedStdioFlag = 1;
#else
            unbuffered_stdio = true;
#endif
            --argc;
            ++argv;
        } else if (arg == CSTR(-x)) {
            if (filetype == FILE_PY) {
                fprintf(stderr, "You cannot specify both -x and -p\n");
                return 4;
            }
            filetype = FILE_XML;
            --argc;
            ++argv;
        } else if (arg == CSTR(-p)) {
            if (filetype == FILE_XML) {
                fprintf(stderr, "You cannot specify both -x and -p\n");
                return 4;
            }
            filetype = FILE_PY;
            --argc;
            ++argv;
        } else if (arg.substr(0, 2) == CSTR(-D)) {
            const system_char* def;
            int drop = 1;
            if (arg.length() > 2)
                def = argv[1] + 2;
            else if (argc > 2) {
                def = argv[2];
                ++drop;
            } else {
                fprintf(stderr, "No define specified for the -D option\n");
                return 4;
            }
            defs.push_back(system_to_utf8(def));
            argc -= drop;
            argv += drop;
        } else if (arg.find(system_char('=')) != std::string::npos) {
            defs.push_back(system_to_utf8(argv[1]));
            --argc;
            ++argv;
        } else if (arg == CSTR(--)) {
            --argc;
            ++argv;
            break;
        } else
            break;
    }

    // Set the Python logger
    if (python_logger)
        plask::python::createPythonLogger();
    else
        plask::createDefaultLogger();
    if (log_color) plask::python::setLoggingColor(log_color);
    if (loglevel) plask::maxLoglevel = *loglevel;

    // Check if we are faking python

    bool banner = !std::getenv("PLASK_NOBANNER");
    if (banner) {
        banner = basename.size() < 6 || basename.substr(0, 6) != CSTR(python);
        if (!banner)
#if defined(_WIN32) || defined(__WIN32__) || defined(WIN32)
            _putenv(const_cast<char*>("PLASK_NOBANNER=1"));
#else
            putenv(const_cast<char*>("PLASK_NOBANNER=1"));
#endif
    }

    // Initialize python and load the plask module
    try {
        initPlask(argc - 1, argv + 1, banner);
    } catch (plask::CriticalException&) {
        plask::writelog(plask::LOG_CRITICAL_ERROR, "Cannot import plask builtin module.");
        endPlask();
        return 101;
    } catch (py::error_already_set&) {
        handlePythonException();
        endPlask();
        return 102;
    }

    // Test if we should run command specified in the command line, use the file or start an interactive mode
    if (command) {  // run command specified in the command line

        try {
            if (!defs.empty()) {
                PyErr_SetString(PyExc_RuntimeError, "Command-line defines can only be specified when running XPL file");
                throw py::error_already_set();
            }
            py::object plask = py::import("plask");
            (*globals)["plask"] = plask;       // import plask
            from_import_all(plask, *globals);  // from plask import *

            PyObject* result = NULL;
            PyObject* code = system_Py_CompileString(command, CSTR(<string>), Py_file_input);
            if (code) result = PyEval_EvalCode(code, globals->ptr(), globals->ptr());
            Py_XDECREF(code);
            if (!result)
                py::throw_error_already_set();
            else
                Py_DECREF(result);

        } catch (py::error_already_set&) {
            int exitcode = handlePythonException();
            endPlask();
            return exitcode;
        } catch (...) {
            endPlask();
            return 0;
        }

    } else if (runmodule) {  // run module specified in the command line

        try {
            if (!defs.empty()) {
                PyErr_SetString(PyExc_RuntimeError, "Command-line defines can only be specified when running XPL file");
                throw py::error_already_set();
            }
            py::object plask = py::import("plask");
            (*globals)["plask"] = plask;       // import plask
            from_import_all(plask, *globals);  // from plask import *

            py::object runpy = py::import("runpy");
            py::object runasmain = runpy.attr("_run_module_as_main");
            runasmain(system_string(runmodule), true);
        } catch (py::error_already_set&) {
            int exitcode = handlePythonException();
            endPlask();
            return exitcode;
        } catch (...) {
            endPlask();
            return 0;
        }

    } else if (argc > 1 && !force_interactive && argv[1][0] != 0) {  // load commands from file

        // Add plask to the global namespace
        try {
            py::object plask = py::import("plask");
            (*globals)["plask"] = plask;       // import plask
            from_import_all(plask, *globals);  // from plask import *
        } catch (py::error_already_set&) {
            int exitcode = handlePythonException();
            endPlask();
            return exitcode;
        }

        system_string filename = argv[1];
        try {
            bool realfile = true;
            if (filename[0] == system_char('-') && (filename.length() == 1 || filename[1] == system_char(':'))) {
                realfile = false;
                if (filename[1] == system_char(':') && filename.length() > 2) {
                    filename = filename.substr(2);
                } else {
                    filename = CSTR(<stdin>);
                }
                py::object sys = py::import("sys");
                sys.attr("argv")[0] = filename;
            }

            // Detect if the file is Python script or PLaSK input
            if (realfile) {
                boost::filesystem::path filepath(filename);
                if (!filetype) {
                    // check file extension
                    try {
                        system_string ext = filename.substr(filename.length() - 4);
                        if (ext == CSTR(.xpl))
                            filetype = FILE_XML;
                        else if (ext == CSTR(.xml))
                            filetype = FILE_XML;
                        else if (ext.substr(1) == CSTR(.py))
                            filetype = FILE_PY;
                    } catch (std::out_of_range&) {
                    }
                }
                if (!filetype) {
                    // check first char (should be '<' in XML)
                    if (boost::filesystem::is_directory(filepath)) {
                        filepath /= "__main__.py";
                        filename = path_to_system_string(filepath);
                    }
                    FILE* file = system_fopen(filename.c_str(), CSTR(r));
                    if (!file) throw std::invalid_argument("No such file: '" + system_to_utf8(filename) + "'");
                    int c;
                    while ((c = std::getc(file))) {
                        if (!std::isspace(c) || c == EOF) break;
                    }
                    std::fclose(file);
                    if (c == '<')
                        filetype = FILE_XML;
                    else
                        filetype = FILE_PY;
                } else {
                    if (filetype == FILE_PY && boost::filesystem::is_directory(filepath)) {
                        filepath /= "__main__.py";
                        filename = path_to_system_string(filepath);
                    }
                    FILE* file = system_fopen(filename.c_str(), CSTR(r));
                    if (!file) throw std::invalid_argument("No such file: '" + system_to_utf8(filename) + "'");
                    std::fclose(file);
                }
                assert(filetype);
            } else {
                if (!filetype) {
                    throw std::invalid_argument("Filetype must by specified (with -x or -p) when reading from <stdin>");
                }
            }
            (*globals)["__file__"] = filename;
            plask::python::setXplFilename(system_to_utf8(filename));

            auto manager = plask::make_shared<plask::python::PythonManager>();
            manager->globals["__file__"] = filename;
            py::object omanager(manager);
            manager->makeWeakRef(omanager);

            if (filetype == FILE_XML) {
                py::dict locals;
                for (std::string& def : defs) {
                    auto keyval = plask::splitString2(def, '=');
                    if (keyval.first == "self") throw plask::python::ValueError("Definition name 'self' is reserved");
                    try {
                        locals[keyval.first] = (plask::python::py_eval(keyval.second, manager->globals, locals));
                    } catch (py::error_already_set&) {
                        plask::writelog(plask::LOG_WARNING, "Cannot parse command-line definition '{}' (storing it as string): {}",
                                        keyval.first, plask::python::getPythonExceptionMessage());
                        PyErr_Clear();
                        locals[keyval.first] = keyval.second;
                    }
                    plask::writelog(plask::LOG_IMPORTANT, "{} = {}", keyval.first, keyval.second);
                }

                (*globals)["__manager__"] = omanager;
                // We export some dictionaries that may be useful in XPL parts (like Python geometry)
                if (realfile)
                    plask::python::loadXpl(omanager, system_str_to_pyobject(filename), locals);
                else {
                    py::object sys = py::import("sys");
                    plask::python::loadXpl(omanager, sys.attr("stdin").attr("buffer"), locals);
                }
                if (manager->scriptline)
                    manager->script = "#coding: utf8\n" + std::string(manager->scriptline - 1, '\n') + manager->script;
                PyDict_Update(globals->ptr(), manager->defs.ptr());
                plask::python::PythonManager::export_dict(omanager, *globals);

                // Set default axes if all loaded geometries share the same
                plask::optional<plask::AxisNames> axes;
                for (const auto& geometry : manager->roots) {
                    if (!axes)
                        axes.reset(geometry->axisNames);
                    else if (geometry->axisNames != *axes) {
                        axes.reset();
                        break;
                    }
                }
                if (axes) plask::python::setCurrentAxes(*axes);

                PyObject* result = NULL;
                PyObject* code = system_Py_CompileString(manager->script.c_str(), filename.c_str(), Py_file_input);
                if (code) result = PyEval_EvalCode(code, globals->ptr(), globals->ptr());
                Py_XDECREF(code);
                if (!result)
                    py::throw_error_already_set();
                else
                    Py_DECREF(result);

            } else {
                if (!defs.empty()) {
                    PyErr_SetString(PyExc_RuntimeError, "Command-line defines can only be specified when running XPL file");
                    throw py::error_already_set();
                }
                PyObject* pyfile = nullptr;
                PyObject* result;
                if (realfile) {
                    FILE* file = system_Py_fopen(filename.c_str(), CSTR(r));
                    // TODO conversion to UTF-8 might not be proper here, especially for windows
                    result = PyRun_FileEx(file, system_to_utf8(filename).c_str(), Py_file_input, globals->ptr(), globals->ptr(), 1);
                } else {
                    result = PyRun_File(stdin, system_to_utf8(filename).c_str(), Py_file_input, globals->ptr(), globals->ptr());
                }
                Py_XDECREF(pyfile);
                if (!result)
                    py::throw_error_already_set();
                else
                    Py_DECREF(result);
            }
        }
        // when PRINT_STACKTRACE_ON_EXCEPTION is defined, we will not catch most exceptions
        // in order to handle it by terminate handler and print a call stack
        catch (std::invalid_argument& err) {
            plask::writelog(plask::LOG_CRITICAL_ERROR, err.what());
            endPlask();
            return -1;
        } catch (plask::python::XMLExceptionWithCause& err) {
            err.print(system_to_utf8(filename).c_str());
            endPlask();
            return 2;
        } catch (plask::XMLException& err) {
            plask::writelog(plask::LOG_CRITICAL_ERROR, "{}, {}", system_to_utf8(filename), err.what());
            endPlask();
            return 2;
        }
#ifndef PRINT_STACKTRACE_ON_EXCEPTION
        catch (plask::Exception& err) {
            plask::writelog(plask::LOG_CRITICAL_ERROR, "{}: {}", system_to_utf8(filename), err.what());
            endPlask();
            return 3;
        }
#endif
        catch (py::error_already_set&) {
            int exitcode = handlePythonException(system_to_utf8(filename).c_str());
            endPlask();
            return exitcode;
        }
#ifndef PRINT_STACKTRACE_ON_EXCEPTION
        catch (std::runtime_error& err) {
            plask::writelog(plask::LOG_CRITICAL_ERROR, err.what());
            endPlask();
            return 3;
        } catch (...) {
            plask::writelog(plask::LOG_CRITICAL_ERROR, "Unrecognized exception");
            endPlask();
            return 3;
        }
#endif
    } else if (console_count) {  // start the interactive console

        if (!defs.empty()) {
            PyErr_SetString(PyExc_RuntimeError, "Command-line defines can only be specified when running XPL file");
            int exitcode = handlePythonException();
            endPlask();
            return exitcode;
        }

        py::object sys = py::import("sys");
        sys.attr("executable") = plask::exePathAndName();

        try {
            py::object interactive = py::import("plask.interactive");
            py::list sys_argv;
            if (argc == 1) sys_argv.append("");
            for (int i = 1; i < argc; i++) sys_argv.append(system_str_to_pyobject(argv[i]));
            interactive.attr("interact")(py::object(), sys_argv);
        } catch (py::error_already_set&) {  // This should not happen
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

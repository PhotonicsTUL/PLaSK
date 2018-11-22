#include "exe_common.h"

//******************************************************************************
#if PY_VERSION_HEX >= 0x03000000
    extern "C" PyObject* PyInit__plask(void);
#   define PLASK_MODULE PyInit__plask
    inline auto PyString_Check(PyObject* o) -> decltype(PyUnicode_Check(o)) { return PyUnicode_Check(o); }
    inline const char* PyString_AsString(PyObject* o) { return py::extract<const char*>(o); }
    inline bool PyInt_Check(PyObject* o) { return PyLong_Check(o); }
    inline long PyInt_AsLong(PyObject* o) { return PyLong_AsLong(o); }
#else
    extern "C" void init_plask(void);
#   define PLASK_MODULE init_plask
#endif

//******************************************************************************

// static PyThreadState* mainTS;   // state of the main thread

namespace plask { namespace python {

    int printPythonException(PyObject* otype, py::object value, PyObject* otraceback, const char* scriptname=nullptr, bool second_is_script=false, int scriptline=0);

    void createPythonLogger();
}}

void showError(const std::string& msg, const std::string& cap="Error");

//******************************************************************************
// Initialize the binary modules and load the package from disk
static py::object initPlask(int argc, const system_char* const argv[])
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
    path.insert(0, plask_path);
    std::string solvers_path = plask_path;
    plask_path += plask::FILE_PATH_SEPARATOR; plask_path += "python";
    solvers_path += plask::FILE_PATH_SEPARATOR; solvers_path += "solvers";
    path.insert(0, plask_path);
    path.insert(1, solvers_path);
    if (argc > 0) // This is correct!!! argv[0] here is argv[1] in `main`
        try {
            path.insert(0, boost::filesystem::absolute(boost::filesystem::path(argv[0])).parent_path().string());
        } catch (std::runtime_error&) { // can be thrown if there is wrong locale set
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
    //PyEval_ReleaseLock();

    return _plask;
}

//******************************************************************************
int handlePythonException(/*const char* scriptname=nullptr*/) {
    PyObject* value;
    PyObject* type;
    PyObject* original_traceback;
    PyErr_Fetch(&type, &value, &original_traceback);
    PyErr_NormalizeException(&type, &value, &original_traceback);
    py::handle<> value_h(value), type_h(type), original_traceback_h(py::allow_null(original_traceback));
    if (type == PyExc_SystemExit) {
        int exitcode = 0;
        if (PyInt_Check(value)) exitcode = (int)PyInt_AsLong(value);
        PyErr_Clear();
        return exitcode;
    }
    std::string msg = py::extract<std::string>(py::str(value_h));
    std::string cap = py::extract<std::string>(py::object(type_h).attr("__name__"));
    showError(msg, cap);
    return 1;
}


//******************************************************************************
// Finalize Python interpreter
void endPlask() {
    // PyEval_RestoreThread(mainTS);
    Py_Finalize(); // Py_Finalize is not supported by Boost
}


//******************************************************************************
#if defined(_WIN32) || defined(__WIN32__) || defined(WIN32)

class Splash {
    HWND hSplashWnd;
    HBITMAP hSplashBMP;
    BITMAP bitmap;

  public:
    Splash(HINSTANCE hInst, int resid) {
        hSplashBMP = LoadBitmap(hInst, MAKEINTRESOURCE(resid));
        GetObject(hSplashBMP, sizeof(BITMAP), &bitmap);
        int width = bitmap.bmWidth;
        int height = bitmap.bmHeight;

        RECT desktopRect;
        GetWindowRect(GetDesktopWindow(), &desktopRect);

        int left = (desktopRect.right + desktopRect.left - width) / 2,
            top = (desktopRect.top + desktopRect.bottom - height) / 2;

        hSplashWnd = CreateWindowEx(WS_EX_CLIENTEDGE, "Static", "PLaSK",
            WS_POPUP | WS_DLGFRAME | SS_BITMAP, left, top, width, height, NULL, NULL, hInst, NULL);
        SendMessage(hSplashWnd, STM_SETIMAGE, IMAGE_BITMAP, (LPARAM)hSplashBMP);
    }

    virtual ~Splash() {
        DestroyWindow(hSplashWnd);
        DeleteObject(hSplashBMP);
    }

    void show() {
        // auto windefer = BeginDeferWindowPos(1);
        // DeferWindowPos(windefer, hSplashWnd, HWND_TOP, 0, 0, 50, 50, SWP_NOMOVE | SWP_NOSIZE | SWP_SHOWWINDOW);
        // EndDeferWindowPos(windefer);
        ShowWindow(hSplashWnd, SW_SHOWNORMAL);
        UpdateWindow(hSplashWnd);
    }

    void hide() {
        ShowWindow(hSplashWnd, SW_HIDE);
    }

    static void destroy();
};

Splash* splash;

void Splash::destroy() {
    if (splash) {
        splash->hide();
        delete splash;
        splash = nullptr;
    }
}

#ifndef _MSC_VER
#   include <boost/tokenizer.hpp>
#endif

void showError(const std::string& msg, const std::string& cap) {
    MessageBox(NULL, msg.c_str(), ("PLaSK - " + cap).c_str(), MB_OK | MB_ICONERROR);
}

// MingW need this (should be in windows.h)
//extern "C" __declspec(dllimport) LPWSTR * __stdcall CommandLineToArgvW(LPCWSTR lpCmdLine, int* pNumArgs);

int WinMain(HINSTANCE hInst, HINSTANCE, LPSTR cmdline, int) {

    splash = new Splash(hInst, 201);
    splash->show();

#ifdef _MSC_VER
	int argc;	// doc: https://msdn.microsoft.com/pl-pl/library/windows/desktop/bb776391(v=vs.85).aspx
	system_char** argv = CommandLineToArgvW(GetCommandLineW(), &argc);
	std::unique_ptr<system_char*, decltype(&LocalFree)> callLocalFreeAtExit(argv, &LocalFree);
#else	// MingW:
	std::string command_line(cmdline);
	boost::tokenizer<boost::escaped_list_separator<char>> tokenizer(command_line, boost::escaped_list_separator<char>('^', ' ', '"'));
	std::deque<std::string> args(tokenizer.begin(), tokenizer.end());
	args.push_front(plask::exePathAndName());
	int argc = args.size();
	std::vector<const char*> argv_vec(argc);
	std::vector<const char*>::iterator dst = argv_vec.begin(); for (const auto& src : args) { *(dst++) = src.c_str(); }
	const char** argv = argv_vec.data();
#endif

#else	// non-windows:

void showError(const std::string& msg, const std::string& cap) {
    plask::writelog(plask::LOG_CRITICAL_ERROR, cap + ": " + msg);
}

int system_main(int argc, const system_char *argv[])
{
#endif
    // Set the Python logger
    // plask::python::createPythonLogger();
    plask::createDefaultLogger();

    // Initalize python and load the plask module
    try {
#       if defined(_WIN32) || defined(__WIN32__) || defined(WIN32)
            py::scope _plask = initPlask(argc, argv);
            py::def("_close_splash", &Splash::destroy);
#       else
            initPlask(argc, argv);
#       endif
    } catch (plask::CriticalException&) {
        showError("Cannot import plask builtin module.");
        endPlask();
        return 101;
    } catch (py::error_already_set&) {
        handlePythonException();
        endPlask();
        return 102;
    }

    // Import and run GUI
    try {
        py::object gui = py::import("gui");
        gui.attr("main")();
    } catch (std::invalid_argument& err) {
        showError(err.what(), "Invalid argument");
        endPlask();
        return -1;
    } catch (plask::Exception& err) {
        showError(err.what());
        endPlask();
        return 3;
    } catch (py::error_already_set&) {
        int exitcode = handlePythonException(/*argv[0]*/);
        endPlask();
        return exitcode;
    } catch (std::runtime_error& err) {
        showError(err.what());
        endPlask();
        return 3;
    }

    // Close the Python interpreter and exit
    endPlask();
    return 0;
}

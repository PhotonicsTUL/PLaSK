#include "exe_common.h" // includes minimal windows.h

#if defined(_WIN32) || defined(__WIN32__) || defined(WIN32)
#include <winuser.h>    // SetProcessDpiAwarenessContext
#include <shellapi.h>   // CommandLineToArgvW
#endif

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

    // Workaround Anaconda bug in Windows preventing finding proper Python home
    #if defined(_WIN32) || defined(__WIN32__) || defined(WIN32)
    if (!getenv("PYTHONHOME")) {
        static const WCHAR key[] = L"Software\\Python\\PythonCore\\" BOOST_PP_STRINGIZE(PY_MAJOR_VERSION) L"." BOOST_PP_STRINGIZE(PY_MINOR_VERSION) L"\\InstallPath";
        WCHAR reg_buf[MAX_PATH + 1];
        DWORD buf_size = MAX_PATH + 1;
        LSTATUS result = RegGetValueW(HKEY_CURRENT_USER, key, NULL, RRF_RT_REG_SZ, NULL, reg_buf, &buf_size);
        if (result != ERROR_SUCCESS)
            result = RegGetValueW(HKEY_LOCAL_MACHINE, key, NULL, RRF_RT_REG_SZ, NULL, reg_buf, &buf_size);
        if (result == ERROR_SUCCESS)
            Py_SetPythonHome(reg_buf);
    }
    #endif

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

#   if defined(_MSC_VER) || defined(__MINGW32__)
        sys.attr("executable") = plask::exePath() + "\\plask.exe";
#   else
        sys.attr("executable") = plask::exePath() + "/plask";
#   endif

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
    HWND hWnd;
    HBITMAP hBitmap;
    BITMAP bitmap;

  public:
    Splash(HINSTANCE hInst) {
        HMODULE user32 = LoadLibrary("user32");
        if (user32 != NULL) {
            auto set_dpi_awareness_context = reinterpret_cast<decltype(SetProcessDpiAwarenessContext)*>(GetProcAddress(user32, "SetProcessDpiAwarenessContext"));
            if (set_dpi_awareness_context != NULL) set_dpi_awareness_context(DPI_AWARENESS_CONTEXT_SYSTEM_AWARE);
        }

        HDC screen = GetDC(NULL);
        double scale = static_cast<FLOAT>(GetDeviceCaps(screen, LOGPIXELSX)) / 96.;
        ReleaseDC(NULL, screen);

        RECT desktopRect;
        GetWindowRect(GetDesktopWindow(), &desktopRect);
        int desktop_width = desktopRect.right + desktopRect.left,
            desktop_height = desktopRect.top + desktopRect.bottom;

        int resid = (scale < 1.4)? 201 : (scale < 1.8)? 202 : 203;

        hBitmap = LoadBitmap(hInst, MAKEINTRESOURCE(resid));
        GetObject(hBitmap, sizeof(BITMAP), &bitmap);
        int width = bitmap.bmWidth;
        int height = bitmap.bmHeight;

        int left = (desktop_width - width) / 2,
            top = (desktop_height - height) / 2;

        hWnd = CreateWindowEx(WS_EX_TOOLWINDOW, "Static", "PLaSK",
            WS_POPUP | SS_BITMAP, left, top, width, height, NULL, NULL, hInst, NULL);
        SendMessage(hWnd, STM_SETIMAGE, IMAGE_BITMAP, (LPARAM)hBitmap);
    }

    virtual ~Splash() {
        DestroyWindow(hWnd);
        DeleteObject(hBitmap);
    }

    void show() {
        // auto windefer = BeginDeferWindowPos(1);
        // DeferWindowPos(windefer, hSplashWnd, HWND_TOP, 0, 0, 50, 50, SWP_NOMOVE | SWP_NOSIZE | SWP_SHOWWINDOW);
        // EndDeferWindowPos(windefer);
        ShowWindow(hWnd, SW_SHOWNORMAL);
        UpdateWindow(hWnd);
    }

    void hide() {
        ShowWindow(hWnd, SW_HIDE);
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

    splash = new Splash(hInst);
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

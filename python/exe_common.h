#if defined(_WIN32) || defined(__WIN32__) || defined(WIN32)
#pragma push_macro("NOMINMAX")
#ifndef NOMINMAX
    #define NOMINMAX    //prevents windows.h from defining min, max macros, see http://stackoverflow.com/questions/1904635/warning-c4003-and-errors-c2589-and-c2059-on-x-stdnumeric-limitsintmax
#endif
#include <windows.h>    //we include whole windows.h, as plask/minimal_windows.h is not enought for gui_main
#pragma pop_macro("NOMINMAX")
#endif

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
#include <plask/license/verify.h>

#include <boost/python.hpp>
#include <boost/python/stl_iterator.hpp>
namespace py = boost::python;

// definitions which helps to use wide or narrow string encoding, depending on system/compiler:
#ifdef _MSC_VER		// ------------- Windows - system_string is wstring -------------

static std::string system_to_utf8(const wchar_t* buffer, int len = -1) {
	int nChars = ::WideCharToMultiByte(CP_UTF8, 0, buffer, len, 0, 0, 0, 0);
	if (nChars == 0) return "";
	std::string newbuffer(nChars, '\0');
	::WideCharToMultiByte(CP_UTF8, 0, buffer, len, const_cast<char*>(newbuffer.data()), nChars, 0, 0);
	return newbuffer;
}

inline std::string system_to_utf8(const std::wstring& str) {
	return system_to_utf8(str.data(), (int)str.size());
}

#define system_to_utf8_cstr(s) system_to_utf8(s).c_str()

typedef wchar_t system_char;
typedef std::wstring system_string;
constexpr auto system_fopen = &_wfopen;
constexpr auto system_Py_fopen = &_Py_wfopen;

static PyObject* system_Py_CompileString(const char *str, const system_char *filename, int start) {
	PyObject* fname = PyUnicode_FromWideChar(filename, -1);
	PyObject* result = Py_CompileStringObject(str, fname, start, 0, -1);
	Py_DECREF(fname);
	return result;
}

inline PyObject* system_Py_CompileString(const system_char *str, const system_char *filename, int start) {
	return system_Py_CompileString(system_to_utf8(str).c_str(), filename, start);
}

inline py::object system_str_to_pyobject(const system_char *str, int len = -1) {
	return py::object(py::handle<>(PyUnicode_FromWideChar(str, len)));
}

inline py::object system_str_to_pyobject(const system_string& str) {
	return system_str_to_pyobject(str.data(), int(str.size()));
}

inline system_string path_to_system_string(const boost::filesystem::path& path) {
	return path.wstring();
}


#define system_main wmain
#define CSTR(s) L ## #s

#else	// ------------- non-Windows (we assume that system_string is std::string, and all strings are UTF-8) -------------

#define system_to_utf8(s) s
#define system_to_utf8_cstr(cstr) cstr
#define system_str_to_pyobject py::str

typedef char system_char;
typedef std::string system_string;
constexpr auto system_fopen = &fopen;
#define system_Py_CompileString Py_CompileString
#define system_Py_fopen _Py_fopen
#define system_main main
#define CSTR(s) #s

inline system_string path_to_system_string(const boost::filesystem::path& path) {
	return path.string();
}

#endif

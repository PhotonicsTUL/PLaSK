#include <windows.h>
#include "win_common.hpp"

typedef int (__stdcall* MainProc)(int argc, const wchar_t *argv[]);

int wmain(int argc, const wchar_t *argv[]) {
    setupPythonPath();
    HMODULE plask = LoadLibrary("plaskexe.dll");
    if (!plask) {
        std::cerr << "Could not import 'plaskexe.dll'\n";
        return 127;
    }
    MainProc plask_main = (MainProc)GetProcAddress(plask, "plask_main");
    return plask_main(argc, argv);
}

#include <windows.h>
#include "win_common.hpp"

typedef int (__stdcall* MainProc)(HINSTANCE hInst, LPSTR cmdline);


int WinMain(HINSTANCE hInst, HINSTANCE, LPSTR cmdline, int) {
    setupPythonPath();
    HMODULE plaskgui = LoadLibrary("plaskgui.dll");
    if (!plaskgui) {
        std::cerr << "Could not import 'plaskgui.dll'\n";
        return 127;
    }
    MainProc plaskgui_main = (MainProc)GetProcAddress(plaskgui, "plaskgui_main");
    return plaskgui_main(hInst, cmdline);
}

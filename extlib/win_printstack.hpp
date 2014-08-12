#ifndef WIN_PRINTSTACK_HPP
#define WIN_PRINTSTACK_HPP

#include <windows.h>

#include <dbghelp.h>

#include <stdio.h>
#include <stdlib.h>
#include <io.h>

#ifdef __GNUC__
#include <cxxabi.h> //abi::__cxa_demangle
#endif

#include "PEparser/symbols.hpp"

inline int backtrace(void **buffer, int size)
{
 USHORT frames;

 //HANDLE hProcess;
 if (size <= 0)
   return 0;
 //hProcess = GetCurrentProcess();
 frames = CaptureStackBackTrace(0, (DWORD) size, buffer, NULL);

 return (int) frames;
}

inline char **backtrace_symbols(void *const *buffer, int size)
{
 size_t t = 0;
 char **r, *cur;
 int i;

 t = ((size_t) size) * ((sizeof(void *) * 2) + 6 + sizeof(char *));
 r = (char**) malloc(t);
 if (!r)
  return r;

 cur = (char*) &r[size];

 for(i = 0; i < size; i++)
   {
     r[i] = cur;
     sprintf(cur, "[+0x%zx]", (size_t) buffer[i]);
     cur += strlen(cur) + 1;
   }

 return r;
}

inline void backtrace_symbols_fd(void *const *buffer, int size, int fd)
{
 char s[128];
 int i;

 for (i = 0; i < size; i++)
   {
     sprintf(s, "[+0x%zx]\n", (size_t) buffer[i]);
     write(fd, s, strlen(s));
   }
 _commit(fd);
}


inline void printStack(void)
{
 unsigned int i;
 void *stack[60];
 unsigned short frames;
 SYMBOL_INFO *symbol;
 HANDLE hProcess;

 hProcess = GetCurrentProcess();
 //SymSetOptions( SYMOPT_DEFERRED_LOADS | SYMOPT_INCLUDE_32BIT_MODULES | SYMOPT_UNDNAME );
 SymSetOptions(SymGetOptions() & ~SYMOPT_UNDNAME);	//allow for names demangle
 if (!SymInitialize(hProcess, NULL, TRUE)) return;
 frames = CaptureStackBackTrace( 0, 60, stack, NULL );
 char buffer[ sizeof(SYMBOL_INFO) + 256 * sizeof(char) ] = { 0 };
 symbol = (SYMBOL_INFO *) buffer;
 symbol->MaxNameLen = 255;
 symbol->SizeOfStruct = sizeof(SYMBOL_INFO);
 for(i = 0; i < frames; i++) {
   if (SymFromAddr(hProcess, (DWORD64) (stack[i]), 0, symbol)) {
       //TODO http://msdn.microsoft.com/en-us/library/windows/desktop/ms680578%28v=vs.85%29.aspx
    #ifdef __GNUC__
       int demangl_status;  //0 for success
       const char *realname = abi::__cxa_demangle(symbol->Name, 0, 0, &demangl_status);
       printf("%u: %p %s = 0x%zx\n", frames - i - 1, stack[i], demangl_status == 0 ? realname : symbol->Name, symbol->Address);
       free((void*)realname);
    #else
       printf("%u: %p %s = 0x%zx\n", frames - i - 1, stack[i], symbol->Name, symbol->Address);
    #endif
    } else {
       const char* fun_name; const char* module_name;
       if (mingw_lookup(stack[i], fun_name, module_name)) {
            printf("%u: %p %s %s\n", frames - i - 1, stack[i], fun_name, module_name);
       } else
        printf("%u: %p UNKNOWN\n", frames - i - 1, stack[i]);
   }
 }
 SymCleanup(hProcess);
}

#endif // WIN_PRINTSTACK_HPP

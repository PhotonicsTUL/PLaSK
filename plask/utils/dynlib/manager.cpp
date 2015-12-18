#include "manager.h"

#ifdef _MSC_VER
#include <algorithm>
#endif

namespace plask {

const DynamicLibrary& DynamicLibraries::load(const std::string &file_name, unsigned flags) {
//#ifdef _MSC_VER
    /*DynamicLibrary lib(file_name, flags);
    auto iter = std::find(loaded.begin(), loaded.end(), );

    for (DynamicLibrary& l: loaded) {
        if (l.)
    }*/
//#else
    return *loaded.emplace(file_name, flags).first;
//#endif
}

void DynamicLibraries::close(const DynamicLibrary &to_close) {
    loaded.erase(to_close);
}

void DynamicLibraries::closeAll() {
    loaded.clear();
}

DynamicLibraries &DynamicLibraries::defaultSet()
{
    static DynamicLibraries default_set;
    return default_set;
}



}   // namespace plask

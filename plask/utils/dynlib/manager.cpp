#include "manager.h"

namespace plask {

const DynamicLibrary& DynamicLibraries::load(const std::string &file_name) {
    return *loaded.insert(DynamicLibrary(file_name)).first;
}

void DynamicLibraries::close(const DynamicLibrary &to_close) {
    loaded.erase(to_close);
}

DynamicLibraries &DynamicLibraries::defaultSet()
{
    static DynamicLibraries default_set;
    return default_set;
}



}   // namespace plask

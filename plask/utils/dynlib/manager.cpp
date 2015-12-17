#include "manager.h"

namespace plask {

const DynamicLibrary& DynamicLibraries::load(const std::string &file_name, unsigned flags) {
    return *loaded.emplace(file_name, flags).first;
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

#include "factory.h"

namespace plask {

FiltersFactory &FiltersFactory::getDefault() {
    static FiltersFactory defaultDb;
    return defaultDb;
}

shared_ptr<Solver> FiltersFactory::get(XMLReader &reader, Manager& manager) {
    if (reader.getTagName() != "filter")
        return shared_ptr<Solver>();
    std::string typeName = reader.requireAttribute("type");
    auto it = filterCreators.find(typeName);
    if (it == filterCreators.end())
        throw Exception("No filter for such type: %1%", typeName);
    return it->second(reader, manager);
}

void FiltersFactory::add(const std::string typeName, FiltersFactory::FilterCreator filterCreator) {
    filterCreators[typeName] = filterCreator;
}


}   // namespace plask

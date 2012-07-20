#include "mesh.h"
#include "../utils/xml.h"

namespace plask {

std::map<std::string, RegisterMeshReader::ReadingFunction*>& RegisterMeshReader::getReaders() {
    static std::map<std::string, RegisterMeshReader::ReadingFunction*> result;
    return result;
}

RegisterMeshReader::RegisterMeshReader(const std::string& tag_name, RegisterMeshReader::ReadingFunction* reader) {
    getReaders()[tag_name] = reader;
}

RegisterMeshReader::ReadingFunction* RegisterMeshReader::getReader(const std::string& name) {
    auto reader = getReaders().find(name);
    if (reader == getReaders().end()) throw Exception("No registered reader for mesh of type '%1%'", name);
    return reader->second;
}


std::map<std::string, RegisterMeshGeneratorReader::ReadingFunction*>& RegisterMeshGeneratorReader::getReaders() {
    static std::map<std::string, RegisterMeshGeneratorReader::ReadingFunction*> result;
    return result;
}

RegisterMeshGeneratorReader::RegisterMeshGeneratorReader(const std::string& tag_name, RegisterMeshGeneratorReader::ReadingFunction* reader) {
    getReaders()[tag_name] = reader;
}

RegisterMeshGeneratorReader::ReadingFunction* RegisterMeshGeneratorReader::getReader(const std::string& name) {
    auto reader = getReaders().find(name);
    if (reader == getReaders().end()) throw Exception("No registered reader for mesh generator of type '%1%'", name);
    return reader->second;
}

}   // namespace plask

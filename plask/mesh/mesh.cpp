#include "mesh.h"

namespace plask {

std::map<std::string, RegisterMeshReader::ReadingFunction>& RegisterMeshReader::getReaders() {
    static std::map<std::string, RegisterMeshReader::ReadingFunction> result;
    return result;
}

RegisterMeshReader::RegisterMeshReader(const std::string& tag_name, ReadingFunction fun) {
    getReaders()[tag_name] = fun;
}

RegisterMeshReader::ReadingFunction RegisterMeshReader::getReader(const std::string& name) {
    auto reader = getReaders().find(name);
    if (reader == getReaders().end()) throw Exception("No registered reader for mesh of type '%1%'", name);
    return reader->second;
}


std::map<std::string, RegisterMeshGeneratorReader::ReadingFunction>& RegisterMeshGeneratorReader::getReaders() {
    static std::map<std::string, RegisterMeshGeneratorReader::ReadingFunction> result;
    return result;
}

RegisterMeshGeneratorReader::RegisterMeshGeneratorReader(const std::string& tag_name, RegisterMeshGeneratorReader::ReadingFunction fun) {
    getReaders()[tag_name] = fun;
}

RegisterMeshGeneratorReader::ReadingFunction RegisterMeshGeneratorReader::getReader(const std::string& name) {
    auto reader = getReaders().find(name);
    if (reader == getReaders().end()) throw Exception("No registered reader for mesh generator of type '%1%'", name);
    return reader->second;
}

template <int dimension>
void MeshD<dimension>::print(std::ostream& out) const {
    print_seq(out << '[', begin(), end(), ", ") << ']';
}

template struct MeshD<1>;
template struct MeshD<2>;
template struct MeshD<3>;

template class MeshGeneratorD<1>;
template class MeshGeneratorD<2>;
template class MeshGeneratorD<3>;

}   // namespace plask

#include "mesh.h"
#include "../utils/xml.h"

std::map<std::string, RegisterMeshReader::ReadingFunction*>& RegisterMeshReader::readers() {
    static std::map<std::string, RegisterMeshReader::ReadingFunction*> result;
    return result;
}

RegisterMeshReader::RegisterMeshReader(const std::string& tag_name, RegisterMeshReader::ReadingFunction* reader) {
    readers()[tag_name] = reader;
}

std::map<std::string, RegisterMeshGeneratorReader::ReadingFunction*>& RegisterMeshGeneratorReader::readers() {
    static std::map<std::string, RegisterMeshGeneratorReader::ReadingFunction*> result;
    return result;
}

RegisterMeshGeneratorReader::RegisterMeshGeneratorReader(const std::string& tag_name, RegisterMeshGeneratorReader::ReadingFunction* reader) {
    readers()[tag_name] = reader;
}

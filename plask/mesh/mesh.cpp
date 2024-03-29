/*
 * This file is part of PLaSK (https://plask.app) by Photonics Group at TUL
 * Copyright (c) 2022 Lodz University of Technology
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 */
#include "mesh.hpp"

namespace plask {

void Mesh::writeXML(XMLElement &) const {
    throw NotImplemented("mesh::writeXML()");
}

void Mesh::onChange(const Mesh::Event &) {}

std::map<std::string, RegisterMeshReader::ReadingFunction>& RegisterMeshReader::getReaders() {
    static std::map<std::string, RegisterMeshReader::ReadingFunction> result;
    return result;
}

RegisterMeshReader::RegisterMeshReader(const std::string& tag_name, ReadingFunction fun) {
    getReaders()[tag_name] = fun;
}

RegisterMeshReader::ReadingFunction RegisterMeshReader::getReader(const std::string& name) {
    auto reader = getReaders().find(name);
    if (reader == getReaders().end()) throw Exception("no registered reader for mesh of type '{0}'", name);
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
    if (reader == getReaders().end()) throw Exception("no registered reader for mesh generator of type '{0}'", name);
    return reader->second;
}

template <int dimension>
void MeshD<dimension>::print(std::ostream& out) const {
    print_seq(out << '[', begin(), end(), ", ") << ']';
}

template<int dimension>
bool MeshD<dimension>::hasSameNodes(const MeshD<dimension> &to_compare) const {
    const std::size_t s = this->size();
    if (s != to_compare.size()) return false;
    for (std::size_t i = 0; i < s; ++i) if (this->at(i) != to_compare.at(i)) return false;
    return true;
}

void MeshGenerator::onChange(const MeshGenerator::Event &) {}

template <int MESH_DIM>
shared_ptr<typename MeshGeneratorD<MESH_DIM>::MeshType> MeshGeneratorD<MESH_DIM>::operator()(const typename MeshGeneratorD<MESH_DIM>::GeometryPtr &geometry) {
    if (auto res = cache.get(geometry))
        return res;
    else
        return cache(geometry, generate(geometry));
}

template struct PLASK_API MeshD<1>;
template struct PLASK_API MeshD<2>;
template struct PLASK_API MeshD<3>;

template class PLASK_API MeshGeneratorD<1>;
template class PLASK_API MeshGeneratorD<2>;
template class PLASK_API MeshGeneratorD<3>;


}   // namespace plask

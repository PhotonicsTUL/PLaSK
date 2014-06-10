#include "utils.h"

namespace plask {

void plask::SameMeshChecker::setMesh(const plask::Mesh *mesh) {
    connection_with_mesh.disconnect();
    this->mesh = mesh;
    if (this->mesh)
        connection_with_mesh = const_cast<Mesh*>(this->mesh)->changed.connect(boost::bind(&SameMeshChecker::onMeshChanged, this, _1), boost::signals2::at_front);
}

}   // namespace plask

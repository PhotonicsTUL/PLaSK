#ifndef PLASK__MESH_EXTRUDED_TRIANGULAR3D_H
#define PLASK__MESH_EXTRUDED_TRIANGULAR3D_H

#include "axis1d.h"
#include "triangular2d.h"

namespace plask {

struct ExtrudedTriangularMesh3D: public MeshD<3> {

    TriangularMesh2D longTranMesh;

    const shared_ptr<MeshAxis> vertAxis;

    /// Iteration order, if true vert axis is changed the fastest, else it is changed the slowest.
    bool vertFastest;

    struct Element {
        const ExtrudedTriangularMesh3D& mesh;
        std::size_t longTranIndex, vertIndex;

        Element(const ExtrudedTriangularMesh3D& mesh, std::size_t longTranIndex, std::size_t vertIndex)
            : mesh(mesh), longTranIndex(longTranIndex), vertIndex(vertIndex) {}

        Element(const ExtrudedTriangularMesh3D& mesh, std::size_t elementIndex);

        /// @return index of this element
        std::size_t getIndex() const { return mesh.elementIndex(longTranIndex, vertIndex); }

        /// @return position of the middle of the element
        inline Vec<3, double> getMidpoint() const;

        /**
         * Get area of this element.
         * @return the area of the element
         */
        double getArea() const;

        /**
         * Check if point @p p is included in @c this element.
         * @param p point to check
         * @return @c true only if @p p is included in @c this
         */
        bool includes(Vec<3, double> p) const {
            return mesh.vertAxis->at(vertIndex) <= p.vert() && p.vert() <= mesh.vertAxis->at(vertIndex+1)
                    && mesh.longTranMesh.element(longTranIndex).includes(vec(p.lon(), p.tran()));
        }

        /**
         * Calculate minimal box which contains this element.
         * @return calculated box
         */
        Box3D getBoundingBox() const;

    private:
        TriangularMesh2D::Element longTranElement() const { return mesh.longTranMesh.element(longTranIndex); }
    };

    Vec<3, double> at(std::size_t index) const override;

    std::size_t size() const override;

    bool empty() const override;

    void writeXML(XMLElement& object) const override;

    /**
     * Calculate index of this mesh using indexes of embeded meshes.
     * @param longTranIndex index of longTranMesh
     * @param vertIndex index of vertAxis
     * @return the index of this mesh
     */
    std::size_t index(std::size_t longTranIndex, std::size_t vertIndex) const {
        return vertFastest ?
            longTranIndex * vertAxis->size() + vertIndex :
            vertIndex * longTranMesh.size() + longTranIndex;
    }

    /**
     * Calculate element index of this mesh using element indexes of embeded meshes.
     * @param longTranIndex index of longTranMesh element
     * @param vertIndex index of vertAxis element
     * @return the element index of this mesh
     */
    std::size_t elementIndex(std::size_t longTranElementIndex, std::size_t vertElementIndex) const {
        return vertFastest ?
            longTranElementIndex * (vertAxis->size()-1) + vertElementIndex :
            vertElementIndex * longTranMesh.getElementsCount() + longTranElementIndex;
    }

    /**
     * Get number of elements in this mesh.
     * @return number of elements
     */
    std::size_t getElementsCount() const {
        const std::size_t vertSize = vertAxis->size();
        return vertSize == 0 ? 0 : (vertSize-1) * longTranMesh.getElementsCount();
    }
};

}   // namespace plask

#endif // PLASK__MESH_EXTRUDED_TRIANGULAR3D_H

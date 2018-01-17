#include <boost/test/unit_test.hpp>

#include <plask/mesh/mesh.h>

BOOST_AUTO_TEST_SUITE(mesh) // MUST be the same as the file name

BOOST_AUTO_TEST_CASE(Mesh) {

    struct OnePoint3DMesh: public plask::MeshD<3> {

        //Held point:
        plask::Vec<3, double> point;

        OnePoint3DMesh(const plask::Vec<3, double>& point)
        : point(point) {}

        //plask::MeshD<plask::space::Cartesian3D> methods implementation:

        virtual std::size_t size() const override {
            return 1;
        }

        plask::Vec<3, double> at(std::size_t index) const override {
            BOOST_CHECK_EQUAL(index, 0);
            return point;
        }

    };

    OnePoint3DMesh mesh(plask::vec(1.0, 2.0, 3.0));
    OnePoint3DMesh::Iterator it = mesh.begin();
    BOOST_CHECK(it != mesh.end());
    BOOST_CHECK_EQUAL(*it, plask::vec(1.0, 2.0, 3.0));
    ++it;
    BOOST_CHECK(it == mesh.end());
}

/*BOOST_AUTO_TEST_CASE(SimpleMeshAdapter) {
    // Create 3d mesh which uses the std::vector of 3d points as internal representation:
    plask::SimpleMeshAdapter< std::vector< plask::Vec<3, double> >, 3 > mesh;
    mesh.internal.push_back(plask::vec(1.0, 1.2, 3.0));
    mesh->push_back(plask::vec(3.0, 4.0, 0.0));
    BOOST_CHECK_EQUAL(mesh.size(), 2);
}*/

BOOST_AUTO_TEST_SUITE_END()

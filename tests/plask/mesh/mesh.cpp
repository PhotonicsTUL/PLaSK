#include <boost/test/unit_test.hpp>

#include <plask/mesh/mesh.h>

BOOST_AUTO_TEST_SUITE(mesh) // MUST be the same as the file name

BOOST_AUTO_TEST_CASE(Mesh) {

    struct OnePoint3DMesh: public plask::Mesh<3> {

        //Held point:
        plask::Vec<3, double> point;

        OnePoint3DMesh(const plask::Vec<3, double>& point)
        : point(point) {}

        //Iterator:
        struct IteratorImpl: public Mesh<3>::IteratorImpl {

            //point to mesh or is equal to nullptr for end iterator
            const OnePoint3DMesh* mesh_ptr;

            //mesh == nullptr for end iterator
            IteratorImpl(const OnePoint3DMesh* mesh)
            : mesh_ptr(mesh) {}

            virtual const plask::Vec<3, double> dereference() const {
                return mesh_ptr->point;
            }

            virtual void increment() {
                mesh_ptr = nullptr; //we iterate only over one point, so next state is end
            }

            virtual bool equal(const typename Mesh<3>::IteratorImpl& other) const {
                return mesh_ptr == static_cast<const IteratorImpl&>(other).mesh_ptr;
            }

            virtual IteratorImpl* clone() const {
                return new IteratorImpl(mesh_ptr);
            }

            std::size_t getIndex() const {
                return 0;
            }

        };

        //plask::Mesh<plask::space::Cartesian3D> methods implementation:

        virtual std::size_t size() const {
            return 1;
        }

        virtual typename Mesh<3>::Iterator begin() const {
            return Mesh<3>::Iterator(new IteratorImpl(this));
        }

        virtual typename Mesh<3>::Iterator end() const {
            return Mesh<3>::Iterator(new IteratorImpl(nullptr));
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

#include <plask/log/log.h>
#include <plask/manager.h>

#include "generator_rectilinear.h"


namespace plask {

shared_ptr<RectilinearMesh2D> makeGeometryGrid(const shared_ptr<GeometryObjectD<2>>& geometry)
{
    auto mesh = make_shared<RectilinearMesh2D>();

    std::vector<Box2D> boxes = geometry->getLeafsBoundingBoxes();

    for (auto& box: boxes) {
        mesh->axis0.addPoint(box.lower.c0);
        mesh->axis0.addPoint(box.upper.c0);
        mesh->axis1.addPoint(box.lower.c1);
        mesh->axis1.addPoint(box.upper.c1);
    }

    mesh->setOptimalIterationOrder();

    return mesh;
}

shared_ptr<RectilinearMesh2D> RectilinearMesh2DSimpleGenerator::generate(const shared_ptr<GeometryObjectD<2>>& geometry)
{
    auto mesh = makeGeometryGrid(geometry);
    if (extend_to_zero) mesh->axis0.addPoint(0.);
    writelog(LOG_DETAIL, "mesh.Rectilinear2D::SimpleGenerator: Generating new mesh (%1%x%2%)", mesh->axis0.size(), mesh->axis1.size());
    return mesh;
}

shared_ptr<RectilinearMesh3D> makeGeometryGrid(const shared_ptr<GeometryObjectD<3>>& geometry)
{
    auto mesh = make_shared<RectilinearMesh3D>();

    std::vector<Box3D> boxes = geometry->getLeafsBoundingBoxes();

    for (auto& box: boxes) {
        mesh->axis0.addPoint(box.lower.c0);
        mesh->axis0.addPoint(box.upper.c0);
        mesh->axis1.addPoint(box.lower.c1);
        mesh->axis1.addPoint(box.upper.c1);
        mesh->axis2.addPoint(box.lower.c2);
        mesh->axis2.addPoint(box.upper.c2);
    }

    mesh->setOptimalIterationOrder();

    return mesh;
}

shared_ptr<RectilinearMesh3D> RectilinearMesh3DSimpleGenerator::generate(const shared_ptr<GeometryObjectD<3>>& geometry)
{
    auto mesh = makeGeometryGrid(geometry);
    writelog(LOG_DETAIL, "mesh.Rectilinear3D::SimpleGenerator: Generating new mesh (%1%x%2%x%3%)", mesh->axis0.size(), mesh->axis1.size(), mesh->axis2.size());
    return mesh;
}


template <int dim>
RectilinearMesh1D RectilinearMeshDivideGenerator<dim>::get1DMesh(const RectilinearMesh1D& initial, const shared_ptr<GeometryObjectD<dim>>& geometry, size_t dir)
{
    if (pre_divisions[dir] == 0) pre_divisions[dir] = 1;
    if (post_divisions[dir] == 0) post_divisions[dir] = 1;

    RectilinearMesh1D result = initial;

    // First add refinement points
    for (auto ref: refinements[dir]) {
        auto object = ref.first.first.lock();
        if (!object) {
             if (warn_missing) writelog(LOG_WARNING, "RectilinearMeshDivideGenerator: Refinement defined for object not existing any more.");
        } else {
            auto path = ref.first.second;
            auto boxes = geometry->getObjectBoundingBoxes(*object, path);
            auto origins = geometry->getObjectPositions(*object, path);
            if (warn_missing && boxes.size() == 0) writelog(LOG_WARNING, "RectilinearMeshDivideGenerator: Refinement defined for object absent from the geometry.");
            else if (warn_multiple && boxes.size() > 1) writelog(LOG_WARNING, "RectilinearMeshDivideGenerator: Single refinement defined for more than one object.");
            auto box = boxes.begin();
            auto origin = origins.begin();
            for (; box != boxes.end(); ++box, ++origin) {
                for (auto x: ref.second) {
                    double zero = (*origin)[dir];
                    double lower = box->lower[dir] - zero;
                    double upper = box->upper[dir] - zero;
                    if (warn_outside && (x < lower || x > upper))
                        writelog(LOG_WARNING, "RectilinearMeshDivideGenerator: Refinement at %1% outside of the object (%2% to %3%).",
                                            x, lower, upper);
                    result.addPoint(zero + x);
                }
            }
        }
    }

    // Next divide each object
    double x = *result.begin();
    std::vector<double> points; points.reserve((pre_divisions[dir]-1)*(result.size()-1));
    for (auto i = result.begin()+1; i!= result.end(); ++i) {
        double w = *i - x;
        for (size_t j = 1; j != pre_divisions[dir]; ++j) points.push_back(x + w*j/pre_divisions[dir]);
        x = *i;
    }
    result.addOrderedPoints(points.begin(), points.end());

    // Now ensure, that the grids do not change to quickly
    if (result.size() > 2 && gradual) {
        size_t end = result.size()-2;
        double w_prev = INFINITY, w = result[1]-result[0], w_next = result[2]-result[1];
        for (size_t i = 0; i <= end;) {
            bool goon = true;
            if (w > 2.001*w_prev) { // .0001 is for border case w == 2*w_prev, to avoid division even in presence of numerical error
                if (result.addPoint(0.5 * (result[i] + result[i+1]))) {
                    ++end;
                    w = w_next = result[i+1] - result[i];
                    goon = false;
                }
            } else if (w > 2.001*w_next) {
                if (result.addPoint(0.5 * (result[i] + result[i+1]))) {
                    ++end;
                    w_next = result[i+1] - result[i];
                    if (i) {
                        --i;
                        w = w_prev;
                        w_prev = (i == 0)? INFINITY : result[i] - result[i-1];
                    } else
                        w = w_next;
                    goon = false;
                }
            }
            if (goon) {
                ++i;
                w_prev = w;
                w = w_next;
                w_next = (i == end)? INFINITY : result[i+2] - result[i+1];
            }
        }
    }

    // Finally divide each object in post- division
    x = *result.begin();
    points.clear(); points.reserve((post_divisions[dir]-1)*(result.size()-1));
    for (auto i = result.begin()+1; i!= result.end(); ++i) {
        double w = *i - x;
        for (size_t j = 1; j != post_divisions[dir]; ++j) points.push_back(x + w*j/post_divisions[dir]);
        x = *i;
    }

    result.addOrderedPoints(points.begin(), points.end());

    return result;
}

template <> shared_ptr<RectilinearMesh2D>
RectilinearMeshDivideGenerator<2>::generate(const boost::shared_ptr<plask::GeometryObjectD<2>>& geometry)
{
    RectangularMesh<2,RectilinearMesh1D> initial;
    std::vector<Box2D> boxes = geometry->getLeafsBoundingBoxes();
    for (auto& box: boxes) {
        initial.axis0.addPoint(box.lower.c0);
        initial.axis0.addPoint(box.upper.c0);
        initial.axis1.addPoint(box.lower.c1);
        initial.axis1.addPoint(box.upper.c1);
    }

    auto mesh = make_shared<RectangularMesh<2,RectilinearMesh1D>>(get1DMesh(initial.axis0, geometry, 0), get1DMesh(initial.axis1, geometry, 1));

    mesh->setOptimalIterationOrder();

    writelog(LOG_DETAIL, "mesh.Rectilinear2D::DivideGenerator: Generating new mesh (%1%x%2%)",
             mesh->axis0.size(), mesh->axis1.size());
    return mesh;
}

template <> shared_ptr<RectilinearMesh3D>
RectilinearMeshDivideGenerator<3>::generate(const boost::shared_ptr<plask::GeometryObjectD<3>>& geometry)
{
    RectangularMesh<3,RectilinearMesh1D> initial;
    std::vector<Box3D> boxes = geometry->getLeafsBoundingBoxes();
    for (auto& box: boxes) {
        initial.axis0.addPoint(box.lower.c0);
        initial.axis0.addPoint(box.upper.c0);
        initial.axis1.addPoint(box.lower.c1);
        initial.axis1.addPoint(box.upper.c1);
        initial.axis2.addPoint(box.lower.c2);
        initial.axis2.addPoint(box.upper.c2);
    }

    auto mesh = make_shared<RectangularMesh<3,RectilinearMesh1D>>(
        get1DMesh(initial.axis0, geometry, 0),
        get1DMesh(initial.axis1, geometry, 1),
        get1DMesh(initial.axis2, geometry, 2)
    );

    mesh->setOptimalIterationOrder();

    writelog(LOG_DETAIL, "mesh.Rectilinear3D::DivideGenerator: Generating new mesh (%1%x%2%x%3%)",
             mesh->axis0.size(), mesh->axis1.size(), mesh->axis2.size());
    return mesh;
}




template <typename GeneratorT>
static shared_ptr<MeshGenerator> readTrivialGenerator(XMLReader& reader, const Manager&)
{
    reader.requireTagEnd();
    return make_shared<GeneratorT>();
}

template <int dim>
static shared_ptr<MeshGenerator> readRectilinearMeshDivideGenerator(XMLReader& reader, const Manager& manager)
{
    auto result = make_shared<RectilinearMeshDivideGenerator<dim>>();

    std::set<std::string> read;
    while (reader.requireTagOrEnd()) {
        if (read.find(reader.getNodeName()) != read.end())
            throw XMLDuplicatedElementException(std::string("<generator>"), reader.getNodeName());
        read.insert(reader.getNodeName());
        if (reader.getNodeName() == "prediv") {
            boost::optional<size_t> into = reader.getAttribute<size_t>("by");
            if (into) {
                if (reader.hasAttribute("by0")) throw XMLConflictingAttributesException(reader, "by", "by0");
                if (reader.hasAttribute("by1")) throw XMLConflictingAttributesException(reader, "by", "by1");
                if (reader.hasAttribute("by2")) throw XMLConflictingAttributesException(reader, "by", "by2");
                for (int i = 0; i < dim; ++i) result->pre_divisions[i] = *into;
            } else
                for (int i = 0; i < dim; ++i) result->pre_divisions[i] = reader.getAttribute<size_t>(format("by%1%", i), 1);
            reader.requireTagEnd();
        } else if (reader.getNodeName() == "postdiv") {
            boost::optional<size_t> into = reader.getAttribute<size_t>("by");
            if (into) {
                if (reader.hasAttribute("by0")) throw XMLConflictingAttributesException(reader, "by", "by0");
                if (reader.hasAttribute("by1")) throw XMLConflictingAttributesException(reader, "by", "by1");
                if (reader.hasAttribute("by2")) throw XMLConflictingAttributesException(reader, "by", "by2");
                for (int i = 0; i < dim; ++i) result->post_divisions[i] = *into;
            } else
                for (int i = 0; i < dim; ++i) result->post_divisions[i] = reader.getAttribute<size_t>(format("by%1%", i), 1);
            reader.requireTagEnd();
        } else if (reader.getNodeName() == "no-gradual") {
            result->setGradual(false);
            reader.requireTagEnd();
        } else if (reader.getNodeName() == "warnings") {
            result->warn_missing = reader.getAttribute<bool>("missing", true);
            result->warn_multiple = reader.getAttribute<bool>("multiple", true);
            result->warn_outside = reader.getAttribute<bool>("outside", true);
            reader.requireTagEnd();
        } else if (reader.getNodeName() == "refinements") {
            while (reader.requireTagOrEnd()) {
                if (reader.getNodeName() != "axis0" && reader.getNodeName() != "axis1" && (dim == 2 || reader.getNodeName() != "axis2")) {
                    if (dim == 2) throw XMLUnexpectedElementException(reader, "<axis0> or <axis1>");
                    if (dim == 3) throw XMLUnexpectedElementException(reader, "<axis0>, <axis1>, or <axis2>");
                }
                auto direction = (reader.getNodeName() == "axis0")? typename Primitive<dim>::Direction(0) :
                                 (reader.getNodeName() == "axis1")? typename Primitive<dim>::Direction(1) :
                                                                    typename Primitive<dim>::Direction(2);
                weak_ptr<GeometryObjectD<dim>> object
                    = manager.requireGeometryObject<GeometryObjectD<dim>>(reader.requireAttribute("object"));
                PathHints path; if (auto pathattr = reader.getAttribute("path")) path = manager.requirePathHints(*pathattr);
                if (auto by = reader.getAttribute<unsigned>("by")) {
                    double objsize = object.lock()->getBoundingBox().size()[unsigned(direction)];
                    for (unsigned i = 1; i < *by; ++i) {
                        double pos = objsize * i / *by;
                        result->addRefinement(direction, object, path, pos);
                    }
                } else if (auto every = reader.getAttribute<double>("every")) {
                    double objsize = object.lock()->getBoundingBox().size()[unsigned(direction)];
                    for (double pos = *every; pos < objsize; pos += *every)
                        result->addRefinement(direction, object, path, pos);
                } else if (auto pos = reader.getAttribute<double>("at")) {
                    result->addRefinement(direction, object, path, *pos);
                } else
                    throw XMLNoAttrException(reader, "at', 'every', or 'by");
                reader.requireTagEnd();
            }
        } else throw XMLUnexpectedElementException(reader, "proper 'divide' generator configuration tag");
    }
    return result;
}

static RegisterMeshGeneratorReader rectilinearmesh2d_simplegenerator_reader("rectilinear2d.simple", readTrivialGenerator<RectilinearMesh2DSimpleGenerator>);
static RegisterMeshGeneratorReader rectilinearmesh3d_simplegenerator_reader("rectilinear3d.simple", readTrivialGenerator<RectilinearMesh3DSimpleGenerator>);

static RegisterMeshGeneratorReader rectilinearmesh2d_dividinggenerator_reader("rectilinear2d.divide", readRectilinearMeshDivideGenerator<2>);
static RegisterMeshGeneratorReader rectilinearmesh3d_dividinggenerator_reader("rectilinear3d.divide", readRectilinearMeshDivideGenerator<3>);


} // namespace plask

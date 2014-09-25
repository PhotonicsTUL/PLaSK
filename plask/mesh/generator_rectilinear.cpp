#include <plask/log/log.h>
#include <plask/manager.h>

#include "generator_rectilinear.h"

namespace plask {

inline static void addPoints(OrderedAxis& dst, double lo, double up, bool singleMaterial, double min_ply, unsigned max_points) {
    dst.addPoint(lo);
    dst.addPoint(up);
    if (!singleMaterial) {
        const double ply = abs(up - lo);
        const unsigned points = (min_ply != 0.)? std::min(unsigned(std::ceil(ply / abs(min_ply))), max_points) : max_points;
        for (long i = points - 1; i > 0; --i) {
            dst.addPoint(lo + i * ply / points, 0.5*ply/points);
        }
    }
}

shared_ptr<OrderedAxis> makeGeometryGrid1D(const shared_ptr<GeometryObjectD<2>>& geometry, bool extend_to_zero)
{
    auto mesh = make_shared<OrderedAxis>();

    std::vector<Box2D> boxes = geometry->getLeafsBoundingBoxes();
    std::vector< shared_ptr<const GeometryObject> > leafs = geometry->getLeafs();

    for (std::size_t i = 0; i < boxes.size(); ++i)
        if (boxes[i].isValid())
            addPoints(*mesh, boxes[i].lower.c0, boxes[i].upper.c0, leafs[i]->isUniform(Primitive<3>::DIRECTION_TRAN), leafs[i]->min_ply, leafs[i]->max_points);

    if (extend_to_zero) mesh->addPoint(0.);

    return mesh;
}

shared_ptr<MeshD<1> > OrderedMesh1DSimpleGenerator::generate(const shared_ptr<GeometryObjectD<2>>& geometry)
{
    auto mesh = makeGeometryGrid1D(geometry, extend_to_zero);
    writelog(LOG_DETAIL, "mesh.Rectilinear1D::SimpleGenerator: Generating new mesh (%1%)", mesh->size());
    return mesh;
}


shared_ptr<RectangularMesh<2>> makeGeometryGrid(const shared_ptr<GeometryObjectD<2>>& geometry, bool extend_to_zero)
{
    shared_ptr<OrderedAxis> axis0(new OrderedAxis), axis1(new OrderedAxis);

    std::vector<Box2D> boxes = geometry->getLeafsBoundingBoxes();
    std::vector< shared_ptr<const GeometryObject> > leafs = geometry->getLeafs();

    for (std::size_t i = 0; i < boxes.size(); ++i)
        if (boxes[i].isValid()) {
            addPoints(*axis0, boxes[i].lower.c0, boxes[i].upper.c0, leafs[i]->isUniform(Primitive<3>::DIRECTION_TRAN), leafs[i]->min_ply, leafs[i]->max_points);
            addPoints(*axis1, boxes[i].lower.c1, boxes[i].upper.c1, leafs[i]->isUniform(Primitive<3>::DIRECTION_VERT), leafs[i]->min_ply, leafs[i]->max_points);
        }

    if (extend_to_zero) axis0->addPoint(0.);

    shared_ptr<RectangularMesh<2>> mesh = make_shared<RectangularMesh<2>>(std::move(axis0), std::move(axis1));
    mesh->setOptimalIterationOrder();
    return mesh;
}

shared_ptr<MeshD<2> > RectilinearMesh2DSimpleGenerator::generate(const shared_ptr<GeometryObjectD<2>>& geometry)
{
    shared_ptr<RectangularMesh<2>> mesh = makeGeometryGrid(geometry, extend_to_zero);
    writelog(LOG_DETAIL, "mesh.Rectangular2D::SimpleGenerator: Generating new mesh (%1%x%2%)", mesh->axis0->size(), mesh->axis1->size());
    return mesh;
}


shared_ptr<MeshD<2> > RectilinearMesh2DFrom1DGenerator::generate(const shared_ptr<GeometryObjectD<2>>& geometry)
{
    return make_shared<RectangularMesh<2>>(horizontal_generator->get<RectangularMesh<1>>(geometry), makeGeometryGrid(geometry)->axis1);
}


shared_ptr<RectangularMesh<3>> makeGeometryGrid(const shared_ptr<GeometryObjectD<3>>& geometry)
{
    shared_ptr<OrderedAxis> axis0(new OrderedAxis), axis1(new OrderedAxis), axis2(new OrderedAxis);

    std::vector<Box3D> boxes = geometry->getLeafsBoundingBoxes();
    std::vector< shared_ptr<const GeometryObject> > leafs = geometry->getLeafs();

    for (std::size_t i = 0; i < boxes.size(); ++i)
        if (boxes[i].isValid()) {
            addPoints(*axis0, boxes[i].lower.c0, boxes[i].upper.c0, leafs[i]->isUniform(Primitive<3>::DIRECTION_LONG), leafs[i]->min_ply, leafs[i]->max_points);
            addPoints(*axis1, boxes[i].lower.c1, boxes[i].upper.c1, leafs[i]->isUniform(Primitive<3>::DIRECTION_TRAN), leafs[i]->min_ply, leafs[i]->max_points);
            addPoints(*axis2, boxes[i].lower.c2, boxes[i].upper.c2, leafs[i]->isUniform(Primitive<3>::DIRECTION_VERT), leafs[i]->min_ply, leafs[i]->max_points);
        }

    shared_ptr<RectangularMesh<3>> mesh = make_shared<RectangularMesh<3>>(std::move(axis0), std::move(axis1), std::move(axis2));
    mesh->setOptimalIterationOrder();
    return mesh;
}

shared_ptr<MeshD<3> > RectilinearMesh3DSimpleGenerator::generate(const shared_ptr<GeometryObjectD<3>>& geometry)
{
    auto mesh = makeGeometryGrid(geometry);
    writelog(LOG_DETAIL, "mesh.Rectangular3D::SimpleGenerator: Generating new mesh (%1%x%2%x%3%)", mesh->axis0->size(), mesh->axis1->size(), mesh->axis2->size());
    return mesh;
}


template <int dim>
shared_ptr<OrderedAxis> RectilinearMeshDivideGenerator<dim>::getAxis(shared_ptr<OrderedAxis> initial_and_result, const shared_ptr<GeometryObjectD<DIM>>& geometry, size_t dir)
{
    assert(bool(initial_and_result));

    if (pre_divisions[dir] == 0) pre_divisions[dir] = 1;
    if (post_divisions[dir] == 0) post_divisions[dir] = 1;

    OrderedAxis& result = *initial_and_result.get();

    // First add refinement points
    for (auto ref: refinements[dir]) {
        auto object = ref.first.first.lock();
        if (!object) {
             if (warn_missing) writelog(LOG_WARNING, "RectilinearMeshDivideGenerator: Refinement defined for object not existing any more");
        } else {
            auto path = ref.first.second;
            auto boxes = geometry->getObjectBoundingBoxes(*object, path);
            auto origins = geometry->getObjectPositions(*object, path);
            if (warn_missing && boxes.size() == 0) writelog(LOG_WARNING, "RectilinearMeshDivideGenerator: Refinement defined for object absent from the geometry");
            else if (warn_multiple && boxes.size() > 1) writelog(LOG_WARNING, "RectilinearMeshDivideGenerator: Single refinement defined for more than one object");
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

    return initial_and_result;
}

template <> shared_ptr<MeshD<1>>
RectilinearMeshDivideGenerator<1>::generate(const boost::shared_ptr<plask::GeometryObjectD<2>>& geometry)
{
    shared_ptr<OrderedAxis> mesh = makeGeometryGrid1D(geometry);
    getAxis(mesh, geometry, 0);
    writelog(LOG_DETAIL, "mesh.Rectilinear1D::DivideGenerator: Generating new mesh (%1%)", mesh->size());
    return mesh;
}

template <> shared_ptr<MeshD<2>>
RectilinearMeshDivideGenerator<2>::generate(const boost::shared_ptr<plask::GeometryObjectD<2>>& geometry)
{
    auto mesh = makeGeometryGrid(geometry);
    getAxis(dynamic_pointer_cast<OrderedAxis>(mesh->axis0), geometry, 0);
    getAxis(dynamic_pointer_cast<OrderedAxis>(mesh->axis1), geometry, 1);
    mesh->setOptimalIterationOrder();
    writelog(LOG_DETAIL, "mesh.Rectangular2D::DivideGenerator: Generating new mesh (%1%x%2%)", mesh->axis0->size(), mesh->axis1->size());
    return mesh;
}

template <> shared_ptr<MeshD<3>>
RectilinearMeshDivideGenerator<3>::generate(const boost::shared_ptr<plask::GeometryObjectD<3>>& geometry)
{
    auto mesh = makeGeometryGrid(geometry);
    getAxis(dynamic_pointer_cast<OrderedAxis>(mesh->axis0), geometry, 0);
    getAxis(dynamic_pointer_cast<OrderedAxis>(mesh->axis1), geometry, 1);
    getAxis(dynamic_pointer_cast<OrderedAxis>(mesh->axis2), geometry, 2);
    mesh->setOptimalIterationOrder();
    writelog(LOG_DETAIL, "mesh.Rectangular3D::DivideGenerator: Generating new mesh (%1%x%2%x%3%)",
                          mesh->axis0->size(), mesh->axis1->size(), mesh->axis2->size());
    return mesh;
}




template <typename GeneratorT>
static shared_ptr<MeshGenerator> readTrivialGenerator(XMLReader& reader, const Manager&)
{
    reader.requireTagEnd();
    return make_shared<GeneratorT>();
}

template <int dim>
static shared_ptr<MeshGenerator> readRectilinearDivideGenerator(XMLReader& reader, const Manager& manager)
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
        } else if (reader.getNodeName() == "gradual") {
            result->setGradual(reader.requireAttribute<bool>("all"));
            reader.requireTagEnd();
        } else if (reader.getNodeName() == "no-gradual") {
            writelog(LOG_WARNING, "Tag '<no-gradual>' is obsolete, type '<gradual all=\"false\"/>' instead");
            result->setGradual(false);
            reader.requireTagEnd();
        } else if (reader.getNodeName() == "warnings") {
            result->warn_missing = reader.getAttribute<bool>("missing", true);
            result->warn_multiple = reader.getAttribute<bool>("multiple", true);
            result->warn_outside = reader.getAttribute<bool>("outside", true);
            reader.requireTagEnd();
        } else if (reader.getNodeName() == "refinements") {
            while (reader.requireTagOrEnd()) {
                if (reader.getNodeName() != "axis0" && (dim == 1 || (reader.getNodeName() != "axis1" && (dim == 2 || reader.getNodeName() != "axis2")))) {
                    if (dim == 1) throw XMLUnexpectedElementException(reader, "<axis0>");
                    if (dim == 2) throw XMLUnexpectedElementException(reader, "<axis0> or <axis1>");
                    if (dim == 3) throw XMLUnexpectedElementException(reader, "<axis0>, <axis1>, or <axis2>");
                }
                auto direction = (reader.getNodeName() == "axis0")? typename Primitive<RectilinearMeshDivideGenerator<dim>::DIM>::Direction(0) :
                                 (reader.getNodeName() == "axis1")? typename Primitive<RectilinearMeshDivideGenerator<dim>::DIM>::Direction(1) :
                                                                    typename Primitive<RectilinearMeshDivideGenerator<dim>::DIM>::Direction(2);
                weak_ptr<GeometryObjectD<RectilinearMeshDivideGenerator<dim>::DIM>> object
                    = manager.requireGeometryObject<GeometryObjectD<RectilinearMeshDivideGenerator<dim>::DIM>>(reader.requireAttribute("object"));
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


static RegisterMeshGeneratorReader rectilinear_simplegenerator_reader  ("ordered.simple",   readTrivialGenerator<OrderedMesh1DSimpleGenerator>);
static RegisterMeshGeneratorReader rectangular2d_simplegenerator_reader("rectangular2d.simple", readTrivialGenerator<RectilinearMesh2DSimpleGenerator>);
static RegisterMeshGeneratorReader rectangular3d_simplegenerator_reader("rectangular3d.simple", readTrivialGenerator<RectilinearMesh3DSimpleGenerator>);

static RegisterMeshGeneratorReader rectilinear_dividinggenerator_reader  ("ordered.divide",   readRectilinearDivideGenerator<1>);
static RegisterMeshGeneratorReader rectangular2d_dividinggenerator_reader("rectangular2d.divide", readRectilinearDivideGenerator<2>);
static RegisterMeshGeneratorReader rectangular3d_dividinggenerator_reader("rectangular3d.divide", readRectilinearDivideGenerator<3>);


// OBSOLETE

template <int dim>
static shared_ptr<MeshGenerator> readRectilinearDivideGenerator_obsolete(XMLReader& reader, const Manager& manager)
{
    if (reader.requireAttribute("type") == "rectilinear1d")
        writelog(LOG_WARNING, "Type 'rectilinear1d' is obsolete, use 'ordered' instead");
    else if (reader.requireAttribute("type") == "rectilinear2d")
        writelog(LOG_WARNING, "Type 'rectilinear2d' is obsolete, use 'rectangular2d' instead");
    else if (reader.requireAttribute("type") == "rectilinear3d")
        writelog(LOG_WARNING, "Type 'rectilinear3d' is obsolete, use 'rectangular3d' instead");
    return readRectilinearDivideGenerator<dim>(reader, manager);
}

template <typename GeneratorT>
static shared_ptr<MeshGenerator> readTrivialGenerator_obsolete(XMLReader& reader, const Manager&)
{
    if (reader.requireAttribute("type") == "rectilinear1d")
        writelog(LOG_WARNING, "Type 'rectilinear1d' is obsolete, use 'ordered' instead");
    else if (reader.requireAttribute("type") == "rectilinear2d")
        writelog(LOG_WARNING, "Type 'rectilinear2d' is obsolete, use 'rectangular2d' instead");
    else if (reader.requireAttribute("type") == "rectilinear3d")
        writelog(LOG_WARNING, "Type 'rectilinear3d' is obsolete, use 'rectangular3d' instead");
    reader.requireTagEnd();
    return make_shared<GeneratorT>();
}

static RegisterMeshGeneratorReader rectilinearmesh1d_simplegenerator_reader("rectilinear1d.simple", readTrivialGenerator_obsolete<OrderedMesh1DSimpleGenerator>);
static RegisterMeshGeneratorReader rectilinearmesh2d_simplegenerator_reader("rectilinear2d.simple", readTrivialGenerator_obsolete<RectilinearMesh2DSimpleGenerator>);
static RegisterMeshGeneratorReader rectilinearmesh3d_simplegenerator_reader("rectilinear3d.simple", readTrivialGenerator_obsolete<RectilinearMesh3DSimpleGenerator>);

static RegisterMeshGeneratorReader rectilinearmesh1d_dividinggenerator_reader("rectilinear1d.divide", readRectilinearDivideGenerator_obsolete<1>);
static RegisterMeshGeneratorReader rectilinearmesh2d_dividinggenerator_reader("rectilinear2d.divide", readRectilinearDivideGenerator_obsolete<2>);
static RegisterMeshGeneratorReader rectilinearmesh3d_dividinggenerator_reader("rectilinear3d.divide", readRectilinearDivideGenerator_obsolete<3>);


} // namespace plask

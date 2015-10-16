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
        for (long i = long(points) - 1; i > 0; --i) {
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
std::pair<double, double> RectilinearMeshRefinedGenerator<dim>::getMinMax(const shared_ptr<OrderedAxis> &axis)
{
    double min = INFINITY, max = 0;
    for (size_t i = 1; i != axis->size(); ++i) {
        double L = axis->at(i) - axis->at(i-1);
        if (L < min) min = L;
        if (L > max) max = L;
    }
    return std::pair<double, double>(min, max);
}

template <int dim>
void RectilinearMeshRefinedGenerator<dim>::divideLargestSegment(shared_ptr<OrderedAxis> axis)
{
    double max = 0;
    double newpoint;
    for (size_t i = 1; i != axis->size(); ++i) {
        double L = axis->at(i) - axis->at(i-1);
        if (L > max) { max = L; newpoint = 0.5 * (axis->at(i-1) + axis->at(i)); }
    }
    axis->addPoint(newpoint);
}

template <int dim>
shared_ptr<OrderedAxis> RectilinearMeshRefinedGenerator<dim>::getAxis(shared_ptr<OrderedAxis> axis, const shared_ptr<GeometryObjectD<DIM>>& geometry, size_t dir)
{
    assert(bool(axis));

    // Add refinement points
    for (auto ref: this->refinements[dir]) {
        auto object = ref.first.first.lock();
        if (!object) {
             if (this->warn_missing) writelog(LOG_WARNING, "%s: Refinement defined for object not existing any more", name());
        } else {
            auto path = ref.first.second;
            auto boxes = geometry->getObjectBoundingBoxes(*object, path);
            auto origins = geometry->getObjectPositions(*object, path);
            if (this->warn_missing && boxes.size() == 0) writelog(LOG_WARNING, "DivideGenerator: Refinement defined for object absent from the geometry");
            else if (this->warn_multiple && boxes.size() > 1) writelog(LOG_WARNING, "DivideGenerator: Single refinement defined for more than one object");
            auto box = boxes.begin();
            auto origin = origins.begin();
            for (; box != boxes.end(); ++box, ++origin) {
                for (auto x: ref.second) {
                    double zero = (*origin)[dir];
                    double lower = box->lower[dir] - zero;
                    double upper = box->upper[dir] - zero;
                    if (this->warn_outside && (x < lower || x > upper))
                        writelog(LOG_WARNING, "%5%: Refinement at specified at %1% lying at %2% in global coords. is outside of the object (%3% to %4%)",
                                            x, x+zero, lower+zero, upper+zero, name());
                    axis->addPoint(zero + x);
                }
            }
        }
    }

    // Have specialization make further axis processing
    return processAxis(axis, geometry, dir);
}

template <> shared_ptr<MeshD<1>>
RectilinearMeshRefinedGenerator<1>::generate(const boost::shared_ptr<plask::GeometryObjectD<2>>& geometry)
{
    shared_ptr<OrderedAxis> mesh = makeGeometryGrid1D(geometry);
    getAxis(mesh, geometry, 0);
    writelog(LOG_DETAIL, "mesh.Rectilinear1D::%s: Generating new mesh (%d)", name(), mesh->size());
    return mesh;
}

template <> shared_ptr<MeshD<2>>
RectilinearMeshRefinedGenerator<2>::generate(const boost::shared_ptr<plask::GeometryObjectD<2>>& geometry)
{
    auto mesh = makeGeometryGrid(geometry);
    auto axis0 = dynamic_pointer_cast<OrderedAxis>(mesh->axis0),
         axis1 = dynamic_pointer_cast<OrderedAxis>(mesh->axis1);
    getAxis(axis0, geometry, 0);
    getAxis(axis1, geometry, 1);

    auto minmax0 = getMinMax(axis0), minmax1 = getMinMax(axis1);
    double asp0 = minmax0.second / minmax1.first, asp1 = minmax1.second / minmax0.first;
    double limit = (1+SMALL) * aspect;
    while (aspect != 0. && (asp0 > limit || asp1 > limit)) {
        if (asp0 > aspect) divideLargestSegment(axis0);
        if (asp1 > aspect) divideLargestSegment(axis1);
        minmax0 = getMinMax(axis0); minmax1 = getMinMax(axis1);
        asp0 = minmax0.second / minmax1.first; asp1 = minmax1.second / minmax0.first;
    }

    mesh->setOptimalIterationOrder();
    writelog(LOG_DETAIL, "mesh.Rectangular2D::%s: Generating new mesh (%dx%d, max. aspect %.0f:1)", name(), 
             mesh->axis0->size(), mesh->axis1->size(), max(asp0, asp1));
    return mesh;
}

template <> shared_ptr<MeshD<3>>
RectilinearMeshRefinedGenerator<3>::generate(const boost::shared_ptr<plask::GeometryObjectD<3>>& geometry)
{
    auto mesh = makeGeometryGrid(geometry);
    auto axis0 = dynamic_pointer_cast<OrderedAxis>(mesh->axis0),
         axis1 = dynamic_pointer_cast<OrderedAxis>(mesh->axis1),
         axis2 = dynamic_pointer_cast<OrderedAxis>(mesh->axis2);
    getAxis(axis0, geometry, 0);
    getAxis(axis1, geometry, 1);
    getAxis(axis2, geometry, 2);

    auto minmax0 = getMinMax(axis0), minmax1 = getMinMax(axis1), minmax2 = getMinMax(axis2);
    double asp0 = minmax0.second / min(minmax1.first, minmax2.first),
           asp1 = minmax1.second / min(minmax0.first, minmax2.first),
           asp2 = minmax2.second / min(minmax0.first, minmax1.first);
    double limit = (1+SMALL) * aspect;
    while (aspect != 0. && (asp0 > limit || asp1 > limit || asp2 > limit)) {
        if (asp0 > aspect) divideLargestSegment(axis0);
        if (asp1 > aspect) divideLargestSegment(axis1);
        if (asp2 > aspect) divideLargestSegment(axis2);
        minmax0 = getMinMax(axis0); minmax1 = getMinMax(axis1); minmax2 = getMinMax(axis2);
        asp0 = minmax0.second / min(minmax1.first, minmax2.first);
        asp1 = minmax1.second / min(minmax0.first, minmax2.first);
        asp2 = minmax2.second / min(minmax0.first, minmax1.first);
    }

    mesh->setOptimalIterationOrder();
    writelog(LOG_DETAIL, "mesh.Rectangular3D::%s: Generating new mesh (%dx%dx%d, max. aspect %.0f:1)", name(),
                          mesh->axis0->size(), mesh->axis1->size(), mesh->axis2->size(), max(asp0, max(asp1, asp2)));
    return mesh;
}


template <int dim>
shared_ptr<OrderedAxis> RectilinearMeshDivideGenerator<dim>::processAxis(shared_ptr<OrderedAxis> axis, const shared_ptr<GeometryObjectD<DIM>>& geometry, size_t dir)
{
    assert(bool(axis));

    if (pre_divisions[dir] == 0) pre_divisions[dir] = 1;
    if (post_divisions[dir] == 0) post_divisions[dir] = 1;

    OrderedAxis& result = *axis.get();

    // Pre-divide each object
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

    return axis;
}


template<>
RectilinearMeshSmoothGenerator<1>::RectilinearMeshSmoothGenerator(): finestep {0.005}, factor {1.2} {}

template<>
RectilinearMeshSmoothGenerator<2>::RectilinearMeshSmoothGenerator(): finestep {0.005, 0.005}, factor {1.2, 1.2} {}

template<>
RectilinearMeshSmoothGenerator<3>::RectilinearMeshSmoothGenerator(): finestep {0.005, 0.005, 0.005}, factor {1.2, 1.2, 1.2} {}


template <int dim>
shared_ptr<OrderedAxis> RectilinearMeshSmoothGenerator<dim>::processAxis(shared_ptr<OrderedAxis> axis, const shared_ptr<GeometryObjectD<DIM>>& geometry, size_t dir)
{
    // Next divide each object
    double x = *axis->begin();
    std::vector<double> points; //points.reserve(...);
    for (auto i = axis->begin()+1; i!= axis->end(); ++i) {
        double w = *i - x;
        if (w+OrderedAxis::MIN_DISTANCE <= finestep[dir])
            continue;
        if (factor[dir] == 1.) {
            double m = ceil(w / finestep[dir]);
            double d = w / m;
            for (size_t i = 1, n = size_t(m); i < n; ++i) points.push_back(x + i*d);
            continue;
        }
        double m = ceil(log(0.5*(w-OrderedAxis::MIN_DISTANCE)/finestep[dir]*(factor[dir]-1)+1) / log(factor[dir])); // number of points in one half
        double end = finestep[dir] * (pow(factor[dir],m)-1) / (factor[dir]-1);
        double last = finestep[dir] * pow(factor[dir],m-1);
        bool odd = 2.*end - w >= last;
        double s;
        if (odd) {
            s = finestep[dir] * 0.5*w / (end-0.5*last);
            m -= 1.;
        } else {
            s = finestep[dir] * 0.5*w / end;
        }
        double dx = 0.;
        for (size_t i = 0, n = size_t(m); i < n; ++i) {
            dx += s; s *= factor[dir];
            points.push_back(x + dx);
        }
        if (odd) { dx += s; points.push_back(x + dx); }
        for (size_t i = 1, n = size_t(m); i < n; ++i) {
            s /= factor[dir]; dx += s;
            points.push_back(x + dx);
        }
        x = *i;
    }
    axis->addOrderedPoints(points.begin(), points.end());

    return axis;
}


template <int dim>
void RectilinearMeshRefinedGenerator<dim>::fromXML(XMLReader& reader, const Manager& manager)
{
    if (reader.getNodeName() == "warnings") {
        warn_missing = reader.getAttribute<bool>("missing", true);
        warn_multiple = reader.getAttribute<bool>("multiple", true);
        warn_outside = reader.getAttribute<bool>("outside", true);
        reader.requireTagEnd();
    } else if (reader.getNodeName() == "refinements") {
        while (reader.requireTagOrEnd()) {
            if (reader.getNodeName() != "axis0" && (dim == 1 || (reader.getNodeName() != "axis1" && (dim == 2 || reader.getNodeName() != "axis2")))) {
                if (dim == 1) throw XMLUnexpectedElementException(reader, "<axis0>");
                if (dim == 2) throw XMLUnexpectedElementException(reader, "<axis0> or <axis1>");
                if (dim == 3) throw XMLUnexpectedElementException(reader, "<axis0>, <axis1>, or <axis2>");
            }
            auto direction = (reader.getNodeName() == "axis0")? typename Primitive<RectilinearMeshRefinedGenerator<dim>::DIM>::Direction(0) :
                             (reader.getNodeName() == "axis1")? typename Primitive<RectilinearMeshRefinedGenerator<dim>::DIM>::Direction(1) :
                                                                typename Primitive<RectilinearMeshRefinedGenerator<dim>::DIM>::Direction(2);
            weak_ptr<GeometryObjectD<RectilinearMeshRefinedGenerator<dim>::DIM>> object
                = manager.requireGeometryObject<GeometryObjectD<RectilinearMeshRefinedGenerator<dim>::DIM>>(reader.requireAttribute("object"));
            PathHints path; if (auto pathattr = reader.getAttribute("path")) path = manager.requirePathHints(*pathattr);
            if (auto by = reader.getAttribute<unsigned>("by")) {
                double objsize = object.lock()->getBoundingBox().size()[unsigned(direction)];
                for (unsigned i = 1; i < *by; ++i) {
                    double pos = objsize * i / *by;
                    addRefinement(direction, object, path, pos);
                }
            } else if (auto every = reader.getAttribute<double>("every")) {
                double objsize = object.lock()->getBoundingBox().size()[unsigned(direction)];
                for (double pos = *every; pos < objsize; pos += *every)
                    addRefinement(direction, object, path, pos);
            } else if (auto pos = reader.getAttribute<double>("at")) {
                addRefinement(direction, object, path, *pos);
            } else
                throw XMLNoAttrException(reader, "at', 'every', or 'by");
            reader.requireTagEnd();
        }
    } else throw XMLUnexpectedElementException(reader, "proper generator configuration tag");
}


template <typename GeneratorT>
static shared_ptr<MeshGenerator> readTrivialGenerator(XMLReader& reader, const Manager&)
{
    reader.requireTagEnd();
    return make_shared<GeneratorT>();
}

static RegisterMeshGeneratorReader rectilinear_simplegenerator_reader  ("ordered.simple",   readTrivialGenerator<OrderedMesh1DSimpleGenerator>);
static RegisterMeshGeneratorReader rectangular2d_simplegenerator_reader("rectangular2d.simple", readTrivialGenerator<RectilinearMesh2DSimpleGenerator>);
static RegisterMeshGeneratorReader rectangular3d_simplegenerator_reader("rectangular3d.simple", readTrivialGenerator<RectilinearMesh3DSimpleGenerator>);


template <int dim>
shared_ptr<MeshGenerator> readRectilinearDivideGenerator(XMLReader& reader, const Manager& manager)
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
        } else if (reader.getNodeName() == "options") {
            result->setGradual(reader.getAttribute<bool>("gradual", result->getGradual()));
            result->setAspect(reader.getAttribute<double>("aspect", result->getAspect()));
            reader.requireTagEnd();
        } else
            result->fromXML(reader, manager);
    }
    return result;
}

static RegisterMeshGeneratorReader rectilinear_dividinggenerator_reader  ("ordered.divide",   readRectilinearDivideGenerator<1>);
static RegisterMeshGeneratorReader rectangular2d_dividinggenerator_reader("rectangular2d.divide", readRectilinearDivideGenerator<2>);
static RegisterMeshGeneratorReader rectangular3d_dividinggenerator_reader("rectangular3d.divide", readRectilinearDivideGenerator<3>);


template <int dim>
shared_ptr<MeshGenerator> readRectilinearSmoothGenerator(XMLReader& reader, const Manager& manager)
{
    auto result = make_shared<RectilinearMeshSmoothGenerator<dim>>();

    std::set<std::string> read;
    while (reader.requireTagOrEnd()) {
        if (read.find(reader.getNodeName()) != read.end())
            throw XMLDuplicatedElementException(std::string("<generator>"), reader.getNodeName());
        read.insert(reader.getNodeName());
        if (reader.getNodeName() == "steps") {
            boost::optional<double> edge = reader.getAttribute<double>("small");
            if (edge) {
                if (reader.hasAttribute("small0")) throw XMLConflictingAttributesException(reader, "small", "small0");
                if (reader.hasAttribute("small1")) throw XMLConflictingAttributesException(reader, "small", "small1");
                if (reader.hasAttribute("small2")) throw XMLConflictingAttributesException(reader, "small", "small2");
                for (int i = 0; i < dim; ++i) result->finestep[i] = *edge;
            } else
                for (int i = 0; i < dim; ++i) result->finestep[i] = reader.getAttribute<size_t>(format("small%d", i), result->finestep[i]);
            boost::optional<double> factor = reader.getAttribute<double>("factor");
            if (factor) {
                if (reader.hasAttribute("factor0")) throw XMLConflictingAttributesException(reader, "factor", "factor0");
                if (reader.hasAttribute("factor1")) throw XMLConflictingAttributesException(reader, "factor", "factor1");
                if (reader.hasAttribute("factor2")) throw XMLConflictingAttributesException(reader, "factor", "factor2");
                for (int i = 0; i < dim; ++i) result->factor[i] = *factor;
            } else
                for (int i = 0; i < dim; ++i) result->factor[i] = reader.getAttribute<size_t>(format("factor%d", i), result->factor[i]);
            reader.requireTagEnd();
        } else if (reader.getNodeName() == "options") {
            result->setAspect(reader.getAttribute<double>("aspect", result->getAspect()));
            reader.requireTagEnd();
        } else
            result->fromXML(reader, manager);
    }
    return result;
}

static RegisterMeshGeneratorReader rectilinear_smoothgenerator_reader  ("ordered.smooth",   readRectilinearSmoothGenerator<1>);
static RegisterMeshGeneratorReader rectangular2d_smoothgenerator_reader("rectangular2d.smooth", readRectilinearSmoothGenerator<2>);
static RegisterMeshGeneratorReader rectangular3d_smoothgenerator_reader("rectangular3d.smooth", readRectilinearSmoothGenerator<3>);






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

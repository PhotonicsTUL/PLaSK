#include "creator.h"

#include <plask/geometry/leaf.h>
#include <plask/geometry/stack.h>

struct BlockCreator: public GeometryObjectCreator {

    virtual plask::shared_ptr<plask::GeometryObject> getObject(int dim) const {
        switch (dim) {
            case 2: return plask::make_shared< plask::Block<2> >(plask::vec(1.0, 1.0));
            case 3: return plask::make_shared< plask::Block<3> >(plask::vec(1.0, 1.0, 1.0));
        }
        return plask::shared_ptr<plask::GeometryObject>();
    }

    virtual std::string getName() const { return "block"; }

};

struct StackCreator: public GeometryObjectCreator {

    virtual plask::shared_ptr<plask::GeometryObject> getObject(int dim) const {
        switch (dim) {
            case 2: return plask::make_shared< plask::StackContainer<2> >();
            case 3: return plask::make_shared< plask::StackContainer<3> >();
        }
        return plask::shared_ptr<plask::GeometryObject>();
    }

    virtual std::string getName() const { return "stack"; }

};

struct MultiStackCreator: public GeometryObjectCreator {

    virtual plask::shared_ptr<plask::GeometryObject> getObject(int dim) const {
        switch (dim) {
            case 2: return plask::make_shared< plask::MultiStackContainer<2> >();
            case 3: return plask::make_shared< plask::MultiStackContainer<3> >();
        }
        return plask::shared_ptr<plask::GeometryObject>();
    }

    virtual std::string getName() const { return "multistack"; }

};

struct ShelfCreator: public GeometryObjectCreator {

    virtual plask::shared_ptr<plask::GeometryObject> getObject(int dim) const {
        switch (dim) {
            case 2: return plask::make_shared< plask::ShelfContainer2D >();
        }
        return plask::shared_ptr<plask::GeometryObject>();
    }

    virtual std::string getName() const { return "shelf2D"; }

    virtual bool supportDimensionsCount(int dim) const {
        return dim == 2;
    }

};

const std::vector<const GeometryObjectCreator*>& getCreators() {
    static std::vector<const GeometryObjectCreator*> vec = { new BlockCreator(), new StackCreator(), new MultiStackCreator(), new ShelfCreator() };
    return vec;
}

const std::vector<const GeometryObjectCreator*>& getCreators(int dim) {
    static std::vector<const GeometryObjectCreator*> vec2 = { new BlockCreator(), new StackCreator(), new MultiStackCreator(), new ShelfCreator() };
    static std::vector<const GeometryObjectCreator*> vec3 = { new BlockCreator(), new StackCreator(), new MultiStackCreator() };
    static std::vector<const GeometryObjectCreator*> empty_vec;
    switch (dim) {
        case 2: return vec2;
        case 3: return vec3;
    }
    return empty_vec;
}

GeometryObjectCreator* GeometryObjectCreator::fromMimeData(const QMimeData * data) {
    QByteArray ptrData = data->data(MIME_PTR_TO_CREATOR);
    QDataStream stream(&ptrData, QIODevice::ReadOnly);

    GeometryObjectCreator* creator = 0;
    stream.readRawData(reinterpret_cast<char*>(&creator), sizeof(creator));
    return creator;
}

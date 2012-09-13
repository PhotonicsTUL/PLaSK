#include "tree.h"

#include "document.h"
#include "geomwrapper/register.h"
#include "modelext/text.h"

GeometryTreeItem::~GeometryTreeItem() {
    //model->changePersistentIndex(this, QModelIndex());    //good, if will store index
    disconnectOnChanged(object);
}

void GeometryTreeItem::ensureInitialized() {
    if (!childrenInitialized) {
        constructChildrenItems();
        childrenInitialized = true;
    }
    if (!miniatureInitialized) {
        miniature = object->getMiniature(50, 50);
        miniatureInitialized = true;
    }
}

void GeometryTreeItem::appendChildrenItemsHelper(const plask::shared_ptr<plask::GeometryObject>& elem, bool reverse) {
    //TODO reverse support
    std::size_t chCount = elem->getRealChildrenCount();
    if (elem->isContainer()) {
        for (std::size_t i = 0; i < chCount; ++i)
            childItems.emplace_back(new InContainerTreeItem(this, ext(elem->getRealChildAt(i))));
    } else {
        for (std::size_t i = 0; i < chCount; ++i)   //should be 0 or 1 child here
            childItems.emplace_back(new GeometryTreeItem(this, ext(elem->getRealChildAt(i))));
    }
}

void GeometryTreeItem::appendChildrenItems() {
    if (auto e = getLowerWrappedObject()) {
        appendChildrenItemsHelper(e->wrappedObject->shared_from_this());
    }
}

void GeometryTreeItem::constructChildrenItems() {
    childItems.clear();
    appendChildrenItems();
}

void GeometryTreeItem::deinitializeChildren() {
    emit model->layoutAboutToBeChanged();
    QModelIndexList indexes;
    for (auto& c: childItems)
        c->getExistsSubtreeIndexes(indexes);
    childItems.clear();
    childrenInitialized = false;
    miniatureInitialized = false;
    for (auto& c: indexes) model->changePersistentIndex(c, QModelIndex());
    emit model->layoutChanged();
}

void GeometryTreeItem::getExistsSubtreeIndexes(QModelIndexList &dst, std::size_t indexInParent)
{
    for (std::size_t i = 0; i < childItems.size(); ++i)
        childItems[i]->getExistsSubtreeIndexes(dst, i);
    dst.append(model->createIndex(indexInParent, 0, this));
}

plask::shared_ptr<ObjectWrapper> GeometryTreeItem::parent() {
    return parentItem ?
        parentItem->getLowerWrappedObject() :
        plask::shared_ptr<ObjectWrapper>();
}

GeometryTreeItem::GeometryTreeItem(GeometryTreeItem* parentItem, std::size_t index)
: model(parentItem->model), childrenInitialized(false), miniatureInitialized(false), parentItem(parentItem)/*, inParentIndex(index)*/ {
    if (auto parent_ptr = parent()) {
        auto child = ext(parent_ptr->wrappedObject->getRealChildAt(index));
        object = child;
        connectOnChanged(child);
        //constructChildrenItems(child);
    }
}

GeometryTreeItem::GeometryTreeItem(GeometryTreeItem* parentItem, const plask::shared_ptr<ObjectWrapper> &object/*, std::size_t index*/)
: model(parentItem->model), childrenInitialized(false), miniatureInitialized(false), parentItem(parentItem), object(object)/*, inParentIndex(index)*/ {
    connectOnChanged(object);
    //constructChildrenItems(object);
}

GeometryTreeItem::GeometryTreeItem(GeometryTreeModel *model, const plask::shared_ptr<ObjectWrapper> &object)
: model(model), childrenInitialized(false), miniatureInitialized(false), parentItem(nullptr), object(object)/*, inParentIndex(index)*/ {
    connectOnChanged(object);
}

GeometryTreeItem::GeometryTreeItem(const std::vector< plask::shared_ptr<plask::Geometry> >& rootObjects, GeometryTreeModel* model)
: model(model), childrenInitialized(true), miniatureInitialized(true), parentItem(0)/*, inParentIndex(0)*/ {
    for (std::size_t i = 0; i < rootObjects.size(); ++i)
        childItems.emplace_back(new GeometryTreeItem(this, ext(rootObjects[i])));
}

GeometryTreeItem * GeometryTreeItem::child(std::size_t index) {
    ensureInitialized();
    if (index > childItems.size()) return nullptr;
    return childItems[index].get();
}

std::size_t GeometryTreeItem::indexInParent() const {
    if (!parentItem) return 0;
    for (std::size_t i = 0; i < parentItem->childItems.size(); ++i)
        if (parentItem->childItems[i].get() == this) return i;
    //TODO
    return 0;

//    return std::find(parentItem->childItems.begin(), parentItem->childItems.end(),
//                     [&](std::unique_ptr<GeometryTreeItem>& p) { return p.get() == this; }) - parentItem->childItems.begin();

   // return parentItem->childItems.indexOf(const_cast<GeometryTreeItem*>(this));
   // return inParentIndex;
}

QString GeometryTreeItem::objectText(plask::shared_ptr<ObjectWrapper> object) const {
    return object->toStr();
}

QVariant GeometryTreeItem::data(int column) {
    return objectText(object);
}

void GeometryTreeItem::fillPropertyBrowser(BrowserWithManagers& browser) {
    object->setupPropertiesBrowser(browser);
}

QModelIndex GeometryTreeItem::getIndex() {
    return model->createIndex(indexInParent(), 0, this);
}

void GeometryTreeItem::onChanged(const ObjectWrapper::Event& evt) {
    if (evt.isDelete()) return;
    miniatureInitialized = false;
 /*   if (childrenInitialized && evt.hasChangedChildrenList()) {
        if (evt.isDelgatedFromWrappedObject() &&
          evt.delegatedEvent->hasAnyFlag(plask::GeometryObject::Event::CHILDREN_REMOVE | plask::GeometryObject::Event::CHILDREN_INSERT)) {
            plask::GeometryObject::ChildrenListChangedEvent* details =
                    static_cast<plask::GeometryObject::ChildrenListChangedEvent*>(evt.delegatedEvent);
            if (details->hasFlag(plask::GeometryObject::Event::CHILDREN_REMOVE)) {
                model->beginRemoveRows(getIndex(), details->beginIndex, details->endIndex-1);
                childItems.erase(childItems.begin() + details->beginIndex, childItems.begin() + details->endIndex);
                model->endRemoveRows();
            } else {
                //model->beginInsertRows(parent(), details->beginIndex, details->endIndex-1);
                //...
                //model->endInsertRows();
                deinitializeChildren();
            }
        } else
            deinitializeChildren();
    }*/
    if (/*childrenInitialized &&*/ evt.hasChangedChildrenList())
        deinitializeChildren();
    auto index = getIndex();
    emit model->dataChanged(index, index);
}

void GeometryTreeItem::connectOnChanged(const plask::shared_ptr<ObjectWrapper>& el) {
    if (el) el->changedConnectMethod(this, &GeometryTreeItem::onChanged);
}

void GeometryTreeItem::disconnectOnChanged(const plask::shared_ptr<ObjectWrapper> &el) {
    if (el) el->changedDisconnectMethod(this, &GeometryTreeItem::onChanged);
}

bool GeometryTreeItem::removeRange(std::size_t begin_index, std::size_t end_index) {
    if (auto e = getLowerWrappedObject()) {
        if (e->wrappedObject->removeRange(begin_index, end_index)) {
            //childItems.erase(childItems.begin() + begin_index, childItems.begin() + end_index);
            //deinitializeChildren();
            return true;
        }
    }
    return false;
}

bool GeometryTreeItem::tryInsert(plask::shared_ptr<plask::GeometryObject> object, int index) {
    auto this_elem = getLowerWrappedObject();
    if (this_elem->tryInsert(object, index)) {
       // childItems.emplace(childItems.begin() + index,
       //                   new InContainerTreeItem(this, ext(this_elem->wrappedObject->getRealChildAt(index)), index));
        //deinitializeChildren();
        return true;
    } else
        return false;
}

bool GeometryTreeItem::tryInsert(const GeometryObjectCreator& object_creator, int index) {
    return getLowerWrappedObject()->tryInsert(object_creator, index);
}

int GeometryTreeItem::getInsertionIndexForPoint(const plask::Vec<2, double> &point)
{
    return getLowerWrappedObject()->getInsertionIndexForPoint(point);
}

int GeometryTreeItem::tryInsertRow2D(const GeometryObjectCreator &to_insert, const plask::Vec<2, double> &point)
{
    auto this_elem = getLowerWrappedObject();
    int index = this_elem->tryInsertNearPoint2D(to_insert, point);
    if (index >= 0) {
       // childItems.emplace(childItems.begin() + index,
       //                   new InContainerTreeItem(this, ext(this_elem->wrappedObject->getRealChildAt(index)), index));
     //   deinitializeChildren();
    }
    return index;
}

plask::Box2D GeometryTreeItem::getInsertPlace2D(const GeometryObjectCreator &to_insert, const plask::Vec<2, double> &point)
{
    return getLowerWrappedObject()->getInsertPlace2D(to_insert, point);
}

// ---------- InContainerTreeItem -----------

/*void InContainerTreeItem::appendChildrenItems() {
    if (auto elem = object.lock()) {
        std::size_t chCount = elem->getRealChildrenCount();
        if (chCount == 0) return;
        appendChildrenItemsHelper(elem->getRealChildAt(0));
    }
}*/

QString InContainerTreeItem::objectText(plask::shared_ptr<ObjectWrapper> object) const {
    if (object->wrappedObject->getRealChildrenCount() == 0) return object->toStr();
    QString result = ext(object->wrappedObject->getRealChildAt(0))->toStr();
    result += "\nat ";
    if (object->wrappedObject->getDimensionsCount() == 2) {
        result += toStr(static_cast<plask::Translation<2>&>(*object->wrappedObject).translation);
    } else
        result += toStr(static_cast<plask::Translation<3>&>(*object->wrappedObject).translation);
    return result;
}

void InContainerTreeItem::fillPropertyBrowser(BrowserWithManagers& browser) {
    auto p = parent();  //should be a container
    if (p) {
        p->setupPropertiesBrowserForChild(indexInParent(), browser);
    } else {
        if (object->wrappedObject->getRealChildrenCount() == 0) return;
        ext(object->wrappedObject->getRealChildAt(0))->setupPropertiesBrowser(browser);
    }
}

// ----------- GeometryTreeModel ------------

GeometryTreeModel::GeometryTreeModel(QObject *parent)
    : QAbstractItemModel(parent) {
}

GeometryTreeModel::~GeometryTreeModel() {
    clear();    //this is becouse crash on application exit
}

void GeometryTreeModel::save(plask::XMLWriter::Element &root_object) {
    plask::XMLWriter::Element geometry_object(root_object, plask::Manager::TAG_NAME_GEOMETRY);
    NamesFromExtensions writerCB;
    for (std::unique_ptr<RootItem>& to_save: rootItems)
        writerCB.prerareToAutonaming(*to_save->geometry);
    for (std::unique_ptr<RootItem>& to_save: rootItems)
        to_save->geometry->writeXML(geometry_object, writerCB);
}

void GeometryTreeModel::refresh(const std::vector< plask::shared_ptr<plask::Geometry> >& roots) {
    beginResetModel();
    rootItems.clear();
    for (std::size_t i = 0; i < roots.size(); ++i)
        rootItems.emplace_back(new RootItem(this, roots[i]));
    endResetModel();
}

void GeometryTreeModel::clear() {
    beginResetModel();
    rootItems.clear();
    endResetModel();
}

QModelIndex GeometryTreeModel::index(int row, int column, const QModelIndex &parent) const {
    if (!hasIndex(row, column, parent))
        return QModelIndex();

    if (!parent.isValid())  //root
        return createIndex(row, column, rootItems[row].get());

    GeometryTreeItem* parentItem = toItem(parent);
    GeometryTreeItem* childItem = parentItem->child(row);

    return childItem ?
        createIndex(row, column, childItem) :
        QModelIndex();
}

QModelIndex GeometryTreeModel::parent(const QModelIndex &index) const {
    if (!index.isValid())
        return QModelIndex();

    GeometryTreeItem *childItem = toItem(index);
    GeometryTreeItem *parentItem = childItem->parentItem;

    if (parentItem == nullptr) return QModelIndex();

    return createIndex(parentItem->indexInParent(), 0, parentItem);
}

int GeometryTreeModel::rowCount(const QModelIndex &parent) const {
    if (parent.column() > 0)    //TODO
        return 0;
    return parent.isValid() ? toItem(parent)->childCount() : rootItems.size();
}

int GeometryTreeModel::columnCount(const QModelIndex &parent) const {
    return parent.isValid() ? toItem(parent)->columnCount() : 1;
}

QVariant GeometryTreeModel::data(const QModelIndex &index, int role) const {
    if (!index.isValid())
        return QVariant();

    GeometryTreeItem* item = toItem(index);

    switch (role) {
        case Qt::DecorationRole: return item->icon();
        case Qt::DisplayRole: return item->data(index.column());
    }

    return QVariant();
}

Qt::ItemFlags GeometryTreeModel::flags(const QModelIndex &index) const
{
    Qt::ItemFlags defaultFlags = QAbstractItemModel::flags(index);

    if (index.isValid())
        return Qt::ItemIsDragEnabled | Qt::ItemIsDropEnabled | defaultFlags;
    else
        return Qt::ItemIsDropEnabled | defaultFlags;

    /*if (!index.isValid())
        return 0;

    return Qt::ItemIsEnabled | Qt::ItemIsSelectable | Qt::ItemIsDropEnabled;*/ //| Qt::ItemIsEditable;
}

QVariant GeometryTreeModel::headerData(int section, Qt::Orientation orientation, int role) const {

    if (role != Qt::DisplayRole)
        return QVariant();

    return "Description";
}

bool GeometryTreeModel::removeRows(int position, int rows, const QModelIndex &parent) {
    if (rows == 0) return true;
    if (parent.isValid()) {
        return toItem(parent)->remove(position, rows);
    } else {
        beginRemoveRows(parent, position, position + rows - 1);
        rootItems.erase(rootItems.begin() + position, rootItems.begin() + position + rows);
        endRemoveRows();
        return true;
    }
}

bool GeometryTreeModel::insertRow(plask::shared_ptr<plask::GeometryObject> to_insert, const QModelIndex &parent, int position) {
    if (!parent.isValid()) return false;

   // beginInsertRows(parent, position, position);
    bool result = toItem(parent)->tryInsert(to_insert, position);
   // endInsertRows();
    return result;
}

int GeometryTreeModel::insertRow2D(const GeometryObjectCreator &to_insert, const QModelIndex &parent, const plask::Vec<2, double> &point) {
    if (!parent.isValid()) return -1;

    GeometryTreeItem* item = toItem(parent);
    int p = item->getInsertionIndexForPoint(point);
    if (p == -1) return -1;
 //   beginInsertRows(parent, p, p);
    p = item->tryInsertRow2D(to_insert, point);
 //   endInsertRows();
    return p;
}

plask::Box2D GeometryTreeModel::insertPlace2D(const GeometryObjectCreator& to_insert, const QModelIndex &parent, const plask::Vec<2, double>& point) {
    return toItem(parent)->getInsertPlace2D(to_insert, point);
}

bool GeometryTreeModel::dropMimeData(const QMimeData *data, Qt::DropAction action, int row, int column, const QModelIndex &parent) {
    if (action == Qt::IgnoreAction)
        return true;

    if (data->hasFormat(MIME_PTR_TO_CREATOR)) {

        //for now we doesn't support creating top-level geometries by creators
        if (!parent.isValid()) return false;

        toItem(parent)->tryInsert(*GeometryObjectCreator::fromMimeData(data), row >= 0 ? row : 0);

        return true;
    }

    return false;
}

QStringList GeometryTreeModel::mimeTypes() const
{
    //TODO append to list also type for internal copy/moves
    return QStringList() << MIME_PTR_TO_CREATOR;
}

Qt::DropActions GeometryTreeModel::supportedDropActions() const {
    return Qt::CopyAction | Qt::MoveAction;
}


void GeometryTreeModel::appendGeometry(plask::shared_ptr<plask::Geometry> geometry) {
    beginInsertRows(QModelIndex(), rootItems.size(), rootItems.size());
    rootItems.emplace_back(new RootItem(this, geometry));
    endInsertRows();
}

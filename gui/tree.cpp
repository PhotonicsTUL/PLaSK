#include "tree.h"

#include "document.h"
#include "modelext/map.h"
#include "modelext/text.h"

void GeometryTreeItem::ensureInitialized() {
    if (!childrenInitialized) {
        constructChildrenItems();
        childrenInitialized = true;
    }
    if (!miniatureInitialized) if (auto e = element.lock()) {
        miniature = ext(*e).getMiniature(50, 50);
        miniatureInitialized = true;
    }
}

void GeometryTreeItem::appendChildrenItemsHelper(const plask::shared_ptr<plask::GeometryElement>& elem, bool reverse) {
    //TODO reverse support
    std::size_t chCount = elem->getRealChildrenCount();
    if (elem->isContainer()) {
        for (int i = 0; i < chCount; ++i)
            childItems.append(new InContainerTreeItem(this, elem->getRealChildAt(i), i));
    } else {
        for (int i = 0; i < chCount; ++i)   //should be 0 or 1 child here
            childItems.append(new GeometryTreeItem(this, elem->getRealChildAt(i), i));
    }
}

void GeometryTreeItem::appendChildrenItems() {
    if (auto e = getLowerWrappedElement()) {
        appendChildrenItemsHelper(e);
    }
}

void GeometryTreeItem::constructChildrenItems() {
    qDeleteAll(childItems);
    childItems.clear();
    appendChildrenItems();
}

plask::shared_ptr<plask::GeometryElement> GeometryTreeItem::parent() {
    return parentItem ?
        parentItem->getLowerWrappedElement() :
        plask::shared_ptr<plask::GeometryElement>();
}

GeometryTreeItem::GeometryTreeItem(GeometryTreeItem* parentItem, std::size_t index)
: model(parentItem->model), childrenInitialized(false), miniatureInitialized(false), parentItem(parentItem), inParentIndex(index) {
    if (auto parent_ptr = parent()) {
        auto child = parent_ptr->getRealChildAt(index);
        element = child;
        connectOnChanged(child);
        //constructChildrenItems(child);
    }
}

GeometryTreeItem::GeometryTreeItem(GeometryTreeItem* parentItem, const plask::shared_ptr<plask::GeometryElement>& element, std::size_t index)
: model(parentItem->model), childrenInitialized(false), miniatureInitialized(false), parentItem(parentItem), element(element), inParentIndex(index) {
    connectOnChanged(element);
    //constructChildrenItems(element);
}

GeometryTreeItem::GeometryTreeItem(const std::vector< plask::shared_ptr<plask::GeometryElement> >& rootElements, GeometryTreeModel* model)
: model(model), childrenInitialized(true), miniatureInitialized(true), parentItem(0), inParentIndex(0) {
    for (int i = 0; i < rootElements.size(); ++i)
        childItems.append(new GeometryTreeItem(this, rootElements[i], i));
}

GeometryTreeItem::~GeometryTreeItem() {
    disconnectOnChanged(element.lock());
    qDeleteAll(childItems);
}

GeometryTreeItem * GeometryTreeItem::child(std::size_t index) {
    ensureInitialized();
    if (index > childItems.size()) return nullptr;
    return childItems[index];
}

std::size_t GeometryTreeItem::indexInParent() const {
   // if (!parentItem) return 0;
   // return parentItem->childItems.indexOf(const_cast<GeometryTreeItem*>(this));
    return inParentIndex;
}

QString GeometryTreeItem::elementText(plask::GeometryElement& element) const {
    return ext(element).toStr();
}

QVariant GeometryTreeItem::data(int column) {
    if (plask::shared_ptr<plask::GeometryElement> e = element.lock()) {
        return elementText(*e);
    } else  //should never happen
        QVariant();
}

void GeometryTreeItem::fillPropertyBrowser(BrowserWithManagers& browser) {
    if (plask::shared_ptr<plask::GeometryElement> e = element.lock()) {
        ext(*e).setupPropertiesBrowser(browser);
    }
}

QModelIndex GeometryTreeItem::getIndex() {
    return model->createIndex(inParentIndex, 0, this);
}

void GeometryTreeItem::onChanged(const plask::GeometryElement::Event& evt) {
    auto index = getIndex();
    miniatureInitialized = false;
    if (evt.hasChangedChildrenList()) childrenInitialized = false;
    emit model->dataChanged(index, index);  //TODO czy ten sygnał jest wystarczający jeśli lista dzieci się zmieniła?
}

void GeometryTreeItem::connectOnChanged(const plask::shared_ptr<plask::GeometryElement>& el) {
    if (el) el->changed.connect(boost::bind(&GeometryTreeItem::onChanged, this, _1));
}

void GeometryTreeItem::disconnectOnChanged(const plask::shared_ptr<plask::GeometryElement>& el) {
    if (el) el->changed.disconnect(boost::bind(&GeometryTreeItem::onChanged, this, _1));
}

bool GeometryTreeItem::remove(std::size_t begin_index, std::size_t end_index) {
    if (auto e = getLowerWrappedElement()) {
        return e->removeRange(begin_index, end_index);
    } else
        return false;
}

// ---------- InContainerTreeItem -----------

/*void InContainerTreeItem::appendChildrenItems() {
    if (auto elem = element.lock()) {
        std::size_t chCount = elem->getRealChildrenCount();
        if (chCount == 0) return;
        appendChildrenItemsHelper(elem->getRealChildAt(0));
    }
}*/

QString InContainerTreeItem::elementText(plask::GeometryElement &element) const {
    if (element.getRealChildrenCount() == 0) return ext(element).toStr();
    QString result = ext(*element.getRealChildAt(0)).toStr();
    result += "\nat ";
    if (element.getDimensionsCount() == 2) {
        result += toStr(static_cast<plask::Translation<2>&>(element).translation);
    } else
        result += toStr(static_cast<plask::Translation<3>&>(element).translation);
    return result;
}

void InContainerTreeItem::fillPropertyBrowser(BrowserWithManagers& browser) {
    if (plask::shared_ptr<plask::GeometryElement> e = element.lock()) {
        auto p = parent();  //should be a container
        if (p) {
            ext(*p).setupPropertiesBrowserForChild(indexInParent(), browser);
        } else {
            if (e->getRealChildrenCount() == 0) return;
            ext(*e->getRealChildAt(0)).setupPropertiesBrowser(browser);
        }
    }
}

// ----------- GeometryTreeModel ------------

GeometryTreeModel::GeometryTreeModel(Document& document, QObject *parent)
    : QAbstractItemModel(parent), rootItem(0) {
    refresh(document);
}

GeometryTreeModel::~GeometryTreeModel() {
    delete rootItem;
}

void GeometryTreeModel::refresh(Document& document) {
    delete rootItem;
    rootItem = new GeometryTreeItem(document.manager.roots, this);
    reset();
}

QModelIndex GeometryTreeModel::index(int row, int column, const QModelIndex &parent) const {
    if (!hasIndex(row, column, parent))
        return QModelIndex();

    GeometryTreeItem* parentItem = parent.isValid() ?
                static_cast<GeometryTreeItem*>(parent.internalPointer()) :
                rootItem;

    GeometryTreeItem *childItem = parentItem->child(row);
    return childItem ?
        createIndex(row, column, childItem) :
        QModelIndex();
}

QModelIndex GeometryTreeModel::parent(const QModelIndex &index) const {
    if (!index.isValid())
        return QModelIndex();

    GeometryTreeItem *childItem = static_cast<GeometryTreeItem*>(index.internalPointer());
    GeometryTreeItem *parentItem = childItem->parentItem;

    if (parentItem == rootItem)
        return QModelIndex();

    return createIndex(parentItem->indexInParent(), 0, parentItem);
}

int GeometryTreeModel::rowCount(const QModelIndex &parent) const {
    if (parent.column() > 0)    //TODO
        return 0;
    return toItem(parent)->childCount();
}

int GeometryTreeModel::columnCount(const QModelIndex &parent) const {
    return toItem(parent)->columnCount();
}

QVariant GeometryTreeModel::data(const QModelIndex &index, int role) const {
    if (!index.isValid())
        return QVariant();

    GeometryTreeItem* item = static_cast<GeometryTreeItem*>(index.internalPointer());

    if (role == Qt::DecorationRole)
        return item->icon();

    if (role != Qt::DisplayRole)
        return QVariant();

    return item->data(index.column());
}

Qt::ItemFlags GeometryTreeModel::flags(const QModelIndex &index) const
{
    if (!index.isValid())
        return 0;

    return Qt::ItemIsEnabled | Qt::ItemIsSelectable; //| Qt::ItemIsEditable;
}

QVariant GeometryTreeModel::headerData(int section, Qt::Orientation orientation, int role) const {

    if (role != Qt::DisplayRole)
        return QVariant();

    return "Description";
}

bool GeometryTreeModel::removeRows(int position, int rows, const QModelIndex &parent) {
    beginRemoveRows(parent, position, position + rows - 1);
    bool result = toItem(parent)->remove(position, rows);
    endRemoveRows();
    return result;
}

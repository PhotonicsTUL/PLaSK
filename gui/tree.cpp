#include "tree.h"

#include "document.h"
#include "modelext/map.h"
#include "modelext/text.h"

void GeometryTreeItem::ensureInitialized() {
    if (initialized) return;
    if (auto e = element.lock()) constructChildrenItems(e);
    initialized = true;
}

void GeometryTreeItem::constructChildrenItems(const plask::shared_ptr<plask::GeometryElement>& elem) {
    std::size_t chCount = elem->getRealChildCount();
    if (elem->isContainer()) {
        for (int i = 0; i < chCount; ++i)
            childItems.append(new InContainerTreeItem(this, i));
    } else {
        for (int i = 0; i < chCount; ++i)
            childItems.append(new GeometryTreeItem(this, i));
    }
}

plask::shared_ptr<plask::GeometryElement> GeometryTreeItem::parent() {
    return parentItem ?
        parentItem->element.lock() :
        plask::shared_ptr<plask::GeometryElement>();
}

GeometryTreeItem::GeometryTreeItem(GeometryTreeItem* parentItem, std::size_t index)
: initialized(false), parentItem(parentItem), inParentIndex(index) {
    if (auto parent_ptr = parent()) {
        auto child = parent_ptr->getRealChildAt(index);
        element = child;
        //constructChildrenItems(child);
    }
}

GeometryTreeItem::GeometryTreeItem(GeometryTreeItem* parentItem, const plask::shared_ptr<plask::GeometryElement>& element, std::size_t index)
: initialized(false), parentItem(parentItem), element(element), inParentIndex(index) {
    //constructChildrenItems(element);
}

GeometryTreeItem::GeometryTreeItem(const std::vector< plask::shared_ptr<plask::GeometryElement> >& rootElements)
: initialized(false), parentItem(0), inParentIndex(0) {
    for (int i = 0; i < rootElements.size(); ++i)
        childItems.append(new GeometryTreeItem(this, rootElements[i], i));
}

GeometryTreeItem::~GeometryTreeItem() {
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

QVariant GeometryTreeItem::data(int column) const {
    if (plask::shared_ptr<plask::GeometryElement> e = element.lock()) {
        return elementText(*e);
    } else  //should never happen
        QVariant();
}


// ---------- InContainerTreeItem -----------

void InContainerTreeItem::constructChildrenItems(const plask::shared_ptr<plask::GeometryElement>& elem) {
    std::size_t chCount = elem->getRealChildCount();
    if (chCount == 0) return;
    GeometryTreeItem::constructChildrenItems(elem->getRealChildAt(0));
}

QString InContainerTreeItem::elementText(plask::GeometryElement &element) const {
    if (element.getRealChildCount() == 0) return ext(element).toStr();
    QString result = ext(*element.getRealChildAt(0)).toStr();
    result += "\nat ";
    if (element.getDimensionsCount() == 2) {
        result += toStr(static_cast<plask::Translation<2>&>(element).translation);
    } else
        result += toStr(static_cast<plask::Translation<3>&>(element).translation);
    return result;
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
    rootItem = new GeometryTreeItem(document.manager.roots);
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

    GeometryTreeItem *parentItem = parent.isValid() ?
                static_cast<GeometryTreeItem*>(parent.internalPointer()) :
                rootItem;

    return parentItem->childCount();
}

int GeometryTreeModel::columnCount(const QModelIndex &parent) const {
    if (parent.isValid())
        return static_cast<GeometryTreeItem*>(parent.internalPointer())->columnCount();
    else
        return rootItem->columnCount();
}

QVariant GeometryTreeModel::data(const QModelIndex &index, int role) const {
    if (!index.isValid())
        return QVariant();

    //if (role == Qt::BackgroundRole)
    //    return index.row() & 1 ? QVariant() : QVariant(QColor(220, 210, 200));

    //if (role == Qt::DecorationRole)
    //

    if (role != Qt::DisplayRole)
        return QVariant();

    return static_cast<GeometryTreeItem*>(index.internalPointer())->data(index.column());
}

Qt::ItemFlags GeometryTreeModel::flags(const QModelIndex &index) const
{
    if (!index.isValid())
        return 0;

    return Qt::ItemIsEnabled | Qt::ItemIsSelectable;
}

QVariant GeometryTreeModel::headerData(int section, Qt::Orientation orientation, int role) const {

    if (role != Qt::DisplayRole)
        return QVariant();

    return "Description";
}


#ifndef PLASK_GUI_TREE_H
#define PLASK_GUI_TREE_H

#include <QAbstractItemModel>
#include <QPixmap>
#include <plask/geometry/element.h>
#include <plask/geometry/space.h>
#include <plask/memory.h>
#include <memory>
#include "geomwrapper/register.h"

#include "utils/propbrowser.h"

QT_BEGIN_NAMESPACE
class QAbstractItemModel;
class QObject;
class QModelIndex;
class QPixmap;
QT_END_NAMESPACE

struct GeometryTreeModel;

/**
 * Geometry tree item. Holds geometry element wrapper.
 */
class GeometryTreeItem {

protected:

    /// Pointer to tree model, used to call update.
    GeometryTreeModel *model;

    /// Children of this item.
    std::vector< std::unique_ptr<GeometryTreeItem> > childItems;

    /// Cache for miniature
    QPixmap miniature;

    /**
     * True only if this item was initialized. Its children are on childItems list, etc.
     */
    bool childrenInitialized;

    /**
     * True only if this item miniature was initialized.
     */
    bool miniatureInitialized;

    /**
     * Ensure that this item is initialized (initialize it if its not).
     */
    void ensureInitialized();

    /**
     * Append to childItems children items for an given element.
     * @param elem element for which children should be constructed, typically (but not always) same as wrapped element,
     *  sometimes same as child of wrapped element
     * @param reverse append children in reverse order
     */
    void appendChildrenItemsHelper(const plask::shared_ptr<plask::GeometryElement>& elem, bool reverse = false);

    /**
     * Append to childItems children items for wrapped element.
     */
    virtual void appendChildrenItems();

    /**
     * Clear children items list and call appendChildrenItems()
     */
    void constructChildrenItems();

    /**
     * Delete children cache and childrenInitialized to @c false.
     */
    void deinitializeChildren();    //TODO can't be used, to remove

public:

    void getExistsSubtreeIndexes(QModelIndexList& dst);

    virtual plask::shared_ptr<ElementWrapper> getLowerWrappedElement() {
        return element;
    }

    /**
     * Index of this in parents item childItems. 0 for root.
     */
    //std::size_t inParentIndex;

    /**
     * Parent of this in tree.
     */
    GeometryTreeItem* parentItem;

    /**
     * Wrapped geometry element.
     */
    plask::shared_ptr<ElementWrapper> element;

    /**
     * Get geometry element from parent item.
     * @return geometry element from parent item or plask::shared_ptr<plask::GeometryElement>()
     *  if can't get it (for example in case of root or parent doesn't wrap existing geometry element object).
     */
    plask::shared_ptr<ElementWrapper> parent();

    /**
     * Construct item using parent item and index in it.
     * @param parentItem parent item which wrap existing geometry element
     * @param index (future) index of this in parent childItems
     */
    GeometryTreeItem(GeometryTreeItem* parentItem, std::size_t index);

    /**
     * Construct item.
     * @param parentItem parent item
     * @param element wrapped element
     * @param index (future) index of this in parent childItems
     */
    GeometryTreeItem(GeometryTreeItem* parentItem, const plask::shared_ptr<ElementWrapper>& element, std::size_t index);

    /**
     * Construct root item (with parentItem = nullptr).
     * @param rootElements children of roots element (showing in tree as roots)
     * @param model model to notify about changes
     */
    GeometryTreeItem(const std::vector< plask::shared_ptr<plask::Geometry> >& rootElements, GeometryTreeModel* model);

    /// Delete children items and disconnect onChanged.
    virtual ~GeometryTreeItem();

    /**
     * Get child item with given @p index.
     * @param index index of child item
     * @return child item or nullptr in case of invalid index
     */
    GeometryTreeItem* child(std::size_t index);

    /**
     * Get number of children.
     * @return number of children
     */
    std::size_t childCount() { ensureInitialized(); return childItems.size(); }

    /// @return 1
    std::size_t columnCount() const { return 1; }

    /// @return inParentIndex
    std::size_t indexInParent() const;

    /**
     * Get text representation of element wrapped by this.
     * @param element element wrapped by this
     * @return text representation of an @p element
     */
    virtual QString elementText(plask::shared_ptr<ElementWrapper> element) const;

    /**
     * @return string returned by elementText or empty QVariant if this wraps non-existing element
     */
    QVariant data(int column);

    /// @return icon
    const QPixmap& icon() { ensureInitialized(); return miniature; }

    /**
     * Add properties of this item to browser.
     */
    virtual void fillPropertyBrowser(BrowserWithManagers& browser);

    /**
     * @return index of this item
     */
    QModelIndex getIndex();

    /**
     * Called when wrapped geometry element was changed.
     * @param evt information about event from model
     */
    void onChanged(const ElementWrapper::Event& evt);

    /**
     * Connect onChanged method to el->changed.
     * @param el element, typically this->element.lock()
     */
    void connectOnChanged(const plask::shared_ptr<ElementWrapper> &el);

    /**
     * Disconnect onChanged method from el->changed.
     * @param el element, typically this->element.lock()
     */
    void disconnectOnChanged(const plask::shared_ptr<ElementWrapper>& el);

    //TODO new subclass for root item and reimplementation of this which remove from manager
    virtual bool removeRange(std::size_t begin_index, std::size_t end_index);

    /**
     * Remove given number of @p rows starting from given @p position.
     * @param position index of first child to remove
     * @param rows number of children to remove
     * @return @c true if remove something
     */
    bool remove(int position, int rows) { return removeRange(position, position + rows); }

    bool tryInsert(plask::shared_ptr<plask::GeometryElement> element, int index);

    int getInsertionIndexForPoint(const plask::Vec<2, double>& point);

    int tryInsertRow2D(const GeometryElementCreator& to_insert, const plask::Vec<2, double>& point);

    plask::Box2D getInsertPlace2D(const GeometryElementCreator& to_insert, const plask::Vec<2, double>& point);

};

/**
 * Wrap translation and child of this translation inside container
 * (this two elements are represented as one item in tree).
 */
struct InContainerTreeItem: public GeometryTreeItem {

    plask::shared_ptr<ElementWrapper> lowerElement;

private:
    void initLowerElement() {
        std::size_t chCount = element->wrappedElement->getRealChildrenCount();
        if (chCount == 0) lowerElement = plask::shared_ptr<ElementWrapper>();
        else lowerElement = ext(element->wrappedElement->getRealChildAt(0));
    }
public:

    virtual plask::shared_ptr<ElementWrapper> getLowerWrappedElement() {
        return lowerElement;
    }

    /**
     * @param parentItem parent item, must wrap plask::Translation<2> or plask::Translation<3>
     * @param index (future) index of this in parent childItems
     */
    InContainerTreeItem(GeometryTreeItem* parentItem, std::size_t index)
        : GeometryTreeItem(parentItem, index) {
        initLowerElement();
    }

    InContainerTreeItem(GeometryTreeItem* parentItem, const plask::shared_ptr<ElementWrapper>& element, std::size_t index)
        : GeometryTreeItem(parentItem, element, index) {
        initLowerElement();
    }

    //virtual void appendChildrenItems();

    virtual QString elementText(plask::shared_ptr<ElementWrapper> element) const;

    virtual void fillPropertyBrowser(BrowserWithManagers& browser);
};


class Document;

/**
 * Implementation of QAbstractItemModel which holds and use GeometryTreeItem.
 */
class GeometryTreeModel: public QAbstractItemModel {

    Q_OBJECT

    /// Root of tree, not wraps real geometry element but its children do that.
    GeometryTreeItem *rootItem;

public:

    friend class GeometryTreeItem;

    /**
     * Refresh all tree content.
     * @param document document from which new content will be read
     */
    void refresh(Document& document);

    /**
     * @param document document from which tree content will be read
     * @param parent
     */
    GeometryTreeModel(Document& document, QObject *parent = 0);

    /// Delete rootItem.
    ~GeometryTreeModel();

    /**
     * Get item from index.
     * @return item from index if index.isValid(), rootItem in another case
     */
    GeometryTreeItem* toItem(const QModelIndex &index) const {
        return index.isValid() ?
               static_cast<GeometryTreeItem*>(index.internalPointer()) :
               rootItem;
    }

    // ---------- implementation of QAbstractItemModel methods: --------

    QModelIndex index(int row, int column, const QModelIndex &parent = QModelIndex()) const;

    QModelIndex parent(const QModelIndex &index) const;

    int rowCount(const QModelIndex &parent = QModelIndex()) const;

    int columnCount(const QModelIndex &parent = QModelIndex()) const;

    QVariant data(const QModelIndex &index, int role) const;

    Qt::ItemFlags flags(const QModelIndex &index) const;

    QVariant headerData(int section, Qt::Orientation orientation, int role) const;

    bool removeRows(int position, int rows, const QModelIndex &parent = QModelIndex());

    bool insertRow(plask::shared_ptr<plask::GeometryElement> to_insert, const QModelIndex &parent = QModelIndex(), int position = 0);

    int insertRow2D(const GeometryElementCreator& to_insert, const QModelIndex &parent, const plask::Vec<2, double>& point);

    plask::Box2D insertPlace2D(const GeometryElementCreator& to_insert, const QModelIndex &parent, const plask::Vec<2, double>& point);
};

#endif // PLASK_GUI_TREE_H

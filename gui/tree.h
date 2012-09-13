#ifndef PLASK_GUI_TREE_H
#define PLASK_GUI_TREE_H

#include <QAbstractItemModel>
#include <QPixmap>
#include <plask/geometry/object.h>
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
 * Geometry tree item. Holds geometry object wrapper.
 */
class GeometryTreeItem {

protected:

    /// Pointer to tree model, used to call update.
    GeometryTreeModel *model;

    /// Children of this item.
    std::vector< std::unique_ptr<GeometryTreeItem> > childItems;

    /// Cache for miniature, valid when miniatureInitialized is @c true, see @ref ensureInitialized().
    QPixmap miniature;

    /**
     * True only if children of this item was initialized, see @ref ensureInitialized().
     */
    bool childrenInitialized;

    /**
     * True only if this item miniature was initialized, see @ref ensureInitialized().
     */
    bool miniatureInitialized;

    /**
     * Ensure that this item is initialized (initialize it if its not).
     *
     * This can require filling children list and construct a miniature.
     * Set both childrenInitialized and miniatureInitialized to @c true.
     */
    void ensureInitialized();

    /**
     * Append to childItems children items for an given object.
     * @param elem object for which children should be constructed, typically (but not always) same as wrapped object,
     *  sometimes same as child of wrapped object
     * @param reverse append children in reverse order
     */
    void appendChildrenItemsHelper(const plask::shared_ptr<plask::GeometryObject>& elem, bool reverse = false);

    /**
     * Append to childItems children items for wrapped object.
     */
    virtual void appendChildrenItems();

    /**
     * Clear children items list and call appendChildrenItems()
     */
    void constructChildrenItems();

    /**
     * Delete children cache and childrenInitialized to @c false. Invalidates all indexes in subtree.
     */
    void deinitializeChildren();

public:

    /**
     * Get indexes for all cached nodes in subtree with this in root.
     * @param[out] dst place to add indexes
     * @param[in] index of this in parent
     */
    void getExistsSubtreeIndexes(QModelIndexList& dst, std::size_t indexInParent);
    
    /**
     * Get indexes for all cached nodes in subtree with this in root.
     * @param[out] dst place to add indexes
     */
    void getExistsSubtreeIndexes(QModelIndexList& dst) { getExistsSubtreeIndexes(dst, indexInParent()); }

    virtual plask::shared_ptr<ObjectWrapper> getLowerWrappedObject() {
        return object;
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
     * Wrapped geometry object.
     */
    plask::shared_ptr<ObjectWrapper> object;

    /**
     * Get geometry object from parent item.
     * @return geometry object from parent item or plask::shared_ptr<plask::GeometryObject>()
     *  if can't get it (for example in case of root or parent doesn't wrap existing geometry object object).
     */
    plask::shared_ptr<ObjectWrapper> parent();

    /**
     * Construct item using parent item and index in it.
     * @param parentItem parent item which wrap existing geometry object
     * @param index (future) index of this in parent childItems
     */
    GeometryTreeItem(GeometryTreeItem* parentItem, std::size_t index);

    /**
     * Construct item.
     * @param parentItem parent item, can't be nullpre
     * @param object wrapped object
     * @param index (future) index of this in parent childItems
     */
    GeometryTreeItem(GeometryTreeItem* parentItem, const plask::shared_ptr<ObjectWrapper>& object/*, std::size_t index*/);

    /**
     * Construct root item (with nullptr as parentItem).
     * @param parentItem parent item, can't be nullpre
     * @param object wrapped object
     * @param index (future) index of this in parent childItems
     */
    GeometryTreeItem(GeometryTreeModel* model, const plask::shared_ptr<ObjectWrapper>& object);

    /**
     * Construct root item (with parentItem = nullptr).
     * @param rootObjects children of roots object (showing in tree as roots)
     * @param model model to notify about changes
     */
    GeometryTreeItem(const std::vector< plask::shared_ptr<plask::Geometry> >& rootObjects, GeometryTreeModel* model);

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
     * Get text representation of object wrapped by this.
     * @param object object wrapped by this
     * @return text representation of an @p object
     */
    virtual QString objectText(plask::shared_ptr<ObjectWrapper> object) const;

    /**
     * @return string returned by objectText or empty QVariant if this wraps non-existing object
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
     * Called when wrapped geometry object was changed.
     * @param evt information about event from model
     */
    void onChanged(const ObjectWrapper::Event& evt);

    /**
     * Connect onChanged method to el->changed.
     * @param el object, typically this->object.lock()
     */
    void connectOnChanged(const plask::shared_ptr<ObjectWrapper> &el);

    /**
     * Disconnect onChanged method from el->changed.
     * @param el object, typically this->object.lock()
     */
    void disconnectOnChanged(const plask::shared_ptr<ObjectWrapper>& el);

    //TODO new subclass for root item and reimplementation of this which remove from manager
    virtual bool removeRange(std::size_t begin_index, std::size_t end_index);

    /**
     * Remove given number of @p rows starting from given @p position.
     * @param position index of first child to remove
     * @param rows number of children to remove
     * @return @c true if remove something
     */
    bool remove(int position, int rows) { return removeRange(position, position + rows); }

    bool tryInsert(plask::shared_ptr<plask::GeometryObject> object, int index);

    bool tryInsert(const GeometryObjectCreator &object_creator, int index);

    int getInsertionIndexForPoint(const plask::Vec<2, double>& point);

    int tryInsertRow2D(const GeometryObjectCreator& to_insert, const plask::Vec<2, double>& point);

    plask::Box2D getInsertPlace2D(const GeometryObjectCreator& to_insert, const plask::Vec<2, double>& point);

};

/**
 * Wrap translation and child of this translation inside container
 * (this two objects are represented as one item in tree).
 */
struct InContainerTreeItem: public GeometryTreeItem {

    plask::shared_ptr<ObjectWrapper> lowerObject;

private:
    void initLowerObject() {
        std::size_t chCount = object->wrappedObject->getRealChildrenCount();
        if (chCount == 0) lowerObject = plask::shared_ptr<ObjectWrapper>();
        else {
            lowerObject = ext(object->wrappedObject->getRealChildAt(0));
            connectOnChanged(lowerObject);
        }
    }
public:

    virtual plask::shared_ptr<ObjectWrapper> getLowerWrappedObject() {
        return lowerObject;
    }

    /**
     * @param parentItem parent item, must wrap plask::Translation<2> or plask::Translation<3>
     * @param index (future) index of this in parent childItems
     */
    InContainerTreeItem(GeometryTreeItem* parentItem, std::size_t index)
        : GeometryTreeItem(parentItem, index) {
        initLowerObject();
    }

    InContainerTreeItem(GeometryTreeItem* parentItem, const plask::shared_ptr<ObjectWrapper>& object)
        : GeometryTreeItem(parentItem, object) {
        initLowerObject();
    }

    ~InContainerTreeItem() {
        disconnectOnChanged(lowerObject);
    }

    //virtual void appendChildrenItems();

    virtual QString objectText(plask::shared_ptr<ObjectWrapper> object) const;

    virtual void fillPropertyBrowser(BrowserWithManagers& browser);
};


class Document;

/**
 * Implementation of QAbstractItemModel which holds and use GeometryTreeItem.
 */
//TODO geometry 2d not inform that children list was changed if only child of extrusion changed
class GeometryTreeModel: public QAbstractItemModel {

    Q_OBJECT

    struct RootItem {
        GeometryTreeItem treeItem;
        plask::shared_ptr<plask::Geometry> geometry;

        RootItem(GeometryTreeModel* model, plask::shared_ptr<plask::Geometry> geometry/*, std::size_t index*/)
            : treeItem(model, ext(geometry)/*, index*/), geometry(geometry) {}
    };

    /// Root of tree, wraps geometries
    std::vector< std::unique_ptr<RootItem> > rootItems;

public:
    
    void appendGeometry(plask::shared_ptr<plask::Geometry> geometry);

    friend class GeometryTreeItem;

    /**
     * Refresh all tree content.
     * @param roots new roots objects
     */
    void refresh(const std::vector< plask::shared_ptr<plask::Geometry> >& roots);
    
    /**
     * Clear whole geometry tree.
     */
    void clear();

    /**
     * @param document document from which tree content will be read
     * @param parent
     */
    GeometryTreeModel(QObject *parent = 0);
    
    ~GeometryTreeModel();
    
    /**
     * Save geometry to file.
     * @param root_object parent (root) object
     */
    void save(plask::XMLWriter::Element& root_object);

    /**
     * Get item from index.
     * @return item from index if index.isValid(), rootItem in another case
     */
    static GeometryTreeItem* toItem(const QModelIndex &index) {
        return static_cast<GeometryTreeItem*>(index.internalPointer());
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

    bool insertRow(plask::shared_ptr<plask::GeometryObject> to_insert, const QModelIndex &parent = QModelIndex(), int position = 0);

    int insertRow2D(const GeometryObjectCreator& to_insert, const QModelIndex &parent, const plask::Vec<2, double>& point);

    plask::Box2D insertPlace2D(const GeometryObjectCreator& to_insert, const QModelIndex &parent, const plask::Vec<2, double>& point);

    bool dropMimeData(const QMimeData * data, Qt::DropAction action, int row, int column, const QModelIndex & parent);

    //virtual QMimeData *	mimeData ( const QModelIndexList & indexes ) const;

    QStringList mimeTypes() const;

    Qt::DropActions supportedDropActions() const;

};

#endif // PLASK_GUI_TREE_H

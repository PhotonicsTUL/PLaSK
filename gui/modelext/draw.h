#ifndef PLASK_GUI_MODEL_EXT_DRAW_H
#define PLASK_GUI_MODEL_EXT_DRAW_H

#include <plask/geometry/object.h>
#include <QGraphicsItem>

QT_BEGIN_NAMESPACE
class QPainter;
class QGraphicsItem;
QT_END_NAMESPACE

/**
 * Wrapper over plask::GeometryObjectD<2> which implement QGraphicsItem interface.
 */
class GeometryObjectItem: public QGraphicsItem {

    /// Wrapped object.
    plask::weak_ptr< plask::GeometryObjectD<2> > object;

    // Cached bounding box of object.
    QRectF boundingBox;

    /**
     * Inform scene that item was changed and should be redrawn.
     * @param e typically object.lock()
     * @param resized true if object @p e was resized (size of bounding box was changed)
     */
    void doUpdate(plask::shared_ptr< const plask::GeometryObjectD<2> > e, bool resized = true);

    /**
     * Inform scene that item was changed and should be redrawn.
     * @param resized true if object was resized (size of bounding box was changed)
     */
    void doUpdate(bool resized = true) { doUpdate(object.lock(), resized); }

    /**
     * Called when wrapped geometry object was changed.
     * @param evt information about event from model
     */
    void onObjectUpdate(const plask::GeometryObject::Event& evt);

    /**
     * Disconnect onChanged method from object changed signal.
     */
    void disconnectOnChanged();

public:

    /// Call disconnectOnChanged().
    ~GeometryObjectItem();

    /**
     * Set (wrapped) object to draw.
     * @param object new wrapped object
     */
    void setObject(const plask::shared_ptr< plask::GeometryObjectD<2> >& object);

    /**
     * Get wrapped object.
     * @return wrapped object
     */
    const plask::weak_ptr< plask::GeometryObjectD<2> >& getObject() const {
        return this->object;
    }

    /**
     * Construct GeometryObjectItem which wrap given @p object
     * @param object geometry object to wrap
     */
    GeometryObjectItem(const plask::shared_ptr< plask::GeometryObjectD<2> >& object) {
        setObject(object);
    }

    //---- QGraphicsItem methods implementation: -----

    QRectF boundingRect() const;

    void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget);

    int type() const;

};

#endif // PLASK_GUI_MODEL_EXT_DRAW_H

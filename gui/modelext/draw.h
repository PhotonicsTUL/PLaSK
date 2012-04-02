#ifndef PLASK_GUI_MODEL_EXT_DRAW_H
#define PLASK_GUI_MODEL_EXT_DRAW_H

#include <plask/geometry/element.h>
#include <QGraphicsItem>

QT_BEGIN_NAMESPACE
class QPainter;
class QGraphicsItem;
QT_END_NAMESPACE

/**
 * Wrapper over plask::GeometryElementD<2> which implement QGraphicsItem interface.
 */
class GeometryElementItem: public QGraphicsItem {

    /// Wrapped element.
    plask::weak_ptr< plask::GeometryElementD<2> > element;

    // Cached bounding box of element.
    QRectF boundingBox;

    /**
     * Inform scene that item was changed and should be redrawn.
     * @param e typically element.lock()
     * @param resized true if element @p e was resized (size of bounding box was changed)
     */
    void doUpdate(plask::shared_ptr< const plask::GeometryElementD<2> > e, bool resized = true);

    /**
     * Inform scene that item was changed and should be redrawn.
     * @param resized true if element was resized (size of bounding box was changed)
     */
    void doUpdate(bool resized = true) { doUpdate(element.lock(), resized); }

    /**
     * Called when wrapped geometry element was changed.
     * @param evt information about event from model
     */
    void onElementUpdate(const plask::GeometryElement::Event& evt);

    /**
     * Disconnect onChanged method from element changed signal.
     */
    void disconnectOnChanged();

public:

    /// Call disconnectOnChanged().
    ~GeometryElementItem();

    /**
     * Set (wrapped) element to draw.
     * @param element new wrapped element
     */
    void setElement(const plask::shared_ptr< plask::GeometryElementD<2> >& element);

    /**
     * Get wrapped element.
     * @return wrapped element
     */
    const plask::weak_ptr< plask::GeometryElementD<2> >& getElement() const {
        return this->element;
    }

    /**
     * Construct GeometryElementItem which wrap given @p element
     * @param element geometry element to wrap
     */
    GeometryElementItem(const plask::shared_ptr< plask::GeometryElementD<2> >& element) {
        setElement(element);
    }

    //---- QGraphicsItem methods implementation: -----

    QRectF boundingRect() const;

    void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget);

    int type() const;

};

#endif // PLASK_GUI_MODEL_EXT_DRAW_H

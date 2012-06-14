#ifndef GUI_GEOMETRY_WRAPPER_ELEMENT_H
#define GUI_GEOMETRY_WRAPPER_ELEMENT_H

/** @file
 * This file includes interface to plask geometry elements model extensions connected with Qt.
 *
 * Typically you should call ext() function with pointer to your geometry element object as argument.
 * This function return Element which provide methods which operates on your element, knowing its type.
 */

#include <QPainter>
#include <QGraphicsItem>
#include "../utils/propbrowser.h"
#include "../modelext/creator.h"

QT_BEGIN_NAMESPACE
class QPainter;
class QGraphicsItem;
class QRectF;
QT_END_NAMESPACE

#include <plask/geometry/element.h>

/**
 * Wrapper over plask::GeometryElement, which:
 * - has extra method used by Qt GUI,
 * - has extra data (like element name).
 *
 * Typically you should not create object of this class or subclasses of this directly,
 * but useing ext function (defined in register.h).
 */
struct ElementWrapper {

    typedef plask::GeometryElement WrappedType;

    plask::shared_ptr<plask::GeometryElement> wrappedElement;

    QString name;

    /// This is typically called once, just after constructor
    virtual void setWrappedElement(plask::shared_ptr<plask::GeometryElement> plaskElement) {
        if (this->wrappedElement) this->wrappedElement->changedDisconnectMethod(this, &ElementWrapper::onWrappedChange);
        this->wrappedElement = plaskElement;
        if (this->wrappedElement) this->wrappedElement->changedConnectMethod(this, &ElementWrapper::onWrappedChange);
    }

    /// Virtual destructor, diconnect from wrapped element.
    virtual ~ElementWrapper();

    /**
     * Store information about event connected with geometry element or its wrapper.
     *
     * Subclasses of this can includes additional information about specific type of event.
     */
    struct Event: public EventWithSourceAndFlags<ElementWrapper> {

        /// Event flags (which describes event properties).
        typedef plask::GeometryElement::Event::Flags Flags;

        /// Non-null if event is delegeted from wrapped element.
        const plask::GeometryElement::Event* delegatedEvent;

        plask::shared_ptr<plask::GeometryElement> wrappedElement() {
            return this->source().wrappedElement;
        }

        /**
         * Check if given @p flag is set.
         * @param flag flag to check
         * @return @c true only if @p flag is set
         */
        bool hasFlag(Flags flag) const { return hasAnyFlag(flag); }

        /**
         * Check if DELETE flag is set, which mean that source of event is deleted.
         * @return @c true only if DELETE flag is set
         */
        bool isDelete() const { return hasFlag(plask::GeometryElement::Event::DELETE); }

        /**
         * Check if RESIZE flag is set, which mean that source of event could be resized.
         * @return @c true only if RESIZE flag is set
         */
        bool isResize() const { return hasFlag(plask::GeometryElement::Event::RESIZE); }

        /**
         * Check if DELEGATED flag is set, which mean that source delegate event from its child.
         * @return @c true only if DELEGATED flag is set
         */
        bool isDelgatedFromChild() const { return hasFlag(plask::GeometryElement::Event::DELEGATED); }

        /**
         * Check if event is delegated from wrapped element.
         * @return @c true only if event is delegated from wrapped element
         */
        bool isDelgatedFromWrappedElement() const { return delegatedEvent != nullptr; }

        /**
         * Check if CHILD_LIST flag is set, which mean that children list of source could be changed.
         * @return @c true only if CHILD_LIST flag is set
         */
        bool hasChangedChildrenList() const { return hasFlag(plask::GeometryElement::Event::CHILD_LIST); }

        /**
         * Construct event.
         * @param source source of event
         * @param flags which describes event's properties
         */
        explicit Event(ElementWrapper& source, unsigned char flags = 0):
            EventWithSourceAndFlags<ElementWrapper>(source, flags), delegatedEvent(0) {}

        /**
         * Construct event which delegete event from wrapped element.
         * @param source source of event
         * @param evt event generated by wrapped element
         */
        Event(ElementWrapper& source, const plask::GeometryElement::Event& evt)
            : EventWithSourceAndFlags<ElementWrapper>(source, evt.flags()), delegatedEvent(&evt) {}
    };

    /// Changed signal, fired when element was changed.
    boost::signals2::signal<void(const Event&)> changed;

    /// Connect a method to changed signal
    template <typename ClassT, typename methodT>
    boost::signals2::connection changedConnectMethod(ClassT* obj, methodT method) {
        return changed.connect(boost::bind(method, obj, _1));
    }

    /// Disconnect a method from changed signal
    template <typename ClassT, typename methodT>
    void changedDisconnectMethod(ClassT* obj, methodT method) {
        changed.disconnect(boost::bind(method, obj, _1));
    }

    /**
     * Call changed with this as event source.
     * @param event_constructor_params_without_source parameters for event constructor (without first - source)
     */
    template<typename EventT = Event, typename ...Args>
    void fireChanged(Args&&... event_constructor_params_without_source) {
        changed(EventT(*this, std::forward<Args>(event_constructor_params_without_source)...));
    }

protected:
    void onWrappedChange(const plask::GeometryElement::Event& evt) {
        fireChanged(evt);
    }

public:

    /**
     * Set new name and inform observers about this.
     * @param new_name new name
     */
    void setName(const QString& new_name) {
        name = new_name;
        fireChanged();
    }

    QString& getName() {
        return name;
    }

    const QString& getName() const {
        return name;
    }

    /**
     * Draw geometry element using given Qt @p painter.
     * @param painter where draw element
     */
    virtual void draw(QPainter& painter) const;

    /**
     * Draw miniature with size close to given.
     * @param painter where miniature shpuld be drawn
     * @param w, h requested miniature size
     */
    virtual void drawMiniature(QPainter& painter, qreal w, qreal h) const;

    /**
     * Draw real part (real children only) of geometry element using given Qt @p painter.
     * @param painter where draw element
     */
    virtual void drawReal(QPainter& painter) const;

    /**
     * Get miniature image with size close to given.
     * @param w, h requested miniature size
     * @return miniature
     */
    QPixmap getMiniature(qreal w, qreal h) const;

    /**
     * Get string representation of given element.
     * @return string representation of wrapped element, can have multiple lines of text
     */
    virtual QString toStr() const;

    /**
     * Fill property browser with properties of wrapped element.
     */
    virtual void setupPropertiesBrowser(BrowserWithManagers& managers, QtAbstractPropertyBrowser& dst);

    void setupPropertiesBrowser(BrowserWithManagers& managers) {
        setupPropertiesBrowser(managers, managers.browser);
    }

    /**
     * Fill property browser with properties of @p container child.
     *
     * This is called only for containers and default implementation call setupPropertiesBrowser for pointed child.
     * Typically, you can call ElementExtensionImplBase::setupPropertiesBrowserForChild in subclasses.
     * @param index real child index
     */
    virtual void setupPropertiesBrowserForChild(std::size_t index, BrowserWithManagers& managers, QtAbstractPropertyBrowser& dst);

    void setupPropertiesBrowserForChild(std::size_t index, BrowserWithManagers& managers) {
        setupPropertiesBrowserForChild(index, managers, managers.browser);
    }

    /**
     * Check if @p to_insert can be insert to this at position @p index.
     *
     * Type of @p to_insert and possible loops are checked. Also @p index is checked.
     * @return @c true if @p to_insert can be insert to this at position @p index
     */
    virtual bool canInsert(plask::shared_ptr<plask::GeometryElement> to_insert, std::size_t index) const {
        return false;
    }

    /**
     * Insert @p to_insert at postion @p index to this.
     * @return @c true if @p to_insert can be insert to this at position @p index
     */
    virtual bool tryInsert(plask::shared_ptr<plask::GeometryElement> to_insert, std::size_t index) {
        return false;
    }

    /**
     * Get child creators for thie element.
     *
     * Default implementation returns empty vector, but sublasses redefine this.
     * @return vector of creators of child of this
     */
    std::vector<const GeometryElementCreator*> getChildCreators() const {
        return std::vector<const GeometryElementCreator*>();
    }

};

template <typename WrappedT, typename BaseClass = ElementWrapper>
struct ElementWrapperFor: public BaseClass {

    typedef WrappedT WrappedType;

    WrappedType& c() const { return static_cast<WrappedType&>(*this->wrappedElement); }

};

#endif // GUI_GEOMETRY_WRAPPER_ELEMENT_H

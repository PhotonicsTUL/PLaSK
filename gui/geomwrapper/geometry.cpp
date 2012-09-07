#include "geometry.h"

#include "register.h"

template <int dim>
QString GeometryWrapper<dim>::toStr() const {
    return QString(QObject::tr("geometry%1d%2"))
        .arg(dim)
            .arg(this->name.isEmpty() ? "" : (" \"" + this->name + "\""));
}

template <int dim>
void GeometryWrapper<dim>::setupPropertiesBrowser(BrowserWithManagers &managers, QtAbstractPropertyBrowser &dst) {
    QtProperty *nameProp = managers.string.addProperty("geometry name");
    managers.string.setValue(nameProp, this->getNameQt());
    dst.addProperty(nameProp);
    managers.connectString(nameProp, [this](const QString& v) {
        this->setName(v);
    });
    auto el3D = this->c().getElement3D();
    if (el3D) ext(el3D)->setupPropertiesBrowser(managers, dst);
}

template <int dim>
void GeometryWrapper<dim>::draw(QPainter &painter) const {
    auto child = this->c().getChild();
    if (child) ext(child)->draw(painter);
}

template <int dim>
void GeometryWrapper<dim>::drawMiniature(QPainter &painter, qreal w, qreal h) const
{
    auto child = this->c().getChild();
    if (child) ext(child)->drawMiniature(painter, w, h);
}

template <int dim>
void GeometryWrapper<dim>::drawReal(QPainter &painter) const
{
    auto child = this->c().getChild();
    if (child) ext(child)->drawReal(painter);
}

template struct GeometryWrapper<2>;
template struct GeometryWrapper<3>;

QString Geometry2DCartesianWrapper::toStr() const {
    QString res = QString(QObject::tr("Cartesian geometry 2d%1")).arg(this->name.isEmpty() ? "" : ("\n\"" + this->name + "\""));
    if (c().getElement3D()) {
        auto ch = ext(c().getElement3D());
        if (!ch->name.isEmpty()) {
            res += '\n';
            res += QString(QObject::tr("Extrusion name \"%1\"")).arg(ch->name);
        }
    }
    return res;
}

plask::shared_ptr<plask::Extrusion> Geometry2DCartesianWrapper::getExtrusion() const {
    return plask::static_pointer_cast<plask::Extrusion>(c().getElement3D());
}

plask::Geometry2DCartesian &Geometry2DCartesianWrapper::getCartesian2D() const
{
    return static_cast<plask::Geometry2DCartesian&>(c());
}

bool Geometry2DCartesianWrapper::canInsert(plask::shared_ptr<plask::GeometryElement> to_insert, std::size_t index) const {
    return index == 0 && c().getElement3D()->getRealChildrenCount() == 0 &&
            (to_insert->getDimensionsCount() == 2 || plask::dynamic_pointer_cast<plask::Extrusion>(to_insert));
}

bool Geometry2DCartesianWrapper::canInsert(const GeometryElementCreator &to_insert, std::size_t index) const  {
    if (index != 0) return false;
    if (to_insert.supportDimensionsCount(2)) return true;
    return plask::dynamic_pointer_cast<plask::Extrusion>(to_insert.getElement(3));  //if is 3d must allow to create extrusion
}

bool Geometry2DCartesianWrapper::tryInsert(plask::shared_ptr<plask::GeometryElement> to_insert, std::size_t index) {
    if (!canInsert(to_insert, index)) return false;
    if (to_insert->getDimensionsCount() == 2) {
        getExtrusion()->setChild(to_insert->asD<2>());
    } else {
        getCartesian2D().setExtrusion(plask::static_pointer_cast<plask::Extrusion>(to_insert));
    }
    return true;
}

bool Geometry2DCartesianWrapper::tryInsert(const GeometryElementCreator& to_insert, std::size_t index) {
    if (to_insert.supportDimensionsCount(2))
        return tryInsert(to_insert.getElement(2), index);
    else
        return tryInsert(to_insert.getElement(3), index);
}



QString Geometry2DCylindricalWrapper::toStr() const
{
    QString res = QString(QObject::tr("Cylindrical geometry 2d%1")).arg(this->name.isEmpty() ? "" : ("\n\"" + this->name + "\""));
    if (c().getElement3D()) {
        auto ch = ext(c().getElement3D());
        if (!ch->name.isEmpty()) {
            res += '\n';
            res += QString(QObject::tr("Revolution name \"%1\"")).arg(ch->name);
        }
    }
    return res;
}

plask::shared_ptr<plask::Revolution> Geometry2DCylindricalWrapper::getRevolution() const {
    return plask::static_pointer_cast<plask::Revolution>(c().getElement3D());
}

plask::Geometry2DCylindrical &Geometry2DCylindricalWrapper::getCylindrical2D() const
{
    return static_cast<plask::Geometry2DCylindrical&>(c());
}

bool Geometry2DCylindricalWrapper::canInsert(plask::shared_ptr<plask::GeometryElement> to_insert, std::size_t index) const {
    return index == 0 && c().getElement3D()->getRealChildrenCount() == 0 &&
            (to_insert->getDimensionsCount() == 2 || plask::dynamic_pointer_cast<plask::Revolution>(to_insert));
}

bool Geometry2DCylindricalWrapper::canInsert(const GeometryElementCreator &to_insert, std::size_t index) const  {
    if (index != 0) return false;
    if (to_insert.supportDimensionsCount(2)) return true;
    return plask::dynamic_pointer_cast<plask::Revolution>(to_insert.getElement(3));  //if is 3d must allow to create extrusion
}

bool Geometry2DCylindricalWrapper::tryInsert(plask::shared_ptr<plask::GeometryElement> to_insert, std::size_t index) {
    if (!canInsert(to_insert, index)) return false;
    if (to_insert->getDimensionsCount() == 2) {
        getRevolution()->setChild(to_insert->asD<2>());
    } else {
        getCylindrical2D().setRevolution(plask::static_pointer_cast<plask::Revolution>(to_insert));
    }
    return true;
}

bool Geometry2DCylindricalWrapper::tryInsert(const GeometryElementCreator& to_insert, std::size_t index) {
    if (to_insert.supportDimensionsCount(2))
        return tryInsert(to_insert.getElement(2), index);
    else
        return tryInsert(to_insert.getElement(3), index);
}

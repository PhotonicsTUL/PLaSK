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
    managers.string.setValue(nameProp, this->getName());
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

template class GeometryWrapper<2>;
template class GeometryWrapper<3>;

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

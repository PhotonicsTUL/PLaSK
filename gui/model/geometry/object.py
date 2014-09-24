from ...utils.xml import AttributeReader, OrderedTagReader

class GeometryObjectModel(object):

    def __init__(self, name = None):
        super(GeometryObjectModel, self).__init__()
        self.name = name

    def attributes_from_XML(self, attribute_reader):
        """
        :param AttributeReader attribute_reader: source of attributes
        :return:
        """
        self.name = attribute_reader.get('name', None)

    def children_from_XML(self, ordered_reader):
        pass

    def from_XML(self, element):
        with AttributeReader(element) as a: self.attributes_from_XML(a)
        with OrderedTagReader(element) as r: self.children_from_XML(r)

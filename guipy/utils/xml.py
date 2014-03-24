from lxml import etree as ElementTree

# default, used XML parser
XML_parser = ElementTree.XMLParser(remove_blank_text = True, remove_comments = False, strip_cdata = False)

def print_interior(element):
    """Print all subnodes of element (all except the element's opening and closing tags)"""
    text = element.text.lstrip('\n') if element.text else ''
    for c in element:
        text += ElementTree.tostring(c, pretty_print = True)
    return text


class AttributeReader(object):
    """
        Helper class to check if all attributes have been read from XML tag, usage::
        
            with AttributeReader(xml_element) as a:
                # use a.get(...) or a[...] to access to xml_element.attrib
    """
    
    def __init__(self, element):
        super(AttributeReader, self).__init__()
        self.element = element
        self.attrib = element.attrib
        self.read = set()
        
    def get(self, key, default = None):
        self.read.add(key)
        return self.attrib.get(key, default)
    
    def __len__(self):
        return len(self.attrib)
        
    def __getitem__(self, key):
        self.read.add(key)
        return self.attrib[key]
    
    def mark_read(self, *keys):
        for k in keys: self.read.add(k)
        
    def require_all_read(self):
        """Raise ValueError if not all attributes have been read from XML tag."""
        not_read = set(self.attrib.keys) - self.read
        if not_read: raise ValueError("XML tag <%s> has unexpected attributes: %s", (self.element.tag, ", ".join(not_read)))

    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_value, traceback):
        """It raise ValueError if any other exception haven't been raised and not all attributes have been read from XML tag."""
        if exc_type == None and exc_value == None and traceback == None: self.require_all_read()
from lxml import etree as ElementTree

# default, used XML parser
#XML_parser = ElementTree.XMLParser(remove_blank_text=True, remove_comments=False, strip_cdata=False)
#TODO remove_comments set to False when all will be ready to support it
XML_parser = ElementTree.XMLParser(remove_blank_text=True, remove_comments=True, strip_cdata=False)


def print_interior(element):
    """Print all subnodes of element (all except the element's opening and closing tags)"""
    text = element.text.lstrip('\n') if element.text else ''
    for c in element:
        text += ElementTree.tostring(c, pretty_print=True)
    return text


class AttributeReader(object):
    """
        Helper class to check if all attributes have been read from XML tag, usage::
        
            with AttributeReader(xml_element) as a:
                # use a.get(...) or a[...] to access to xml_element.attrib
    """
    
    def __init__(self, element):
        super(AttributeReader, self).__init__()
        if isinstance(element, AttributeReader):
            self.element = element.element
            self.read = element.read
            self.is_sub_reader = True
        else:
            self.element = element
            self.read = set()
            self.is_sub_reader = False
        
    def get(self, key, default=None):
        self.read.add(key)
        return self.element.attrib.get(key, default)
    
    def __len__(self):
        return len(self.element.attrib)
        
    def __getitem__(self, key):
        self.read.add(key)
        return self.element.attrib[key]

    def __contains__(self, key):
        self.read.add(key)
        return key in self.element.attrib
    
    def mark_read(self, *keys):
        for k in keys: self.read.add(k)
        
    def require_all_read(self):
        """Raise ValueError if not all attributes have been read from XML tag."""
        if self.is_sub_reader: return
        not_read = set(self.element.attrib.keys()) - self.read
        if not_read:
            raise ValueError("XML tag <{}> has unexpected attributes: {}".format(self.element.tag, ", ".join(not_read)))

    @property
    def attrib(self):
        """Thanks to this property, AttributeReader can pretend to be ElementTree.Element in many contexts."""
        return self

    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_value, traceback):
        """
            It raise ValueError if any other exception haven't been raised
            and not all attributes have been read from XML tag.
        """
        if exc_type is None and exc_value is None and traceback is None: self.require_all_read()


class OrderedTagReader(object):
    """Helper class to read children of XML element in required order.
       It checks if all children has been read.

       Usage::

         with OrderedTagReader(parent_element) as r:
                # use r.get(...) or r.require(...) to access children of parent_element
    """

    def __init__(self, parent_element, first_index = 0):
        super(OrderedTagReader, self).__init__()
        self.parent_element = parent_element
        self.current_index = first_index

    def _next_element(self):
        """:return: next element"""
        return self.parent_element[self.current_index]

    def _has_next(self):
        """:return: True if there is next element, False in other cases."""
        return self.current_index != len(self.parent_element)

    def _goto_next(self):
        """
            Increment current_index if there is next element.
            :return: Next element or None if there is no such element
        """
        if not self._has_next(): return None
        res = self._next_element()
        self.current_index += 1
        return res

    def require_end(self):
        """Raise ValueError if self.parent_element still has unread children."""
        if self._has_next():
            raise ValueError("XML tag <{}> has unexpected child <{}>.".format(
                self.parent_element.tag, self._next_element().tag))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
            It raise ValueError if any other exception haven't been raised
            and self.parent_element still has unread children.
        """
        if exc_type is None and exc_value is None and traceback is None: self.require_end()

    def get(self, expected_tag_name = None):
        """
            Get next child of wrapped self.parent_element.
            :param str expected_tag_name: optional required name of returned tag
            :return: Next child of wrapped self.parent_element or None if there is no more child or next child has name
                    other than expected_tag_name
        """
        if expected_tag_name is None:
            return self._goto_next()
        else:
            if self._has_next() and self._next_element().tag == expected_tag_name:
                return self._goto_next()
            else:
                return None

    def require(self, expected_tag_name = None):
        """
            Get next child of wrapped self.parent_element or raise ValueError if
                there is no more child or next child has name other than expected_tag_name.
            :param str expected_tag_name: optional required name of returned tag
            :return: Next child of wrapped self.parent_element.
        """
        res = self.get(expected_tag_name)
        if res is None:
            if expected_tag_name is None:
                raise ValueError('Unexpected end of <{}> tag.'.format(self.parent_element.tag))
            else:
                raise ValueError('<{}> tag does not have required <{}> child.'.format(
                    self.parent_element.tag, expected_tag_name))
        return res

    def iter(self, expected_tag_name = None):
        """
            Iterator over the rest children.
            :param str expected_tag_name: optional required name of returned tags
            :return: yield the same as get(expected_tag_name) as long as this is not None
        """
        res = self.get(expected_tag_name)
        while res is not None:
            yield res
            res = self.get(expected_tag_name)


class UnorderedTagReader(object):
    """Helper class to read children of XML element, if the children can be in any order.
       Two or more children with the same name are not allowed.
       It checks if all children has been read.

       Usage::

         with UnorderedTagReader(parent_element) as r:
                # use r.get(...) or r.require(...) to access children of parent_element
    """

    def __init__(self, parent_element, first_index = 0):
        super(UnorderedTagReader, self).__init__()
        self.parent_element = parent_element
        self.read = set()
        tag_names = set()
        for child in parent_element:
            if child.tag in tag_names:
                raise ValueError("Duplicated tags <{}> in <{}> are not allowed.".format(child.tag, parent_element.tag))
            tag_names.add(child.tag)

    def get(self, child_name):
        """
            Get child of wrapped self.parent_element.
            :param str child_name: expected name of returned tag
            :return: child of wrapped self.parent_element with name child_name
                        or None if there is no child with expected name
        """
        res = self.parent_element.find(child_name)
        if res is not None: self.read.add(child_name)
        return res

    find = get  #alias for compatibility with unwrapped ElementTree.Element

    def require(self, child_name):
        """
            Get child of wrapped self.parent_element or raise ValueError if there is no child with expected name.
            :param str child_name: expected name of returned tag
            :return: child of wrapped self.parent_element with name child_name
        """
        res = self.get(child_name)
        if res is None:
            raise ValueError('<{}> tag does not have required <{}> child.'.format(
                    self.parent_element.tag, child_name))
        return res

    def __len__(self):
        return len(self.parent_element)

    def mark_read(self, *children_names):
        for k in children_names: self.read.add(k)

    def require_all_read(self):
        """Raise ValueError if not all children have been read from XML tag."""
        not_read = set(e.tag for e in self.parent_element) - self.read
        if not_read:
            raise ValueError("XML tag <{}> has unexpected child(ren): {}".format(self.parent_element.tag, ", ".join(not_read)))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
            It raise ValueError if any other exception haven't been raised
            and not all attributes have been read from XML tag.
        """
        if exc_type is None and exc_value is None and traceback is None: self.require_all_read()



def require_no_children(element):
    """Check if there are no children in element, raise error if there is any child."""
    with OrderedTagReader(element) as ensure_no_child_reader: pass

def require_no_attributes(element):
    """Check if there are no attributes in element, raise error if there is any attribute."""
    with AttributeReader(element) as ensure_no_attrib: pass
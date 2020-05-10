# Copyright (C) 2014 Photonics Group, Lodz University of Technology
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of GNU General Public License as published by the
# Free Software Foundation; either version 2 of the license, or (at your
# opinion) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

from lxml import etree

from .config import CONFIG

XMLparser = etree.XMLParser(remove_blank_text=True, strip_cdata=False,
                            remove_comments=not CONFIG['experimental/preserve_comments'])

class Element:

    def __init__(self, element, comments=None):
        self.__dict__['_element'] = element
        if comments is None:
            self.__dict__['comments'] = []
        else:
            self.__dict__['comments'] = comments

    def __getattr__(self, item):
        return getattr(self._element, item)

    def __setattr__(self, key, value):
        setattr(self._element, key, value)

    def __len__(self):
        return len(self._element)

    def __iter__(self):
        return iter(self._element)

    def __getitem__(self, item):
        return self._element[item]

    def __setitem__(self, key, value):
        self._element[key] = value


def print_interior(element):
    """Print all subnodes of element (all except the element's opening and closing tags)"""
    text = element.text.lstrip('\n') if element.text else ''
    for c in element:
        text += etree.tostring(c, pretty_print=True, encoding='unicode')
    return text


def attr_to_xml(src_obj, dst_element, *attr_names, **defaults):
    """
        Set XML attributes in dst_element.attrib using attributes from src_obj.
        Only existing, not-None attributes are set.
        :param src_obj: source object
        :param dst_element: destination element
        :param str *attr_names: names of attributes to transfer
    """
    for attr in attr_names:
        a = getattr(src_obj, attr, None)
        if a is not None: dst_element.attrib[attr] = a
        elif attr in defaults: dst_element.attrib[attr] = str(defaults[attr])


def xml_to_attr(src, dst_obj, *attr_names):
    """
        Set dst_obj attributes using data from src_element attributes.
        All attributes not included in src are set to None.
        If src is None all attributes are set to None.
        :param src: elementtree element or AttributeReader
        :param dst_obj: destination object
        :param *attr_names: names of attributes to transfer
    """
    if src is None:
        for attr in attr_names: setattr(dst_obj, attr, None)
    else:
        with AttributeReader(src) as a:
             for attr in attr_names:
                 setattr(dst_obj, attr, a.get(attr, None))


def at_line_str(element, template=' at line {}'):
    return template.format(element.sourceline) if element.sourceline is not None else ''


def get_text(element):
    """
    Get text from XML element
    :param element: XML element
    :return: read texr
    """
    text = element.text
    for comment in element:
        if comment.tag is not etree.Comment:
            raise ValueError("Only text allowed in <{}>".format(element.tag))
        if comment.tail is not None:
            if text is None:
                text = comment.tail
            else:
                text += comment.tail
    return text


class AttributeReader:
    """
        Helper class to check if all attributes have been read from XML tag, usage::

            with AttributeReader(xml_element) as a:
                # use a.get(...) or a[...] to access to xml_element.attrib
    """
    
    def __init__(self, element):
        """
            :param element: elementtree Element or AttributeReader (in such case self.is_sub_reader is set to True and set of attributes read are shared)
        """
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

    def require(self, key):
        res = self.get(key)
        if res is None:
            raise KeyError('Attribute "{}" is expected in tag <{}>{}.'.format(key, self.element.tag, at_line_str(self.element)))
        return res
    
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
        """Raise ValueError if not all attributes have been read from XML tag. Do nothing if self.is_sub_reader is True."""
        if self.is_sub_reader: return
        not_read = set(self.element.attrib.keys()) - self.read
        if not_read:
            raise ValueError("XML tag <{}>{} has unexpected attributes: {}".format(
                self.element.tag, at_line_str(self.element), ", ".join(not_read)))

    @property
    def attrib(self):
        """Thanks to this property, AttributeReader can pretend to be etree.Element in many contexts."""
        return self

    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_value, traceback):
        """
            It raises ValueError if any other exception haven't been raised
            and not all attributes have been read from XML tag.
        """
        if exc_type is None and exc_value is None and traceback is None:
            self.require_all_read()

    def __iter__(self):
        return self.element.attrib.__iter__()


class OrderedTagReader:
    """Helper class to read children of XML element in required order.
       It checks if all children has been read and optionally (by default) ignores comments.

       Usage::

         with OrderedTagReader(parent_element) as r:
                # use r.get(...) or r.require(...) to access children of parent_element
    """

    def __init__(self, parent_element, first_index=0):
        # super(OrderedTagReader, self).__init__()
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
            Increment _current_index if there is next element.
            :return: Next element or None if there is no such element
        """
        if not self._has_next(): return None
        res = self._next_element()
        self.current_index += 1
        return res

    def require_end(self):
        """Raise ValueError if self.parent_element still has unread children."""
        if self._has_next():
            raise ValueError("XML tag <{}>{} has unexpected child <{}>{}.".format(
                self.parent_element.tag, at_line_str(self.parent_element),
                self._next_element().tag, at_line_str(self._next_element())))

    def recent_was_unexpected(self):
        """Raise ValueError about tag that has been read recently and move reader one tag back."""
        self._revert()
        self.require_end()  # raise exception

    def _revert(self):
        self.current_index -= 1
        while self.current_index >= 0 and self._next_element().tag is etree.Comment:
            self.current_index -= 1

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
            It raises ValueError if any other exception haven't been raised
            and self.parent_element still has unread children.
        """
        if exc_type is None and exc_value is None and traceback is None:
            self.require_end()

    @staticmethod
    def _expected_tag_to_str(*expected_tag_names):
        return ', '.join('<{}>'.format(s) for s in expected_tag_names)

    def _iter_comments(self):
        """
        Iterate over comments until the next element
        :return:
        """
        while self._has_next() and self._next_element().tag is etree.Comment:
            text = self._next_element().text
            self._goto_next()
            yield text


    def get_comments(self):
        """
        Get list of comments until the next element
        :return:
        """
        return list(self._iter_comments())

    def get(self, *expected_tag_names):
        """
            Get next child of wrapped self.parent_element.
            :param expected_tag_names: optional required names of returned tag
            :return: Next child of wrapped self.parent_element or None if there is no more child or next child has name
                    other than expected_tag_name
        """
        comments = self.get_comments()
        if len(expected_tag_names) == 0:
            res = self._goto_next()
        else:
            if self._has_next() and self._next_element().tag in expected_tag_names:
                res = self._goto_next()
            else:
                res = None
        if res is not None:
            res = Element(res, comments)
        else:
            self._revert()
            self.current_index += 1
        return res

    def require(self, *expected_tag_names):
        """
            Get next child of wrapped self.parent_element or raise ValueError if
                there is no more child or next child has name other than expected_tag_name.
            :param expected_tag_names: optional required name of returned tag
            :return: Next child of wrapped self.parent_element.
        """
        res = self.get(*expected_tag_names)
        if res is None:
            if len(expected_tag_names) == 0:
                raise ValueError('Unexpected end of <{}> tag{}.'.format(
                    self.parent_element.tag, at_line_str(self.parent_element, ' (which is opened at line {})')))
            else:
                raise ValueError('<{}> tag{} does not have required {} child.'.format(
                    self.parent_element.tag, at_line_str(self.parent_element),
                    OrderedTagReader._expected_tag_to_str(*expected_tag_names)))
        return res

    def iter(self, *expected_tag_names):
        """
            Iterator over the rest children.
            :param expected_tag_names: optional required name of returned tags
            :return: yield the same as get(expected_tag_name) as long as this is not None
        """
        res = self.get(*expected_tag_names)
        while res is not None:
            yield res
            res = self.get(*expected_tag_names)

    def __iter__(self):
        return self.iter()

class UnorderedTagReader:
    """Helper class to read children of XML element, if the children can be in any order.
       Two or more children with the same name are not allowed.
       It checks if all children has been read.

       Usage::

         with UnorderedTagReader(parent_element) as r:
                # use r.get(...) or r.require(...) to access children of parent_element
    """

    def __init__(self, parent_element):
        # super(UnorderedTagReader, self).__init__()
        self.parent_element = parent_element
        self.read = set()
        tag_names = set()
        for child in parent_element:
            if child.tag is etree.Comment: continue
            if child.tag in tag_names:
                raise ValueError("Duplicated tags <{}>{} in <{}>{} are not allowed.".format(
                    child.tag, at_line_str(self.child, ' (recurrence at line {})'),
                    parent_element.tag, at_line_str(self.parent_element, ' (which is opened at line {})')))
            tag_names.add(child.tag)

    def _iter_comments(self, element=None, inside=False):
        """
        Iterate comments preceding the specified element or the comments after the last element
        """
        if element is None:
            if len(self.parent_element) == 0:
                return
            element = self.parent_element[-1]
            if element.tag is not etree.Comment:
                return
        elif inside:
            if len(element) == 0:
                return
            element = element[-1]
        else:
            assert element.tag is not etree.Comment
        prev = element.getprevious()
        while prev is not None and prev.tag is etree.Comment:
            element = prev
            prev = element.getprevious()
        while element is not None and element.tag is etree.Comment:
            yield element.text
            element = element.getnext()

    def get_comments(self, element=None, inside=False):
        """
        Get comments preceding the specified element or the comments after the last element
        """
        return list(self._iter_comments(element, inside))

    def get(self, child_name):
        """
            Get child of wrapped self.parent_element.
            :param str child_name: expected name of returned tag
            :return: child of wrapped self.parent_element with name child_name
                        or None if there is no child with expected name
        """
        res = self.parent_element.find(child_name)
        if res is not None:
            self.read.add(child_name)
            res = Element(res, self.get_comments(res))
        return res

    find = get  # alias for compatibility with unwrapped etree.Element

    def require(self, child_name):
        """
            Get child of wrapped self.parent_element or raise ValueError if there is no child with expected name.
            :param str child_name: expected name of returned tag
            :return: child of wrapped self.parent_element with name child_name
        """
        res = self.get(child_name)
        if res is None:
            raise KeyError('<{}> tag{} does not have required <{}> child.'.format(
                    self.parent_element.tag, at_line_str(self.parent_element), child_name))
        return res

    def __len__(self):
        return len(self.parent_element)

    def mark_read(self, *children_names):
        for k in children_names: self.read.add(k)

    def require_all_read(self):
        """Raise ValueError if not all children have been read from XML tag."""
        not_read = set(e.tag for e in self.parent_element if e.tag is not etree.Comment) - self.read
        if not_read:
            raise ValueError("XML tag <{}>{} has unexpected child(ren):\n {}".format(
                self.parent_element.tag, at_line_str(self.parent_element), ", ".join(not_read)))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
            It raise ValueError if any other exception haven't been raised
            and not all attributes have been read from XML tag.
        """
        if exc_type is None and exc_value is None and traceback is None:
            self.require_all_read()



def require_no_children(element):
    """Check if there are no children in element, raise error if there is any child."""
    with OrderedTagReader(element) as _: pass

def require_no_attributes(element):
    """Check if there are no attributes in element, raise error if there is any attribute."""
    with AttributeReader(element) as _: pass


def elements_equal(e1, e2):
    """
        Check if two XML etree elements are equal.
        :return: True only if e1 and e2 are equal
    """
    return e1.tag == e2.tag and e1.text == e2.text and e1.tail == e2.tail and e1.attrib == e2.attrib and len(e1) == len(e2) and all(elements_equal(c1, c2) for c1, c2 in zip(e1, e2))
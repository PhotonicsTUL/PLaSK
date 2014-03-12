from lxml.etree import ElementTree

def print_interior(element):
    """Print all subnodes of element (all except the element's opening and closing tags)"""
    text = element.text.lstrip('\n') if element.text else ''
    for c in element:
        text += ElementTree.tostring(c, pretty_print = True)
    return text
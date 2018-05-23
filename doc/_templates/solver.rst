`{{ objname }}` Class
===============================================================================

.. currentmodule:: {{ module }}

.. autoclass:: {{ fullname }}

{%- block classes %}
{%- if classes and classes != ['dtype'] %}

Subclasses
----------

.. autosummary::
   :nosignatures:
   :toctree: {{ objname }}
   :template: class.rst
{% for item in classes %}
   ~{{ module }}.{{ objname }}.{{ item }}
{%- endfor %}

{%- endif %}
{%- endblock %}

{%- block methods %}
{%- if methods %}

Methods
-------

.. autosummary::
{% for item in methods %}
{%- if not item.startswith('_') and item not in ['load_xpl', 'on_initialize', 'on_invalidate'] %}
   ~{{ fullname }}.{{ item }}
{%- endif %}
{%- endfor %}

{%- endif %}
{%- endblock %}

{% block attributes -%}
{% if attributes -%}

Attributes
----------

Receivers
^^^^^^^^^

.. autosummary::
{% for item in attributes %}
{%- if item.startswith('in') and (item+'x')[2].isupper() %}
   ~{{ fullname }}.{{ item }}
{%- endif %}
{%- endfor %}

Providers
^^^^^^^^^

.. autosummary::
{% for item in attributes %}
{%- if item.startswith('out') and (item+'x')[3].isupper() %}
   ~{{ fullname }}.{{ item }}
{%- endif %}
{%- endfor %}

Other
^^^^^

.. autosummary::
{% for item in attributes %}
{%- if not (item.startswith('in') and (item+'x')[2].isupper()) and not (item.startswith('out') and (item+'x')[3].isupper()) %}
   ~{{ fullname }}.{{ item }}
{%- endif %}
{%- endfor %}

{%- endif %}
{%- endblock %}

Descriptions
------------

{%- block methods_desc %}
{%- if methods %}

.. rubric:: Method Details

{%- for item in methods %}
{%- if not item.startswith('_') and item not in ['load_xpl', 'on_initialize', 'on_invalidate']  %}

.. automethod:: {{ fullname }}.{{ item }}

{%- endif %}
{%- endfor %}

{%- endif %}
{%- endblock %}

{%- block attributes_desc %}
{%- if attributes %}

.. rubric:: Receiver Details

{%- for item in attributes %}
{%- if item.startswith('in') and (item+'x')[2].isupper() %}

.. autoattribute:: {{ fullname }}.{{ item }}

{%- endif %}
{%- endfor %}

.. rubric:: Provider Details

{%- for item in attributes %}
{%- if item.startswith('out') and (item+'x')[3].isupper() %}

.. autoattribute:: {{ fullname }}.{{ item }}
   :show-signature:

{%- endif %}
{%- endfor %}

.. rubric:: Attribute Details

{%- for item in attributes %}
{%- if not (item.startswith('in') and (item+'x')[2].isupper()) and not (item.startswith('out') and (item+'x')[3].isupper()) %}

.. autoattribute:: {{ fullname }}.{{ item }}

{%- endif %}
{%- endfor %}

{%- endif %}
{%- endblock %}

.. template class.rst

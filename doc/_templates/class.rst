`{{ objname }}` Class
===============================================================================

.. autoclass:: {{ module }}.{{ objname }}

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
   ~{{ module }}.{{ objname }}.{{ item }}
{%- endfor %}

{%- endif %}
{%- endblock %}

{% block attributes -%}
{% if attributes -%}

Attributes
----------

.. autosummary::
{% for item in attributes %}
   ~{{ module }}.{{ objname }}.{{ item }}
{%- endfor %}

{%- endif %}
{%- endblock %}

{% block static_attributes -%}
{% if static_attributes -%}

Static Attributes
-----------------

======= ========================================================
{% if 'dtype' in static_attributes -%}
|dtype| Value type.
{%- endif %}
======= ========================================================

.. |dtype| replace:: :attr:`~{{fullname}}.dtype`

{%- endif %}
{%- endblock %}


{% block descriptions -%}
{% if (methods) or attributes -%}

Descriptions
------------

{%- block methods_desc %}
{%- if methods %}

.. rubric:: Method Details

{%- for item in methods %}

.. automethod:: {{ module }}.{{ objname }}.{{ item }}

{%- endfor %}

{%- endif %}
{%- endblock %}

{%- block attributes_desc %}
{%- if attributes %}

.. rubric:: Attribute Details

{%- for item in attributes %}
{%- if not item.startswith('_') %}

.. autoattribute:: {{ module }}.{{ objname }}.{{ item }}

{%- endif %}
{%- endfor %}

{%- endif %}
{%- endblock %}

{%- endif %}
{%- endblock %}

{% block static_attributes_desc -%}
{% if static_attributes -%}

.. rubric:: Static Attribute Details

{% if 'dtype' in static_attributes -%}
.. attribute:: {{ module }}.{{ objname }}.dtype

   Value type.

   This attribute is the type of a single element in this vector.
{%- endif %}

{%- endif %}
{%- endblock %}

.. template class.rst

`{{ objname }}` Class
===============================================================================

.. inheritance-diagram:: {{ fullname }}
   :parts: 1

.. autoclass:: {{ fullname }}

{%- block methods %}
{%- if methods and methods != ['__init__'] %}

Methods
-------

.. autosummary::
{% for item in methods %}
{%- if not item.startswith('_') %}
   ~{{ fullname }}.{{ item }}
{%- endif %}
{%- endfor %}

{%- endif %}
{%- endblock %}

{% block attributes -%}
{% if attributes -%}

Attributes
----------

.. autosummary::
{% for item in attributes %}
   ~{{ fullname }}.{{ item }}
{%- endfor %}

{%- endif %}
{%- endblock %}

{% block descriptions -%}
{% if (methods and methods != ['__init__']) or attributes -%}

Descriptions
------------

{%- block methods_desc %}
{%- if methods and methods != ['__init__'] %}

.. rubric:: Method Details

{%- for item in methods %}
{%- if not item.startswith('_') %}

.. automethod:: {{ fullname }}.{{ item }}

{%- endif %}
{%- endfor %}

{%- endif %}
{%- endblock %}

{%- block attributes_desc %}
{%- if attributes %}

.. rubric:: Attribute Details

{%- for item in attributes %}
{%- if not item.startswith('_') %}

.. autoattribute:: {{ fullname }}.{{ item }}

{%- endif %}
{%- endfor %}

{%- endif %}
{%- endblock %}

{%- endif %}
{%- endblock %}

.. template class.rst

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

{% if attributes | select('is_receiver') | first -%}
Receivers
^^^^^^^^^

.. autosummary::
{% for item in attributes | select('is_receiver') %}
   ~{{ fullname }}.{{ item }}
{%- endfor %}
{%- endif %}

{% if attributes | select('is_provider') | first -%}
Providers
^^^^^^^^^

.. autosummary::
{% for item in attributes | select('is_provider') %}
   ~{{ fullname }}.{{ item }}
{%- endfor %}
{%- endif %}

{% if attributes | select('is_other') | first -%}
Other
^^^^^

.. autosummary::
{% for item in attributes | select('is_other') %}
   ~{{ fullname }}.{{ item }}
{%- endfor %}
{%- endif %}

{%- endif %}
{%- endblock %}


Descriptions
------------

{%- block methods_desc %}
{%- if methods %}

Method Details
^^^^^^^^^^^^^^

{%- for item in methods %}
{%- if not item.startswith('_') and item not in ['load_xpl', 'on_initialize', 'on_invalidate']  %}

.. automethod:: {{ fullname }}.{{ item }}

{%- endif %}
{%- endfor %}

{%- endif %}
{%- endblock %}

{%- block attributes_desc %}
{%- if attributes %}

{% if attributes | select('is_receiver') | first -%}
Receiver Details
^^^^^^^^^^^^^^^^

{%- for item in attributes | select('is_receiver') %}

.. autoattribute:: {{ fullname }}.{{ item }}

{%- endfor %}
{%- endif %}

{% if attributes | select('is_provider') | first -%}
Provider Details
^^^^^^^^^^^^^^^^

{%- for item in attributes | select('is_provider') %}

.. autoattribute:: {{ fullname }}.{{ item }}
   :show-signature:

{%- endfor %}
{%- endif %}

{% if attributes | select('is_other') | first -%}
Attribute Details
^^^^^^^^^^^^^^^^^

{%- for item in attributes | select('is_other') %}

.. autoattribute:: {{ fullname }}.{{ item }}

{%- endfor %}
{%- endif %}

{%- endif %}
{%- endblock %}

.. template class.rst

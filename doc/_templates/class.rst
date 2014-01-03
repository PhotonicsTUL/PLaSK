`{{ objname }}` Class
==============================================================

.. template class.rst

.. currentmodule:: {{ module }}

.. inheritance-diagram:: {{ objname }}
   :parts: 1

.. autoclass:: {{ objname }}

{% block methods %}
{% if methods %}

Methods
-------

.. autosummary::
{% for item in methods %}
{% if not item.startswith('_') %}
   ~{{ name }}.{{ item }}
{% endif %}
{%- endfor %}

{% endif %}
{% endblock %}

{% block attributes %}
{% if attributes %}

Attributes
----------

.. autosummary::
{% for item in attributes %}

   ~{{ name }}.{{ item }}
{%- endfor %}

{% endif %}
{% endblock %}



Descriptions
------------

.. class:: {{ objname }}


{% block methods_desc %}
{% if methods %}

   .. rubric:: Method Details

{% for item in methods %}
{% if not item.startswith('_') %}

   .. automethod:: {{ item }}

{% endif %}
{%- endfor %}
{% endif %}
{% endblock %}

{% block attributes_desc %}
{% if attributes %}

   .. rubric:: Attribute Details

{% for item in attributes %}
{% if not item.startswith('_') %}

   .. autoattribute:: {{ item }}

{% endif %}
{%- endfor %}
{% endif %}
{% endblock %}





{% if inherited %}

   .. rubric:: Inherited Member Details

{% block inh_events_desc %}
{% if inh_events %}
{% for item in inh_events %}

   .. automethod:: {{ item }}
      :noindex:

{%- endfor %}
{% endif %}
{% endblock %}

{% block inh_methods_desc %}
{% if inh_methods %}
{% for item in inh_methods %}

   .. automethod:: {{ item }}
      :noindex:

{%- endfor %}
{% endif %}
{% endblock %}

{% block inh_attributes_desc %}
{% if inh_attributes %}
{% for item in inh_attributes %}

   .. autoattribute:: {{ item }}
      :noindex:

{%- endfor %}
{% endif %}
{% endblock %}

{% endif %}


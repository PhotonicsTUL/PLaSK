``{{ fullname }}``
===============================================================================

.. automodule:: {{ fullname }}

{%- block submodules %}
{%- if submodules %}

Modules
-------

.. autosummary::
   :toctree: {{ objname }}
   :template: module.rst
{% for item in submodules %}
   {{ item }}
{%- endfor %}

{%- endif %}
{%- endblock %}


{%- block classes %}
{%- if classes %}

Classes
-------

.. autosummary::
   :nosignatures:
   :toctree: {{ objname }}
   :template: class.rst
{% for item in classes %}
{%- if not item.endswith('_list') and not item.endswith('_entry') %}
   {{ item }}
{%- endif %}
{%- endfor %}

{%- endif %}
{%- endblock %}


{%- block functions %}
{%- if functions %}

Functions
---------

.. autosummary::
   :template: function.rst
{% for item in functions %}
   {{ item }}
{%- endfor %}

{%- endif %}
{%- endblock %}


{%- block exceptions %}
{%- if exceptions %}

Exceptions
----------

.. autosummary::
   :template: exception.rst

{% for item in exceptions %}
   {{ item }}
{%- endfor %}

{%- endif %}
{%- endblock %}


{%- block constants %}
{%- if constants %}

Defined
-------

{%- for item in constants %}
* {{ item }}
{%- endfor %}

{%- endif %}
{%- endblock %}


{%- block functions_dsc %}
{%- if functions %}

Descriptions
------------

Function Details
^^^^^^^^^^^^^^^^

{%- for item in functions %}
{%- if not item.startswith('_') %}

.. autofunction:: {{ item }}

{%- endif %}
{%- endfor %}

{%- endif %}
{%- endblock %}

.. template module.rst

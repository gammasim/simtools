---
myst:
  html_meta:
    "description lang=en": "Documentation for simtools"
sphinx:
  config:
    html_theme_options:
      show_toc: false
      secondary_sidebar_title: "simtools"
      secondary_sidebar_items: []
html_theme.sidebar_secondary.remove: true
---

```{image} ../_static/simtools_logo.png
:align: center
:alt: simtools logo
```

```{eval-rst}
.. currentmodule:: simtools
```

**simtools** is a toolkit for model parameter management, configuring simulation productions, setting, validation workflows.
It is part of the simulation pipeline [SimPipe](http://cta-computing.gitlab-pages.cta-observatory.org/dpps/simpipe/simpipe/latest/) of the [Cherenkov Telescope Array Observatory (CTAO)](https://www.cta-observatory.org/).

````{div} sd-d-flex-row
```{button-ref} user-guide/getting_started
:ref-type: doc
:color: primary
:class: sd-rounded-pill sd-mr-3

Getting Started
```

```{button-link} https://github.com/gammasim/simtools
:color: success
:class: sd-rounded-pill sd-mr-3
:external:

GitHub Repository
```

```{button-link} https://github.com/gammasim/simtools/blob/main/LICENSE
:color: info
:class: sd-rounded-pill sd-mr-3
:external:

License BSD-3
```

````

(simtools_docs)=

```{toctree}
:hidden:
:maxdepth: 1

User Guide <user-guide/index>
Components <components/index>
Data Model <data-model/index>
Developer Guide <developer-guide/index>
API Docs <api-reference/index>
Changelog <changelog>
```

---

::::{grid} 1 1 2 3
:class-container: cards

:::{grid-item-card} {fas}`bolt;pst-color-primary` User Guide
:link: user-guide/index
:link-type: doc
:class-card: sd-d-flex-column

Learn how to get started as a user.
This guide will cover how to install simtools,
and introduce the main functionalities.
+++

```{button-ref} user-guide/index
:expand:
:color: primary
:click-parent:
:class: sd-mt-auto
To the user guide
```
:::

:::{grid-item-card} {fas}`puzzle-piece;pst-color-primary` Components
:link: components/index
:link-type: doc
:class-card: sd-d-flex-column

Learn about the main components working together with simtools:
simulation software and data bases.
+++

```{button-ref} components/index
:expand:
:color: primary
:click-parent:
:class: sd-mt-auto
To the components guide
```
:::

:::{grid-item-card} {fas}`database;pst-color-primary` Data Model
:link: data-model/index
:link-type: doc
:class-card: sd-d-flex-column

Learn about the data model used by simtools for its input,
simulation model parameters, and output.
+++

```{button-ref} data-model/index
:expand:
:color: primary
:click-parent:
:class: sd-mt-auto
To the data model guide
```
:::


:::{grid-item-card} {fas}`code;pst-color-primary` Developer Guide
:link: developer-guide/index
:link-type: doc
:class-card: sd-d-flex-column

Learn how to contribute to simtools.
This guide will cover how to set up a development environment,
how to add code, and how to run tests.
+++
```{button-ref} developer-guide/index
:expand:
:color: primary
:click-parent:

To the developer guide
```
:::

:::{grid-item-card} {fas}`book;pst-color-primary` API Reference
:link: api-reference/index
:link-type: doc
:class-card: sd-d-flex-column

Learn about the details of the simtools API, its functions,
classes, and methods.
+++
```{button-ref} api-reference/index
:expand:
:color: primary
:click-parent:

To the API docs
```
:::

:::{grid-item-card} {fas}`hammer;pst-color-primary` Changelog
:link: changelog
:link-type: doc
:class-card: sd-d-flex-column

Learn what has changed in simtools.
+++
```{button-ref} changelog
:expand:
:color: primary
:click-parent:
To the changelog
```
:::

::::

:::{warning} simtools is currently under heavy development with continuous changes and additions applied. Please contact the developers before using it: simtools-developer@desy.de.
:::

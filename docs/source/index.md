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

**simtools** is a toolkit for simulation of imaging atmospheric Cherenkov telescopes.
It provides tools for model parameter management, configuring simulation productions, and setting up validation workflows.

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

::::{div} sd-my-4

```{raw} html
<svg class="chain-figure" viewBox="0 0 760 285" role="img"
     aria-label="simulation-models and simulation-model-parameter-setting feed simtools, which configures and runs Air showers, Light sources, Atmosphere, Ray tracing, and Camera simulations."
     font-family="inherit" font-size="14" fill="currentColor">
  <defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5"
            markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 10 5 L 0 8.5 z" fill="currentColor" opacity="0.7"/>
    </marker>
  </defs>

  <!-- Top row: Model & Parameter Management -->
  <g fill="none" stroke="var(--pst-color-secondary, #6c757d)" stroke-width="1.5">
    <rect x="64" y="10" width="230" height="44" rx="8"/>
    <rect x="384" y="10" width="312" height="44" rx="8"/>
  </g>

  <!-- Middle hub: simtools -->
  <g fill="none" stroke="var(--pst-color-primary)" stroke-width="2.5">
    <rect x="234" y="92" width="292" height="50" rx="8"/>
  </g>

  <!-- Bottom row: Simulation Chain -->
  <g fill="none" stroke="var(--pst-color-primary)" stroke-width="1.8" stroke-dasharray="5 4">
    <rect x="4" y="182" width="150" height="36" rx="8"/>
    <rect x="4" y="232" width="150" height="36" rx="8"/>
    <rect x="224" y="202" width="140" height="50" rx="8"/>
    <rect x="434" y="202" width="140" height="50" rx="8"/>
    <rect x="644" y="202" width="112" height="50" rx="8"/>
  </g>

  <!-- Model <-> Parameter <-> simtools arrows -->
  <g fill="none" stroke="currentColor" stroke-width="1.5" opacity="0.65" marker-end="url(#arrow)">
    <!-- simulation-models <-> simulation-model-parameter-setting -->
    <path d="M294 26 H382"/>
    <path d="M384 38 H296"/>
    <!-- Top row into simtools -->
    <path d="M179 54 C179 75 310 72 310 90"/>
    <path d="M540 54 C540 75 450 72 450 90"/>
  </g>

  <!-- simtools configuring simulation backends -->
  <g fill="none" stroke="currentColor" stroke-width="1.3" stroke-dasharray="3 3" opacity="0.6" marker-end="url(#arrow)">
    <path d="M280 142 C280 165 79 160 79 180"/>
    <path d="M340 142 C340 172 294 172 294 200"/>
    <path d="M420 142 C420 172 504 172 504 200"/>
    <path d="M480 142 C480 172 700 172 700 200"/>
  </g>

  <!-- Photon stream arrows -->
  <g fill="none" stroke="currentColor" stroke-width="1.5" opacity="0.75" marker-end="url(#arrow)">
    <path d="M154 200 C190 200 190 227 222 227"/>
    <path d="M154 250 C190 250 190 227 222 227"/>
    <path d="M364 227 H432"/>
    <path d="M574 227 H642"/>
  </g>

  <!-- Labels -->
  <g text-anchor="middle">
    <!-- Top row -->
    <text x="179" y="37" font-size="13">simulation-models</text>
    <text x="540" y="37" font-size="13">simulation-model-parameter-setting</text>

    <!-- simtools hub -->
    <text x="380" y="116" font-weight="700" font-size="15">simtools</text>
    <text x="380" y="133" font-size="11.5" opacity="0.8">model management, configuration &amp; workflows</text>

    <!-- Simulation chain -->
    <text x="79" y="205" font-size="13">Air showers</text>
    <text x="79" y="255" font-size="13">Light sources</text>

    <text x="294" y="224" font-weight="600">Atmosphere</text>
    <text x="294" y="241" font-size="11.5" opacity="0.8">propagation</text>

    <text x="504" y="224" font-weight="600">Ray tracing</text>
    <text x="504" y="241" font-size="11.5" opacity="0.8">optics</text>

    <text x="700" y="224" font-weight="600">Camera</text>
    <text x="700" y="241" font-size="11.5" opacity="0.8">Trigger, Digitization</text>

    <!-- Legend / Footer -->
    <text x="380" y="280" font-size="11.5" opacity="0.8">solid horizontal arrows: photon streams; dashed vertical arrows: configuration &amp; model parameters</text>
  </g>
</svg>
```

::::

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

````{card-footer}

```{button-ref} user-guide/index
:expand:
:color: primary
:click-parent:
:class: sd-mt-auto
To the user guide
```
````
:::

:::{grid-item-card} {fas}`puzzle-piece;pst-color-primary` Components
:link: components/index
:link-type: doc
:class-card: sd-d-flex-column

Learn about the main components working together with simtools:
simulation software and data bases.

````{card-footer}

```{button-ref} components/index
:expand:
:color: primary
:click-parent:
:class: sd-mt-auto
To the components guide
```
````
:::

:::{grid-item-card} {fas}`diagram-project;pst-color-primary` Data Model
:link: data-model/index
:link-type: doc
:class-card: sd-d-flex-column

Learn about the data model used by simtools for its input,
simulation model parameters, and output.

````{card-footer}

```{button-ref} data-model/index
:expand:
:color: primary
:click-parent:
:class: sd-mt-auto
To the data model guide
```
````
:::


:::{grid-item-card} {fas}`code;pst-color-primary` Developer Guide
:link: developer-guide/index
:link-type: doc
:class-card: sd-d-flex-column

Learn how to contribute to simtools.
This guide will cover how to set up a development environment,
how to add code, and how to run tests.

````{card-footer}

```{button-ref} developer-guide/index
:expand:
:color: primary
:click-parent:

To the developer guide
```
````
:::

:::{grid-item-card} {fas}`book;pst-color-primary` API Reference
:link: api-reference/index
:link-type: doc
:class-card: sd-d-flex-column

Learn about the details of the simtools API, its functions,
classes, and methods.

````{card-footer}

```{button-ref} api-reference/index
:expand:
:color: primary
:click-parent:

To the API docs
```
````
:::

:::{grid-item-card} {fas}`hammer;pst-color-primary` Changelog
:link: changelog
:link-type: doc
:class-card: sd-d-flex-column

Learn what has changed in simtools.

````{card-footer}

```{button-ref} changelog
:expand:
:color: primary
:click-parent:
To the changelog
```
````
:::

::::

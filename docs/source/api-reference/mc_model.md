(mcmodel)=

# MC Model

The array of imaging atmospheric Cherenkov telescopes is abstracted in the simulation model and divided into the following components:

- [telescope_model] representing a telescope. Defined by its telescope name, allowing to read model parameters from the databases using this name.
- sub-elements of the telescope represented by the modules [mirrors] and [camera]
- an array of telescopes (especially the telescope arrangement) represented by [array_model].

## array_model

(array-model)=

```{eval-rst}
.. automodule:: model.array_model
   :members:

```

## calibration_model

(calibration-model)=

```{eval-rst}
.. automodule:: model.calibration_model
   :members:
```

## camera

(camera-1)=

```{eval-rst}
.. automodule:: model.camera
   :members:

```

## flasher_model

(flasher-model)=

```{eval-rst}
.. automodule:: model.flasher_model
   :members:
```

## mirrors

(mirrors-1)=

```{eval-rst}
.. automodule:: model.mirrors
   :members:

```

## model_parameter

(model-parameters)=

```{eval-rst}
.. automodule:: model.model_parameter
   :members:

```

## model_repository

(model-repository)=

```{eval-rst}
.. automodule:: model.model_repository
   :members:
```

## model utilities

(model-utils)=

```{eval-rst}
.. automodule:: model.model_utils
   :members:
```

## site_model

(site-model)=

```{eval-rst}
.. automodule:: model.site_model
   :members:
```

## telescope_model

(telescope-model)=

```{eval-rst}
.. automodule:: model.telescope_model
   :members:
```

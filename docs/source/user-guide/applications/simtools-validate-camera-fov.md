# simtools-validate-camera-fov

```{eval-rst}
.. automodule:: simtools.applications.validate_camera_fov
   :members:
   :exclude-members: main
```

```{eval-rst}
The orientation for the plotting is "as seen for an observer facing the camera".

An example of the camera plot can be found below.

.. _camera_fov_plot:
.. image:: images/validate_camera_fov_North-LST-1_pixelLayout.png
  :width: 50 %


**Command line arguments**
site (str, required)
    North or South.
telescope (str, required)
    Telescope model name (e.g. LST-1, SST-D, ...)
model_version (str, optional)
    Model version
camera_in_sky_coor (bool, optional)
    Plot the camera layout in sky coordinates akin to looking at it from behind for single \
     mirror telescopes
print_pixels_id (bool, optional)
    Up to which pixel ID to print. To suppress printing of pixel IDs, set to zero\
     (--print_pixels_id 0). To print all pixels, set to 'All'."

**Example**
LST - 5.0.0

.. code-block:: console

    simtools-validate-camera-fov --site North \\
        --telescope LSTN-01 --model_version 5.0.0

The output is saved in simtools-output/validate_camera_fov.

Expected final print-out message:

.. code-block:: console

    Saved camera plot in /workdir/external/simtools/simtools-output/validate_camera_fov\\
    /application-plots/validate_camera_fov_LST-1_pixel_layout.png
```

## Command line arguments

```{eval-rst}
.. simtools-cli-help::
   :application: validate_camera_fov
   :no-heading:
```

## Example

```{eval-rst}
.. simtools-integration-example::
    :file: validate_camera_fov_run.yml
```

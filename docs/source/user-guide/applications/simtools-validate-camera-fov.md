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

To highlight trigger patches, select their model group IDs:

```console
simtools-validate-camera-fov --site North --telescope LSTN-design \
  --model_version 7.0.0 --trigger_patches 0 1 2
```

The plot contains a full-camera overview and a close-up labelled with the selected
pixels' IDs. Patch colours distinguish membership; striped pixels belong to more
than one selected patch, retaining each patch's colour. The legend identifies each
combination. IDs refer
to `group_id` in the model's `camera_trigger_groups` table, not pixel IDs. Select
fewer patches by providing fewer IDs. Omitting `--trigger_patches` keeps the
ordinary camera layout when `--trigger_pixel` is also omitted. In the selected-patch view,
only the close-up labels
pixels; `--print_pixels_id` controls the ordinary layout.

The maximum number of selected patches is the largest number of distinct patches
containing any one pixel in the selected model. For `LSTN-design` model 7.0.0 this
is six. For example, show all six patches containing pixel 0 with
`--trigger_patches 0 3 4 33 34 35`. The selected patches need not share a pixel;
the membership count sets the plotting limit.

Alternatively, select every patch containing a given pixel automatically:

```console
simtools-validate-camera-fov --site North --telescope LSTN-design \
  --model_version 7.0.0 --trigger_pixel 75
```

`--trigger_pixel` accepts a model pixel ID, including zero, and cannot be combined
with `--trigger_patches`. The selected patch IDs are logged. Unknown pixels and
pixels without trigger-group membership are rejected.

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

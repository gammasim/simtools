"""Model-parameter changes of ray-tracing modes, the same for every backend.

Each mode is a set of model-parameter values replacing those of the telescope
model: sim_telarray receives them as configuration options, other backends as changed
parameter records.
"""

# No random mirror errors (focal lengths, reflection angles, alignment).
PERFECT_MIRROR = {
    "random_focal_length": [0.0, 0.0],
    "mirror_reflection_random_angle": [0.0, 0.0, 0.0],
    "mirror_align_random_distance": 0.0,
    "mirror_align_random_horizontal": [0.0, 28.0, 0.0, 0.0],
    "mirror_align_random_vertical": [0.0, 28.0, 0.0, 0.0],
}


def single_mirror(mirror_list, mirror_focal_length, camera_distance):
    """
    Return the changes of the single-mirror mode.

    The mirror is placed alone at the dish centre and imaged from twice its focal
    length; the camera is at that distance.

    Parameters
    ----------
    mirror_list: str or Path
        Mirror list with the single mirror.
    mirror_focal_length: float
        Mirror focal length [cm].
    camera_distance: float
        Distance of camera and light source from the mirror [cm].

    Returns
    -------
    dict
        Model-parameter values.
    """
    return {
        "mirror_list": str(mirror_list),
        "focal_length": camera_distance,
        "dish_shape_length": mirror_focal_length,
        "mirror_focal_length": mirror_focal_length,
        "focus_offset": [0.0, 0.0, 0.0, 0.0],
        "camera_body_diameter": 0.0,
        "parabolic_dish": False,
        "mirror_align_random_distance": 0.0,
        "mirror_align_random_vertical": [0.0, 28.0, 0.0, 0.0],
    }


def sim_telarray_options(changes):
    """
    Return the changes as sim_telarray configuration options (name=value).

    Parameters
    ----------
    changes: dict
        Model-parameter values.

    Returns
    -------
    dict
        Option values as sim_telarray reads them.
    """

    def text(value):
        if isinstance(value, bool):
            return str(int(value))
        if isinstance(value, list | tuple):
            return ",".join(str(v) for v in value)
        return str(value)

    return {name: text(value) for name, value in changes.items()}

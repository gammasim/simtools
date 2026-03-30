# Coordinate Systems

simtools uses multiple coordinate systems for telescope positioning, simulations, and analysis. This document describes the systems and how they are used.
Please refer to the CORSIKA and sim_telarray documentation for details on systems used in these simulation tools.

## Geospatial Coordinate Systems

Three coordinate systems define array element positions on Earth, implemented via `GeoCoordinates` and `TelescopePosition` classes in `simtools.layout`.
The [pyproj](https://pyproj4.github.io/pyproj/stable/) library is used for transformations between these systems.

### Local Transverse Mercator (WGS84)

Geographic latitude and longitude

- **Axes:** Latitude (x), Longitude (y)
- **Units:** Degrees
- **Altitude:** Height above sea level (meters)

This implementation uses **Latitude=x** and **Longitude=y**, which is unconventional compared to standard GIS and geographic conventions. Most geographic information systems follow (longitude, latitude) or equivalently (x, y) where:

- x = Longitude (east-west axis)
- y = Latitude (north-south axis)

### UTM (Universal Transverse Mercator) System

UTM coordinates

- **Axes:** Easting (x), Northing (y)
- **Units:** Meters
- **Scale factor:** $k_0 = 0.9996$ at central meridian
- **Altitude:** Height above sea level (meters)

### Ground (Local/sim_telarray) System

Local Cartesian system centered at array center.

- **Origin:** Array center (reference point)
- **Axes:** X→North, Y→West, Z→Up (NWU)
- **Units:** Meters
- **Projection:** Transverse Mercator centered at array
- **Scale factor:** $k_0=(R+altitude)/R$ depends on latitude and altitude (WGS84 ellipsoid; R is the geocentric radius)
- **Z coordinate:** `position_z`, height above a reference altitude (e.g., CORSIKA observation level), in meters
- **Usage:** CORSIKA and sim_telarray simulations (native system)

Note the differences between the altitude of the array center and the CORSIKA observation level.

#### CORSIKA Specifics

Telescope positions in CORSIKA are defined by their (x, y) coordinates in the ground system and an altitude (z) calculated as:

$$\text{position\_z} = \text{altitude} - \text{corsika\_observation\_level} + \text{telescope\_axis\_height}$$
The CORSIKA observation level must be below the altitude of any telescope to ensure positive position_z values.

CORSIKA uses the ARRANGE card to specify geomagnetic field rotation, aligning the shower coordinate system with geographic North when needed.

## Shower Coordinate System

Local frame aligned with shower propagation direction. Used for ray-tracing and event reconstruction. Transforms ground coordinates based on shower azimuth and zenith angle.

**Implementation:** `simtools.utils.geometry.transform_ground_to_shower_coordinates()`

## Camera Coordinate Systems

### Pixel Coordinates

Photosensor positions in the focal plane.

- **Plane:** Focal plane
- **Origin:** Camera center
- **Units:** Centimeters (cm)
- **Pixel shapes:** Hexagonal (codes 1, 3 for different orientation) or Square (code 2)
- **Data source:** sim_telarray camera configuration files
- **Storage:** `simtools.model.camera.Camera` class

Includes pixel position (x, y), ID, on/off status, and diameter.

### Camera Coordinates

same as pixel coordinates, but no camera rotation applied.

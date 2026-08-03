from astropy.table import Column, Table


def _create_merged_table():
    return Table(
        [
            Column(data=[20.0, 40.0], name="zenith"),
            Column(data=[0.0, 180.0], name="azimuth"),
            Column(data=["dark", "moon"], name="nsb_level"),
            Column(data=["alpha", "alpha"], name="array_name"),
            Column(data=[0.1, 0.2], name="lower_energy_limit"),
            Column(data=[1200.0, 1500.0], name="upper_radius_limit"),
            Column(data=[8.0, 10.0], name="viewcone_radius"),
        ]
    )

# Global monitoring of volcanic impacts on glacier flow

Create graphs of raw and [TICOI](https://github.com/ticoi/ticoi) regularised [ITS_LIVE](https://its-live.jpl.nasa.gov/) velocity time series along the centre lines of glaciers from the [Randolph Glacier Inventory](https://www.glims.org/rgi_user_guide/welcome.html) (around volcanoes from the [Smithsonian's Global Volcanism Program](https://volcano.si.edu/)).

See notebook [gv_velocity.ipynb](../main/gv_velocity.ipynb) for an easy example to create centreline velocities using default TICOI parameters.

To run the analysis for the entire globe, first follow the instructions in [glac_by_volc](https://github.com/TryggviU/glac_by_volc) ([Unnsteinsson et al., 2025](https://doi.org/10.1038/s41467-025-63332-2)) to locate all glaciers around volcanoes.
Then execute:
````commandline
python src\gv_climate.py
````
to download all ERA5 climate files for each volcano, and then execute:
````commandline
python src\gv_velocity.py
````
to run the analysis on the selected volcanoes and glaciers.

To choose which volcanoes and glaciers to analyse, edit the `ids` dictionary in [gv_velocity.py](src/gv_velocity.py) and give entries as `GVP_ID: {RGI_G_ID: [RGI_L_ID1, RGI_L_ID2, ...]}`.
The IDs are `GVP_ID` is the volcano number from the [Smithsonian's Global Volcanism Program](https://volcano.si.edu/), and `RGI_G_ID` and `RGI_L_ID` are the glacier and centreline numbers, respectively, from the [Randolph Glacier Inventory](https://www.glims.org/rgi_user_guide/welcome.html).

To choose which filters to use for ITS_LIVE and hyperparameters for TICOI, edit the `itslive_kwargs` and `ticoi_kwargs` dictionaries. See avaiable hyperparameters for TICOI here: [https://github.com/ticoi/ticoi](https://github.com/ticoi/ticoi)
The script includes all parameter sets discussed in the manuscript below with the defaults selected, but the ITS_LIVE filter and the TICOI tests 01 and 02 can be uncommented to be used.

Please cite the following if using the code:

Unnsteinsson T, Spagnolo M, Rea BR, Girona T, Mullan D, and Barr I (2026). _Monitoring volcanic impacts on glacier flow_ [In press]. University of Aberdeen

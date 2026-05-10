import os, sys
from dotenv import load_dotenv
load_dotenv("..\.env")
env = os.environ

import pandas as pd
import geopandas as gpd
import shapely
import numpy as np
import bokeh.models as bkhm
import bokeh.colors as bokeh_colors

sys.path.append(os.path.abspath('../../src')) # Ideally ST_Visions will be installed as a module, therefore this is more of a development setting for quick testing.

from visualization.st_visualizer import st_visualizer
import express.st_vizexpress as viz_express
import visualization.callbacks as viz_callbacks
import visualization.geom_helper as viz_helper

pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


st_viz = st_visualizer(limit=500)
st_viz.get_data_csv(env['SARONIC_GULF_AIS'], nrows=1000)
st_viz.create_canvas(title=f'Prototype Plot', sizing_mode='scale_width', height=540, tools="pan, box_zoom, lasso_select, wheel_zoom, hover, save, reset")
st_viz.add_marker(marker='circle', size=10, color='royalblue', alpha=0.5, fill_alpha=0.5, muted_alpha=0, legend_label=f'GPS Locations A')
st_viz.add_hover_tooltips(tooltips = [('Vessel ID','@vessel_id'), ('Coordinates','(@lon, @lat)')], renderers=st_viz.renderers)

st_viz2 = st_visualizer(limit=500)
st_viz2.get_data_csv(env['CATEGORICAL_SUBSET_DEMO'], nrows=1000)
st_viz2.set_figure(st_viz.figure)
st_viz2.create_source()
st_viz2.add_marker(marker='triangle', size=10, color='red', alpha=0.5, fill_alpha=0.5, muted_alpha=0, legend_label=f'GPS Locations B')
st_viz2.add_hover_tooltips(tooltips = [('Vessel ID','@vessel_id'), ('Coordinates','(@lon, @lat)')], renderers=st_viz2.renderers)

st_viz2.figure.legend.location = "top_left"
st_viz2.figure.legend.click_policy = "mute"
st_viz2.figure.toolbar.active_scroll = st_viz2.figure.select_one(bkhm.WheelZoomTool)
st_viz2.show_figures(notebook=False, sizing_mode='stretch_both')


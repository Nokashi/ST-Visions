import sys, os
sys.path.append(os.path.abspath('../../src'))

from st_visions.config.load_env import load_environment
from bokeh.models import Div, WheelZoomTool
from bokeh.layouts import row
from st_visions.visualization.st_visualizer import st_visualizer
import st_visions.express.st_vizexpress as viz_express
env = load_environment()

# TO EXECUTE SCRIPT USE (ON REMOTE SERVER)
# python -m bokeh serve --show data/scripts/test.py --allow-websocket-origin=<NODE_IP_ADDRESS>:<BOKEH_PORT>

# TO EXECUTE SCRIPT USE (ON LOCAL SERVER)
# python -m bokeh serve --show data/scripts/test.py


st_viz = st_visualizer(limit=5000)
st_viz.get_data_csv(env['CATEGORICAL_SUBSET_DEMO'])

tooltips = [('Vessel ID','@vessel_id'), ('Timestamp','@t'), ('Speed (knots)','@speed'),
            ('Course over Ground (degrees)','@course'), ('Heading (degrees)','@heading'), ('Coordinates','(@lon, @lat)'), ('Vessel Type','@vessel_type')]


viz_express.plot_points_on_map(st_viz, size=7, tools=['hover,lasso_select'], tooltips=tooltips, sizing_mode='fixed', width=1200)

st_viz.add_categorical_filter(title='Vessel Type', categorical_name='vessel_type')
st_viz.add_numerical_filter(filter_mode='>=', callback_policy='value_throttled', title='Speed (knots) >=', numeric_name='speed', step=1)

st_viz.figure.legend.location = "top_left"
st_viz.figure.legend.click_policy = "mute"
st_viz.figure.toolbar.active_scroll = st_viz.figure.select_one(WheelZoomTool)
st_viz.show_figures(notebook=False)


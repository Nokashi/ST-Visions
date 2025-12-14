import sys, os
sys.path.append(os.path.abspath('../../src'))

from bokeh.models import Div, WheelZoomTool
from bokeh.layouts import row
from visualization.st_visualizer import st_visualizer
from dotenv import load_dotenv
load_dotenv("..\.env")
env = os.environ

# TO EXECUTE SCRIPT USE (ON REMOTE SERVER)
# python -m bokeh serve --show data/scripts/PY1-simple_visualization.py --allow-websocket-origin=<NODE_IP_ADDRESS>:<BOKEH_PORT>

# TO EXECUTE SCRIPT USE (ON LOCAL SERVER)
# python -m bokeh serve --show data/scripts/PY1-simple_visualization.py


st_viz = st_visualizer(limit=5000) # Initialize a VISIONS Instance (ST Visualizer Object)
st_viz.get_data_csv(filepath=env['SARONIC_GULF_AIS'], nrows=10000)

st_viz.create_canvas(title=f'Simple Visualization', tile_provider="CARTODBPOSITRON", sizing_mode='fixed', height=800, tools="pan, box_zoom, lasso_select, wheel_zoom, hover, save, reset", width = 800)
circ = st_viz.add_marker(marker='circle', size=10, color='royalblue', alpha=0.7, fill_alpha=0.5, muted_alpha=0, legend_label=f'Vessel GPS Locations')


tooltips = [('Vessel ID','@vessel_id'), ('Timestamp','@t'), ('Speed (knots)','@speed'),
            ('Course over Ground (degrees)','@course'), ('Heading (degrees)','@heading'), ('Coordinates','(@lon, @lat)')]

st_viz.add_hover_tooltips(tooltips)
st_viz.add_lasso_select()

st_viz.figure.legend.location = "top_left"
st_viz.figure.legend.click_policy = "mute"
st_viz.figure.toolbar.active_scroll = st_viz.figure.select_one(WheelZoomTool)

st_viz.add_temporal_filter(temporal_name='t', temporal_unit='ms', step_ms=180000, title='Temporal Horizon')

#description = Div(text="""
#<h3>Python Example 1</h3>
##<p>This simple python file generates an st_visions figure along with a simple temporal filter</p>
#""", width=500)

# figure_with_text = row(st_viz.figure, description)

st_viz.show_figures(notebook=False, sizing_mode='fixed')

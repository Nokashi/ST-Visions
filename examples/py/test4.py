import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import bokeh.models as bokeh_models
from st_visions.st_visualizer import st_visualizer

# TO EXECUTE SCRIPT USE (ON REMOTE SERVER)
# python -m bokeh serve --show data/scripts/test2.py --allow-websocket-origin=<NODE_IP_ADDRESS>:<BOKEH_PORT>

# TO EXECUTE SCRIPT USE (ON LOCAL SERVER)
# python -m bokeh serve --show data/scripts/test2.py


st_viz = st_visualizer(limit=5000) # Initialize a VISIONS Instance (ST Visualizer Object)
st_viz.get_data_csv(filepath=r'ST-Visions\data\unipi_ais_dynamic_2017\unipi_ais_dynamic_dec2017.csv', nrows=10000)

st_viz.create_canvas(title=f'Prototype Plot', tile_provider="CARTODBPOSITRON", sizing_mode='scale_width', height=540, tools="pan, box_zoom, lasso_select, wheel_zoom, hover, save, reset")
circ = st_viz.add_marker(marker='circle', size=10, color='royalblue', alpha=0.7, fill_alpha=0.5, muted_alpha=0, legend_label=f'Vessel GPS Locations')

tooltips = [('Vessel ID','@vessel_id'), ('Timestamp','@t'), ('Speed (knots)','@speed'),
            ('Course over Ground (degrees)','@course'), ('Heading (degrees)','@heading'), ('Coordinates','(@lon, @lat)')]

st_viz.add_hover_tooltips(tooltips)
st_viz.add_lasso_select()

st_viz.figure.legend.location = "top_left"
st_viz.figure.legend.click_policy = "mute"
st_viz.figure.toolbar.active_scroll = st_viz.figure.select_one(bokeh_models.WheelZoomTool)

st_viz.add_temporal_filter(temporal_name='t', temporal_unit='ms', step_ms=180000, title='Temporal Horizon')

st_viz.show_figures(notebook=False, sizing_mode='stretch_both')

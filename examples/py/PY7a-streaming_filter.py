import os, sys
import pyarrow as pa
import bokeh.models as bokeh_models

sys.path.append(os.path.abspath('../../src'))

from st_visions.visualization.st_visualizer import st_visualizer
from st_visions.streaming.st_vizstream import ST_KafkaStream



expected_schema = pa.schema([
    ("lon", pa.float64()),
    ("lat", pa.float64()),
    ("vessel_id", pa.int64()),
    ("speed", pa.float32()),
    ("course", pa.float32()),
    ("heading", pa.float32()),
    ("t", pa.timestamp('ms'))
])


st_viz = st_visualizer(limit=5000, expected_schema=expected_schema) # Initialize a VISIONS Instance (ST Visualizer Object)
st_viz.create_canvas(title=f'Showing Streaming Data', tile_provider="CARTODBPOSITRON", sizing_mode='scale_width', height=540, tools="pan, box_zoom, lasso_select, wheel_zoom, hover, save, reset")
circ = st_viz.add_marker(marker='circle', size=10, color='royalblue', alpha=0.7, fill_alpha=0.5, muted_alpha=0, legend_label=f'Vessel GPS Locations')


stream = ST_KafkaStream(topic_name='st-viz-topic')
st_viz.get_data_stream(stream=stream, notebook=False, refresh_rate=500)


tooltips = [('Vessel ID','@vessel_id'), ('Timestamp','@t'), ('Speed (knots)','@speed'),
            ('Course over Ground (degrees)','@course'), ('Heading (degrees)','@heading'), ('Coordinates','(@lon_merc, @lat_merc)')]

st_viz.add_numerical_filter(title='Speed (knots) >=', live=True, filter_mode='>=', numeric_name='speed', step=1, callback_policy='value_throttled')
# st_viz.add_temporal_filter(temporal_name='t', live=True, temporal_unit='ms', step_ms=180000, title='Temporal Horizon', callback_policy='value_throttled')
st_viz.add_hover_tooltips(tooltips)
st_viz.add_lasso_select()


st_viz.figure.legend.location = "top_left"
st_viz.figure.legend.click_policy = "mute"
st_viz.figure.toolbar.active_scroll = st_viz.figure.select_one(bokeh_models.WheelZoomTool)

st_viz.show_figures(notebook=False, live=True, height=600, width=600, sizing_mode='stretch_both')
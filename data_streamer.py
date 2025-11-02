import time
from kafka import KafkaProducer
import json
import pandas as pd
from loguru import logger
import numpy as np

import bokeh
import bokeh.io as bokeh_io
import bokeh.plotting as bokeh_plt
import bokeh.models as bokeh_mdl
import bokeh.palettes as palettes

from bokeh.plotting import figure, reset_output, output_notebook, show
import src.st_visions.visualization.providers as providers


def simulate_kafka_stream(
    csv_path,
    topic="test_topic",
    bootstrap_servers="localhost:9092",
    key_field=None,
    delay=0.1
):
    df = pd.read_csv(csv_path).head(1000)
    print(f"Loaded CSV: {len(df)} rows. Streaming to '{topic}'...")

    producer = KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )

    records = df.to_dict(orient="records")  #TODO: orient(list)
    for record in records:
        producer.send(topic, value=record)
        print(f"[KAFKA PRODUCER] Sending the following:  {record}")
        producer.flush()
        time.sleep(delay) 

    producer.close()
    print("Test stream finalized")

def create_canvas(self, title, x_range=None, y_range=None, tile_provider='CARTODBPOSITRON', suffix='_merc', tile_kwargs={}, **kwargs):        
    """
    Create the instance's Canvas and CDS.

    Parameters
    ----------
    title: str
        The Canvas' title. If the limit set at the constructor is less than the length of the loaded data, the title will be suffixed by ``` - Showing {self.limit} out of {len(self.data)} records'```
    x_range: NumPy Array
        The Canvas' spatial horizon at the longitude dimension
    y_range: Numpy Array
        The Canvas' spatial horizon at the latitude dimension
    tile_provider : str or WMTSTileSource
        Either a provider name string (e.g., 'osm') from the allowed or
        an existing WMTSTileSource instance.
    suffix: str (default: ```'_merc'```)
        A suffix for the column name of the extracted spatial coordinates
    tile_kwargs: Dict
        Additional Keyword arguments related to the tile provider of the instance's canvas (consult the WMTSTileSource Docs)
        https://docs.bokeh.org/en/latest/docs/reference/models/tiles.html
    **kwargs: Dict
        Other arguments related to creating the instance's Canvas (consult bokeh.plotting.figure method)
    """
    if self.data is None:
        logger.error('You must set a DataFrame first')
        raise ValueError('No DataFrame set.')
    
    if self.limit < len(self.data):
        title = f'{title} - Showing {self.limit} out of {len(self.data)} records'

    bbox = self.data.total_bounds
    if x_range is None:
        x_range=(np.floor(bbox[0]), np.ceil(bbox[2]))
    if y_range is None:
        y_range=(np.floor(bbox[1]), np.ceil(bbox[3]))

    fig = figure(x_range=x_range, y_range=y_range, x_axis_type="mercator", y_axis_type="mercator", title=title, **kwargs)
    
    self.set_figure(fig)   
    
    if self.source is None:
        self.create_source(suffix)

    providers.add_tile_to_canvas(self, tile_provider=tile_provider, **tile_kwargs)


if __name__ == "__main__":
    simulate_kafka_stream(r'data\unipi_ais_dynamic_2017\unipi_ais_dynamic_dec2017.csv', 'st-viz-topic')
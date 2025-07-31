import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import pandas as pd
import geopandas as gpd
import shapely
import numpy as np
import bokeh.models as bkhm
import bokeh.colors as bokeh_colors

from st_visions.st_visualizer import st_visualizer
import st_visions.express as viz_express
import st_visions.geom_helper as viz_helper
import st_visions.providers as viz_providers
import st_visions.callbacks as viz_callbacks

pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)



### Loading GeoLife Dataset
gdf = pd.read_csv(r'..\..\data\unipi_ais_dynamic_2017\unipi_ais_dynamic_dec2017.csv')
gdf = viz_helper.create_geometry(gdf, crs=4326)

### Creating Choropleth (Grid) Geometry
# get bounding box as (minx, miny, maxx, maxy)
bbox = gdf.total_bounds
west, south, east, north = bbox

polygon_corners = [
    (west, north),
    (east, north),
    (east, south),
    (west, south)
]

bbox_polygon = shapely.geometry.Polygon(polygon_corners)
cut_result = viz_helper.quadrat_cut_geometry(bbox_polygon, 0.7)

spatial_coverage_cut = gpd.GeoDataFrame(
    geometry=list(cut_result.geoms) if hasattr(cut_result, 'geoms') else [cut_result],
    crs=4326
)


#Classify points into spatial areas
classified_gdf = viz_helper.classify_area_proximity(gdf.copy(), spatial_coverage_cut, compensate=True, verbose=True)

# Get how many points are in each area
cnt = classified_gdf['area_id'].value_counts()

# Assign counts to spatial_coverage_cut GeoDataFram. Note: spatial_coverage_cut index should match the area_id values
spatial_coverage_cut = spatial_coverage_cut.copy() 
spatial_coverage_cut['count'] = 0 
spatial_coverage_cut.loc[cnt.index, 'count'] = cnt.values

if spatial_coverage_cut.geometry.name != 'geometry':
    spatial_coverage_cut = spatial_coverage_cut.rename_geometry('geometry')



st_viz = st_visualizer(limit=len(spatial_coverage_cut))
st_viz.set_data(spatial_coverage_cut.dropna())

st_viz.create_canvas(title=f'Prototype Plot', sizing_mode='scale_width', height=540, tools="pan, box_zoom, lasso_select, wheel_zoom, hover, save, reset")

st_viz.add_numerical_colormap('Viridis256', 'count', colorbar=True, cb_orientation='vertical', cb_location='right', label_standoff=12, border_line_color=None, location=(0,0))
st_viz.add_polygon(fill_color=st_viz.cmap, line_color=st_viz.cmap, fill_alpha=0.6, muted_alpha=0, legend_label=f'GPS Locations (Choropleth Map)')

st_viz.figure.legend.location = "top_left"
st_viz.figure.legend.click_policy = "mute"
st_viz.figure.toolbar.active_scroll = st_viz.figure.select_one(bkhm.WheelZoomTool)



data_points = st_visualizer(limit=len(gdf))
data_points.set_data(gdf)
data_points.set_figure(st_viz.figure)


categorical_name='label'

class Callback(callbacks.BokehFilters):
    def __init__(self, vsn_instance, widget):
        super().__init__(vsn_instance, widget)
        
        
    def callback_prepare_data(self, new_pts, ready_for_output):
        self.vsn_instance.canvas_data = new_pts

        if ready_for_output:
            cnt = viz_helper.classify_area_proximity(self.vsn_instance.canvas_data, st_viz.data, compensate=True, verbose=True).area_id.value_counts()            
            st_viz.canvas_data = st_viz.data.loc[cnt.index].copy()
            st_viz.canvas_data.loc[:, 'count'] = cnt.values               
           
            st_viz.canvas_data = st_viz.prepare_data(st_viz.canvas_data)

            low, high = st_viz.canvas_data[st_viz.cmap['field']].agg([np.min, np.max])
            st_viz.cmap['transform'].low = 0 if low == high else low
            st_viz.cmap['transform'].high = high
                    
            st_viz.source.data = st_viz.canvas_data.drop(st_viz.canvas_data.geometry.name, axis=1).to_dict(orient="list")

            # print ('Releasing Lock...')
            st_viz.canvas_data = None
            self.vsn_instance.canvas_data = None
            self.vsn_instance.aquire_canvas_data = None
        
        
    def callback(self, attr, old, new):
        self.callback_filter_data()

        cat_value = self.widget.value
        new_pts = self.get_data()

        # print (cat_value, categorical_name)
        if cat_value:
            new_pts = new_pts.loc[new_pts[categorical_name] == cat_value].copy()

        self.callback_prepare_data(new_pts, self.widget.id==self.vsn_instance.aquire_canvas_data)

        
data_points.add_categorical_filter(title='Vehicle', categorical_name=categorical_name, height_policy='min', callback_class=Callback)


### Camera, Lights, Action
data_points.figure.legend.location = "top_left"
data_points.figure.legend.click_policy = "mute"
data_points.figure.toolbar.active_scroll = data_points.figure.select_one(bkhm.WheelZoomTool)


data_points.show_figures(notebook=False)



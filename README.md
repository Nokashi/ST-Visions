# ST_Visions+ (Name Tentative)
### A fork of ST_Visions focused on Modernization and Online Analytics Feature Extention.

## Overview
---
Built on the already existing [ST_Visions](https://github.com/DataStories-UniPi/ST-Visions) library (**S**patio-**T**emporal **Vis**ualizat**ions**) (Tritsarolis et al.), ST_Visions+ (Name Tentative) aims to Improve, Modernize and expand on the capabilities of the existing implementation, with respect to todays needs for clear Visual Analytics and Data Visualization. Work on the Library has two main pillars

- Modernization of the library to match the cutting edge versions of all the dependecies needed
- Further Implementation of important features for the modern Data Science space, such as Datastream visualization in real time, and additional Visual Analytics capabilities

Work on this Fork is done as part of the following BSc Thesis 

**ST_Visions+: Extending the ST_Visions Library for Real-Time Visual Analytics**. Paraschos Moraitis


## Installation
---
In order to use ST_Visions in your project, download all necessary modules in your directory of choice via pip or conda, install the class’ dependencies, as the following commands suggest:

```Python
# Using pip/virtualenv
pip install −r requirements.txt

# Using conda
conda install --file requirements.txt
```


## Usage
---
ST_Visions can be used in two variations, depending on the use-case. For baseline visualizations, the module ```express.py``` provides 3 methods for visualizing Point, (Multi)Polygon and (Multi)Line datasets, respectively. For example, to visualize a Point geometry dataset:

* Using the ```st_visualizer.py``` module: 

```Python
import st_visions.st_visualizer as viz

# Initialize an ST_Visions Instance
plot = viz.st_visualizer()

# Load Dataset
plot.get_data_csv("<PATH-TO-CSV-FILE>")

# Create the canvas of the instance
plot.create_canvas(title=f'Prototype Plot', sizing_mode='fixed', height=540)


# Visualize the points to the canvas 
__ = plot.add_glyph(marker='circle', size=10, color='royalblue', alpha=0.7, fill_alpha=0.5, muted_alpha=0, legend_label=f'Vessel GPS Locations')

# Set WheelZoomTool as the active scroll tool
plot.figure.toolbar.active_scroll = plot.figure.select_one(viz.WheelZoomTool)
```

* Using the ```express.py``` module: 

```Python
import pandas as pd
import st_visions.st_visualizer as viz
import st_visions.express as viz_express

# Initialize an ST_Visions Instance
plot = viz.st_visualizer()

# Load Dataset
plot.get_data_csv("<PATH-TO-CSV-FILE>")

# Visualize data on the map
viz_express.plot_points_on_map(plot)
```

Finally, to show our figure, the ```show_figures``` method is used. Depending on the use-case, figures can be visualized either within a Jupyter Notebook cell or a Browser Window (as a Python Script).

```Python
# Render on Jupyter Notebook; or
plot.show_figures(notebook=True, notebook_url='http://<NOTEBOOK_IP_ADDRESS>:<NOTEBOOK_PORT>')

# Render on Browser Window (via Python Script)
plot.show_figures(notebook=False)
```

## Documentation
---
To learn more about ```ST_Visions``` and its capabilities, please consult the technical report at ```./doc/report.pdf```. Example codes that show both baseline and advanced use-cases, can be found at ```./examples/ipynb/``` for Jupyter Notebooks and ```./examples/py/``` for Python Scripts.


## Contributors
---
Andreas Tritsarolis, Christos Doulkeridis, Yannis Theodoridis and Nikos Pelekis; Data Science Lab., University of Piraeus


## Acknowledgement
---
This  project  has  received  funding  from  the  Hellenic Foundation for Research and Innovation (HFRI) and the General Secretariat for Research and Technology (GSRT), under grant agreement No 1667, from 2018 National Funds Programme of the GSRT, and from EU/H2020 project VesselAI (grant agreement No 957237).
# Examples

This directory contains notebooks and python scripts showcasing 

## Prerequisites
- Docker & Docker Compose
- Python 3.x
- Jupyter Notebook 6.x 

## Setup

### 1. Start Kafka through the docker image

```bash
docker compose up -d
```
### 2. Set Up Your Environment
create a `data` folder for your datasets inside the `examples` folder and a `.env` file containing the paths to your datasets.

```
# example variables from the thesis demos, 
# be advised that the "categorical" dataset path assumes
# at least once categorical column 

# for python scripts and jupyter notebooks we have to go one level back to access the data
SARONIC_GULF_AIS = ..\data\your_numerical_dataset.csv
CATEGORICAL_SUBSET_DEMO = ..\data\your_categorical_dataset.csv

# for the data_streamer.py we can access the folder directly
SARONIC_GULF_AIS_STREAMER = data\your_numerical_dataset.csv
CATEGORICAL_SUBSET_DEMO_STREAMER = data\your_categorical_dataset.csv
```

### 3. Run the Data Streamer

Used for the live streaming demos. The `data_streamer.py` script simulates a real-time AIS data feed by producing 
messages to a Kafka topic. It supports two streaming modes:

```bash
# Stream the numerical subset
python data_streamer.py numerical

# Stream the categorical subset  
python data_streamer.py categorical
```

### 4. Run a Demo
Static data demos dont require a data streamer. Either open one of the notebooks in `ipynb/` or run a script from `py/`.

## Notebooks

| Name | Description |
|------|-------------|
| LD-1 Visualization | Introductory notebook to basic visualization tasks with ST_Visions|
| LD-2 Data Colorization | Colormapping features |
| LD-3 Advanced Use Cases| Several advanced use cases using the library e.g. chloropleths, multiple datasets, etc. |
| LD-4 Streaming| Live data streaming showcase on a Jupyter notebook |

## Scripts

| Name | Description |
|------|-------------|
| PY1-simple_visualization | A basic ST_Vision static dataset figure |
| PY2-numerical_filter | Applying a numerical filter on a static dataset |
| PY3-filter_interaction | Showcase Filter interaction between multiple filters on a static dataset |
| PY4-colorization| Data colormapping example |
| PY5-data_table| Example figure with accompanying data table |
| PY6a-streaming| Basic data streaming example |
| PY6b-streaming_express| Basic data streaming example usiong the express module |
| PY7a-streaming_numerical_filter | Data streaming with an active numerical filter  |
| PY7b-streaming_multiple_filters | Categorical and numemrical filter interaction on the categorical subset |
| PY7c-streaming_numerical_filter_colormap | Data streaming with an active numerical filter and a numerical colormap |
| PY8-multiple_datasets | Handling multiple datasets on the same figure |


## Stopping the kafka image
```bash
docker compose down
```
# data
import json
import pandas as pd
import numpy as np

# visualization
from bokeh.models.sources import ColumnDataSource, CDSView
from bokeh.layouts import column, gridplot, row
from bokeh.transform import transform, Jitter, linear_cmap
from bokeh.models import (
                        DataTable, TableColumn, Panel, Tabs,
                        Button, Slider, MultiChoice,Dropdown,RadioButtonGroup,
                        ColorBar, LinearColorMapper,
                        )

from bokeh.palettes import RdYlBu5, Category10, Turbo256, Inferno256
from bokeh.plotting import figure, curdoc, show
from jinja2 import Template

# machine learning
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# for the command line:

# conda activate bokehDashboard
# cd <path to myapp>
# bokeh serve --show mainPart1.py

# use your browser and go to this address to see your dashboard 
# http://localhost:5006/


# ==== util functions ====
def get_second(x):
  return x[1]
def get_rank(data):
    dataID = [(i,x) for i, x in enumerate(data)]
    dataID.sort(key=get_second)
    ranks = [0]* len(data)
    for i in range(len(ranks)):
        ranks[dataID[i][0]] = i
    return ranks
#=============================

# load data & create a dataframe
csvPath =  r".\data\Final_centroids_4.csv"
df = pd.read_csv(csvPath)

#====================================================================================
# ===== Part I: understanding the data with tables, distribution plots and maps =====

# 1) get base stats on each column/varaible of our dataset
df_description = df.describe().T

# 2) transform to bokeh data format 
tableSource = ColumnDataSource(df_description)

# 3) create a interactive table
tableColumns = []
for colName in tableSource.data.keys():
    tableColumns.append(TableColumn(field=colName, title=colName))

data_table = DataTable(
    source=tableSource,
    columns=tableColumns,
    width=500,
    height=300,
    )
#===============================================
# ==== creating figures and plotting data  =====

# 1) create ColumnDataSource for rank/value + map plot
distData = pd.DataFrame()
distData["value"]= df["B_Co2_%"]
distData["rank"] = get_rank(df["B_Co2_%"])
distData["x_coord"] =  df["x_coord"]
distData["y_coord"] = df["y_coord"]
# 2) transform to bokeh data format
distSource = ColumnDataSource(distData)

# 3) Figures & Glyphs

# == create figure for rank/value plot ==
distFig = figure(
            plot_width=300,
            plot_height=200,
            toolbar_location ="left",
            tools="lasso_select"
            )
# add glyphs to the figure 
distFig.circle(
    x="rank",
    y="value",
    fill_color = "blue",
    selection_fill_color = "red",
    selection_line_width = 2,
    selection_line_color = "red",
    fill_alpha = 0.05,
    line_width=0,
    muted_alpha = 0.05,
    size=4,
    name="distView",
    source=distSource
)
# == create figure for map plot ==
mapFig = figure(
        plot_width=300,
        plot_height=200,
        toolbar_location ="left",
        tools="lasso_select"
    )

# add glyphs to map plot

mapper= LinearColorMapper(palette=Turbo256)
mapFig.circle(
    x="x_Coord",
    y="y_Coord",
    fill_color = {'field': 'value', 'transform': mapper},
    selection_fill_color = "red",
    fill_alpha = 1,
    line_width=0,
    muted_alpha = 0.4,
    size=4,
    name="mapView",
    source=distSource
)


# ==================================
# == callbacks: add interactivity ==

def updateDistCDS():
    # 1) re-create ColumnDataSource for dist & map figure
    distData = pd.DataFrame()
    distData["value"]= df[selectionName]
    distData["rank"] = get_rank(df[selectionName])
    distData["x_coord"] =  df["x_coord"]
    distData["y_coord"] = df["y_coord"]
    # 2) update data
    distFig.select("distView").data_source.data = distData
    mapFig.select("mapView").data_source.data = distData
    # 3) update titles
    distFig.title = selectionName + " distribution"
    mapFig.title = selectionName + " map"


# trigger a callback when a row on the table is selected 
def tableSelection_callback(attrname, old, new):
    # 1) get selected row id
    selectionIndex=tableSource.selected.indices[0]
    # 2) translate to column name
    selectionName = tableSource.data["index"][selectionIndex]
    # 3) call functio to update plots
    updateDistCDS(selectionName)

tableSource.selected.on_change('indices', tableSelection_callback)
    


#==========================
#=== create layout ========
bokehLayout = row(data_table,distFig,mapFig)


# add layout to curDoc (HTML)
curdoc().add_root(bokehLayout)


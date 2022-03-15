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
distData["value"] = df["B_NPP_%"]
distData["rank"] = get_rank(df["B_NPP_%"])
distData["x_coord"] = df["x_coord"]
distData["y_coord"] = df["y_coord"]


distSource = ColumnDataSource(distData)
# 3) Figures & Glyphs
distFig = figure(
    plot_width=600,
    plot_height=400,
    tools="lasso_select"
    )
distFig.circle(
    x="rank",
    y="value",
    source=distSource,
    size=2,
    fill_color="blue",
    name="distFig_circle"

)
# == create figure for rank/value plot ==

# add glyphs to the figure 


# == create figure for map plot ==

# add glyphs to map plot
mapFig = figure(
    plot_width = 600,
    plot_height = 400,
    tools="lasso_select"
)
mapFig.circle(
    x="x_coord",
    y="y_coord",
    source=distSource,
    name="mapFig_circle",
    size=2,
)

# ==================================
# == callbacks: add interactivity ==

def updateDistCDS(selectionName):
    # 1) re-create ColumnDataSource for dist & map figure
    distData = pd.DataFrame()
    distData["value"] = df[selectionName]
    distData["rank"] = get_rank(df[selectionName])
    distData["x_coord"] = df["x_coord"]
    distData["y_coord"] = df["y_coord"]
    # 2) update data
    mapFig.select("mapFig_circle").data_source.data = distData
    distFig.select("distFig_circle").data_source.data = distData
    # 3) update titles
    pass


# trigger a callback when a row on the table is selected 
def tableSelection_callback(attrname, old, new):
    # 1) get selected row id
    selectionIndex = tableSource.selected.indices[0]
    # 2) translate to column name
    selectionName = tableSource.data["index"][selectionIndex]
    # 3) call functio to update plots
    updateDistCDS(selectionName)

tableSource.selected.on_change("indices", tableSelection_callback)
    


#==========================
#=== create layout ========
bokehLayout = row(data_table,distFig,mapFig)


# add layout to curDoc (HTML)
curdoc().add_root(bokehLayout)


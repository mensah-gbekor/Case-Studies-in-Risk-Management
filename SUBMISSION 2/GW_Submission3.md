# Submission 3

### Case Studies in Risk Management

Note: The code produced in this research is an adaptation of the code used in Module 3 to 5 of Case Studies in Risk Management. All the definitions of the indicators used in the research were gathered from World Bank Data.

__The plot analysis is produced in the PDF file.__


```python
# Must be run with Jupyter Notebook
# Import Libraries

import os
import pickle
from functools import reduce
from operator import mul

import pandas as pd
import numpy as np

import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from sklearn import linear_model
from sklearn.decomposition import PCA

from scipy.optimize import fsolve
from scipy.stats import iqr
from scipy import signal

import holoviews as hv
import hvplot
import hvplot.pandas

from sklearn import preprocessing

np.random.seed(42)

pd.core.common.is_list_like = pd.api.types.is_list_like
import pandas_datareader.wb as wb

hv.extension('bokeh')


```















<div class="logo-block">
<img src='data:image/png;base64,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
roTBs2RqAAAAAElFTkSuQmCC'
     style='height:25px; border-radius:12px; display: inline-block; float: left; vertical-align: middle'></img>


  <img src='data:image/png;base64,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'
       style='height:15px; border-radius:12px; display: inline-block; float: left'></img>





</div>




```python
# We downloaded the availible countries and 
#  looked up the iso number of the countries we are interested in
countries = wb.get_countries()

countries_used = countries.loc[countries.iso3c.str.contains('ZAF'),:]

countries_used
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>iso3c</th>
      <th>iso2c</th>
      <th>name</th>
      <th>region</th>
      <th>adminregion</th>
      <th>incomeLevel</th>
      <th>lendingType</th>
      <th>capitalCity</th>
      <th>longitude</th>
      <th>latitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>301</th>
      <td>ZAF</td>
      <td>ZA</td>
      <td>South Africa</td>
      <td>Sub-Saharan Africa</td>
      <td>Sub-Saharan Africa (excluding high income)</td>
      <td>Upper middle income</td>
      <td>IBRD</td>
      <td>Pretoria</td>
      <td>28.1871</td>
      <td>-25.746</td>
    </tr>
  </tbody>
</table>
</div>




```python
# We download and select our desired indicators
indicators = wb.get_indicators()

indicators.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>name</th>
      <th>unit</th>
      <th>source</th>
      <th>sourceNote</th>
      <th>sourceOrganization</th>
      <th>topics</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0.HCount.1.90usd</td>
      <td>Poverty Headcount ($1.90 a day)</td>
      <td></td>
      <td>LAC Equity Lab</td>
      <td>The poverty headcount index measures the propo...</td>
      <td>b'LAC Equity Lab tabulations of SEDLAC (CEDLAS...</td>
      <td>Poverty</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0.HCount.2.5usd</td>
      <td>Poverty Headcount ($2.50 a day)</td>
      <td></td>
      <td>LAC Equity Lab</td>
      <td>The poverty headcount index measures the propo...</td>
      <td>b'LAC Equity Lab tabulations of SEDLAC (CEDLAS...</td>
      <td>Poverty</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0.HCount.Mid10to50</td>
      <td>Middle Class ($10-50 a day) Headcount</td>
      <td></td>
      <td>LAC Equity Lab</td>
      <td>The poverty headcount index measures the propo...</td>
      <td>b'LAC Equity Lab tabulations of SEDLAC (CEDLAS...</td>
      <td>Poverty</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0.HCount.Ofcl</td>
      <td>Official Moderate Poverty Rate-National</td>
      <td></td>
      <td>LAC Equity Lab</td>
      <td>The poverty headcount index measures the propo...</td>
      <td>b'LAC Equity Lab tabulations of data from Nati...</td>
      <td>Poverty</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.0.HCount.Poor4uds</td>
      <td>Poverty Headcount ($4 a day)</td>
      <td></td>
      <td>LAC Equity Lab</td>
      <td>The poverty headcount index measures the propo...</td>
      <td>b'LAC Equity Lab tabulations of SEDLAC (CEDLAS...</td>
      <td>Poverty</td>
    </tr>
  </tbody>
</table>
</div>




```python
#  tranforming our data easier and more consistent
class Indicators:
    def __init__(self, country=['ZAF'], start='1994-01-01 00:00:00', end='1994-01-01 00:00:00'):
        # This class that initialization parameters for use in all downloads
        self.country = country
        self.start = start
        self.end = end
        self.indicator = None
    
    def get_indicator(self, indicator='CM.MKT.TRAD.CD', long=True, detrend=False, scale=True, interpolate=True):
        # We download the data
        self.indicator = wb.download(indicator=indicator, 
                                    country=self.country, 
                                    start=self.start, 
                                    end=self.end)        
        # reshape it
        self.indicator = self.indicator.reset_index().iloc[::-1,:].reset_index(drop=True) if long else self.indicator
        self.indicator = self.indicator.rename(index=str, columns={indicator: indicators.loc[indicators.id == indicator,'name'].tolist()[0]})
        indicator = indicators.loc[indicators.id == indicator,'name'].tolist()[0]
        
        # and if required, interpolate missing data
        self.indicator.loc[:,indicator] = self.indicator.groupby('country')[indicator]\
        .apply(lambda x: pd.Series(x).interpolate()) if interpolate else self.indicator
        
        # detrend the data
        self.indicator.loc[:,indicator] = self.indicator.groupby('country')[indicator]\
        .apply(lambda x: pd.Series(signal.detrend(x))).reset_index().iloc[:,2].values if detrend else self.indicator
        
        # scale the data
        self.indicator.loc[:,indicator] = self.indicator.groupby('country')[indicator]\
        .apply(lambda x: (x-x.iloc[0])/iqr(x)) if scale else self.indicator
        
        # convert the data for easier use
        self.indicator.year = pd.to_numeric(self.indicator.year)
        
        return self.indicator
```


```python
# we initialize our class with our countries and dates
Country_Indicators = Indicators(country=countries_used.iso3c.tolist(), start=pd.to_datetime('1994', yearfirst=True), end=pd.to_datetime('2009', yearfirst=True))

```


```python
%%opts Curve [width=400, height=350]
investment = Indicators(country=countries_used.iso3c.tolist(), 
                        start=pd.to_datetime('1994', yearfirst=True), 
                        end=pd.to_datetime('2009', yearfirst=True))\
                        .get_indicator(indicator='BX.KLT.DINV.CD.WD', scale=False)
interest = Country_Indicators.get_indicator(indicator='FR.INR.RINR', scale=False)
gdp = Country_Indicators.get_indicator(indicator='NY.GDP.PCAP.CD', scale=False)
money = Country_Indicators.get_indicator(indicator='FM.LBL.BMNY.GD.ZS', scale=False)

investment_plot = investment.hvplot.line(x='year', by='country', title='Foreign Investment')
interest_plot = interest.hvplot.line(x='year', by='country', title='Real Interest Rate (%)')
gdp_plot = gdp.hvplot.line(x='year', by='country', title='GDP per Capita')
money_plot = money.hvplot.line(x='year', by='country', title='Broad Money')
```


```python
investment_plot
```






<div id='1001'>





  <div class="bk-root" id="07366b98-6c64-4a99-8c29-1af3048202ef" data-root-id="1001"></div>
</div>
<script type="application/javascript">(function(root) {
  function embed_document(root) {
  var docs_json = {"d7e9f48f-1fb4-448f-b260-a4b9097aec4c":{"roots":{"references":[{"attributes":{},"id":"1016","type":"BasicTicker"},{"attributes":{},"id":"1041","type":"Selection"},{"attributes":{"axis_label":"Foreign direct investment, net inflows (BoP, current US$)","bounds":"auto","formatter":{"id":"1038"},"major_label_orientation":"horizontal","ticker":{"id":"1020"}},"id":"1019","type":"LinearAxis"},{"attributes":{"callback":null,"renderers":[{"id":"1046"}],"tags":["hv_created"],"tooltips":[["country","@{country}"],["year","@{year}"],["Foreign direct investment, net inflows (BoP, current US$)","@{Foreign_direct_investment_comma_net_inflows_left_parenthesis_BoP_comma_current_US_right_parenthesis}"]]},"id":"1005","type":"HoverTool"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"1028","type":"BoxAnnotation"},{"attributes":{},"id":"1020","type":"BasicTicker"},{"attributes":{"source":{"id":"1040"}},"id":"1047","type":"CDSView"},{"attributes":{"line_alpha":0.2,"line_color":"#1f77b3","line_width":2,"x":{"field":"year"},"y":{"field":"Foreign direct investment, net inflows (BoP, current US$)"}},"id":"1045","type":"Line"},{"attributes":{"end":10836060378.680498,"reset_end":10836060378.680498,"reset_start":-576648644.1625104,"start":-576648644.1625104,"tags":[[["Foreign direct investment, net inflows (BoP, current US$)","Foreign direct investment, net inflows (BoP, current US$)",null]]]},"id":"1004","type":"Range1d"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"1005"},{"id":"1023"},{"id":"1024"},{"id":"1025"},{"id":"1026"},{"id":"1027"}]},"id":"1029","type":"Toolbar"},{"attributes":{"axis":{"id":"1015"},"grid_line_color":null,"ticker":null},"id":"1018","type":"Grid"},{"attributes":{},"id":"1055","type":"UnionRenderers"},{"attributes":{"children":[{"id":"1002"},{"id":"1006"},{"id":"1130"}],"margin":[0,0,0,0],"name":"Row02490","tags":["embedded"]},"id":"1001","type":"Row"},{"attributes":{},"id":"1011","type":"LinearScale"},{"attributes":{"items":[]},"id":"1056","type":"Legend"},{"attributes":{"text":"Foreign Investment","text_color":{"value":"black"},"text_font_size":{"value":"12pt"}},"id":"1007","type":"Title"},{"attributes":{},"id":"1027","type":"ResetTool"},{"attributes":{"line_color":"#1f77b3","line_width":2,"x":{"field":"year"},"y":{"field":"Foreign direct investment, net inflows (BoP, current US$)"}},"id":"1043","type":"Line"},{"attributes":{"below":[{"id":"1015"}],"center":[{"id":"1018"},{"id":"1022"},{"id":"1056"}],"left":[{"id":"1019"}],"margin":null,"min_border_bottom":10,"min_border_left":10,"min_border_right":10,"min_border_top":10,"plot_height":300,"plot_width":700,"renderers":[{"id":"1046"}],"sizing_mode":"fixed","title":{"id":"1007"},"toolbar":{"id":"1029"},"x_range":{"id":"1003"},"x_scale":{"id":"1011"},"y_range":{"id":"1004"},"y_scale":{"id":"1013"}},"id":"1006","subtype":"Figure","type":"Plot"},{"attributes":{"end":2009.0,"reset_end":2009.0,"reset_start":1994.0,"start":1994.0,"tags":[[["year","year",null]]]},"id":"1003","type":"Range1d"},{"attributes":{},"id":"1025","type":"WheelZoomTool"},{"attributes":{},"id":"1036","type":"BasicTickFormatter"},{"attributes":{"margin":[5,5,5,5],"name":"HSpacer02494","sizing_mode":"stretch_width"},"id":"1002","type":"Spacer"},{"attributes":{},"id":"1013","type":"LinearScale"},{"attributes":{"line_alpha":0.1,"line_color":"#1f77b3","line_width":2,"x":{"field":"year"},"y":{"field":"Foreign direct investment, net inflows (BoP, current US$)"}},"id":"1044","type":"Line"},{"attributes":{},"id":"1024","type":"PanTool"},{"attributes":{"axis":{"id":"1019"},"dimension":1,"grid_line_color":null,"ticker":null},"id":"1022","type":"Grid"},{"attributes":{"data_source":{"id":"1040"},"glyph":{"id":"1043"},"hover_glyph":null,"muted_glyph":{"id":"1045"},"nonselection_glyph":{"id":"1044"},"selection_glyph":null,"view":{"id":"1047"}},"id":"1046","type":"GlyphRenderer"},{"attributes":{"data":{"Foreign direct investment, net inflows (BoP, current US$)":{"__ndarray__":"VgwTyQxRtkHF/i75XJrSQeax4EyOVMhBuw9hJglk7EFOZPwRwGbAQWg/mhnCZtZBfsP6vZjfzEH5oKeRihX7QTElMKsBDdZBuwohrtpWx0GZnNDbbOfEQV7jIiD1S/hBf/grsFaTwkFVwdGTp4n4QXSMazKJaQJCORxev0dn/EE=","dtype":"float64","order":"little","shape":[16]},"Foreign_direct_investment_comma_net_inflows_left_parenthesis_BoP_comma_current_US_right_parenthesis":{"__ndarray__":"VgwTyQxRtkHF/i75XJrSQeax4EyOVMhBuw9hJglk7EFOZPwRwGbAQWg/mhnCZtZBfsP6vZjfzEH5oKeRihX7QTElMKsBDdZBuwohrtpWx0GZnNDbbOfEQV7jIiD1S/hBf/grsFaTwkFVwdGTp4n4QXSMazKJaQJCORxev0dn/EE=","dtype":"float64","order":"little","shape":[16]},"country":["South Africa","South Africa","South Africa","South Africa","South Africa","South Africa","South Africa","South Africa","South Africa","South Africa","South Africa","South Africa","South Africa","South Africa","South Africa","South Africa"],"year":[1994,1995,1996,1997,1998,1999,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009]},"selected":{"id":"1041"},"selection_policy":{"id":"1055"}},"id":"1040","type":"ColumnDataSource"},{"attributes":{"axis_label":"year","bounds":"auto","formatter":{"id":"1036"},"major_label_orientation":"horizontal","ticker":{"id":"1016"}},"id":"1015","type":"LinearAxis"},{"attributes":{},"id":"1038","type":"BasicTickFormatter"},{"attributes":{"margin":[5,5,5,5],"name":"HSpacer02495","sizing_mode":"stretch_width"},"id":"1130","type":"Spacer"},{"attributes":{},"id":"1023","type":"SaveTool"},{"attributes":{"overlay":{"id":"1028"}},"id":"1026","type":"BoxZoomTool"}],"root_ids":["1001"]},"title":"Bokeh Application","version":"2.2.1"}};
  var render_items = [{"docid":"d7e9f48f-1fb4-448f-b260-a4b9097aec4c","root_ids":["1001"],"roots":{"1001":"07366b98-6c64-4a99-8c29-1af3048202ef"}}];
  root.Bokeh.embed.embed_items_notebook(docs_json, render_items);
  }
if (root.Bokeh !== undefined) {
    embed_document(root);
  } else {
    var attempts = 0;
    var timer = setInterval(function(root) {
      if (root.Bokeh !== undefined) {
        clearInterval(timer);
        embed_document(root);
      } else if (document.readyState == "complete") {
        attempts++;
        if (attempts > 100) {
          clearInterval(timer);
          console.log("Bokeh: ERROR: Unable to run BokehJS code because BokehJS library is missing");
        }
      }
    }, 10, root)
  }
})(window);</script>




```python
interest_plot
```






<div id='1172'>





  <div class="bk-root" id="091439f9-04fe-4d0b-b3a1-695501347591" data-root-id="1172"></div>
</div>
<script type="application/javascript">(function(root) {
  function embed_document(root) {
  var docs_json = {"e31088f1-8ad7-4a0b-be6a-c5428742a059":{"roots":{"references":[{"attributes":{"end":2009.0,"reset_end":2009.0,"reset_start":1994.0,"start":1994.0,"tags":[[["year","year",null]]]},"id":"1174","type":"Range1d"},{"attributes":{},"id":"1196","type":"WheelZoomTool"},{"attributes":{},"id":"1182","type":"LinearScale"},{"attributes":{},"id":"1207","type":"BasicTickFormatter"},{"attributes":{"line_alpha":0.2,"line_color":"#1f77b3","line_width":2,"x":{"field":"year"},"y":{"field":"Real interest rate (%)"}},"id":"1216","type":"Line"},{"attributes":{"axis":{"id":"1190"},"dimension":1,"grid_line_color":null,"ticker":null},"id":"1193","type":"Grid"},{"attributes":{"source":{"id":"1211"}},"id":"1218","type":"CDSView"},{"attributes":{"end":13.976297888256994,"reset_end":13.976297888256994,"reset_start":2.175756326573877,"start":2.175756326573877,"tags":[[["Real interest rate (%)","Real interest rate (%)",null]]]},"id":"1175","type":"Range1d"},{"attributes":{"callback":null,"renderers":[{"id":"1217"}],"tags":["hv_created"],"tooltips":[["country","@{country}"],["year","@{year}"],["Real interest rate (%)","@{Real_interest_rate_left_parenthesis_percent_right_parenthesis}"]]},"id":"1176","type":"HoverTool"},{"attributes":{},"id":"1187","type":"BasicTicker"},{"attributes":{"data_source":{"id":"1211"},"glyph":{"id":"1214"},"hover_glyph":null,"muted_glyph":{"id":"1216"},"nonselection_glyph":{"id":"1215"},"selection_glyph":null,"view":{"id":"1218"}},"id":"1217","type":"GlyphRenderer"},{"attributes":{"margin":[5,5,5,5],"name":"HSpacer02707","sizing_mode":"stretch_width"},"id":"1301","type":"Spacer"},{"attributes":{"overlay":{"id":"1199"}},"id":"1197","type":"BoxZoomTool"},{"attributes":{},"id":"1209","type":"BasicTickFormatter"},{"attributes":{},"id":"1191","type":"BasicTicker"},{"attributes":{},"id":"1212","type":"Selection"},{"attributes":{"line_alpha":0.1,"line_color":"#1f77b3","line_width":2,"x":{"field":"year"},"y":{"field":"Real interest rate (%)"}},"id":"1215","type":"Line"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"1199","type":"BoxAnnotation"},{"attributes":{},"id":"1198","type":"ResetTool"},{"attributes":{"text":"Real Interest Rate (%)","text_color":{"value":"black"},"text_font_size":{"value":"12pt"}},"id":"1178","type":"Title"},{"attributes":{"items":[]},"id":"1227","type":"Legend"},{"attributes":{},"id":"1195","type":"PanTool"},{"attributes":{"data":{"Real interest rate (%)":{"__ndarray__":"cDsGYKj8FUBIko1KueEbQAe8vrVBhyVAJv2V/JU/JkADBFLvX/wpQChfM6mygCRAMXAfgV74FEBjQ4JlgsYWQBZH93XoRQlANVSJEmRTIUCxGb8CDeQRQN8UcNU7ohNAN9brkTd9EkBDOzDq7roPQF9SYR+SIRdAln6+N2pID0A=","dtype":"float64","order":"little","shape":[16]},"Real_interest_rate_left_parenthesis_percent_right_parenthesis":{"__ndarray__":"cDsGYKj8FUBIko1KueEbQAe8vrVBhyVAJv2V/JU/JkADBFLvX/wpQChfM6mygCRAMXAfgV74FEBjQ4JlgsYWQBZH93XoRQlANVSJEmRTIUCxGb8CDeQRQN8UcNU7ohNAN9brkTd9EkBDOzDq7roPQF9SYR+SIRdAln6+N2pID0A=","dtype":"float64","order":"little","shape":[16]},"country":["South Africa","South Africa","South Africa","South Africa","South Africa","South Africa","South Africa","South Africa","South Africa","South Africa","South Africa","South Africa","South Africa","South Africa","South Africa","South Africa"],"year":[1994,1995,1996,1997,1998,1999,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009]},"selected":{"id":"1212"},"selection_policy":{"id":"1226"}},"id":"1211","type":"ColumnDataSource"},{"attributes":{"axis_label":"Real interest rate (%)","bounds":"auto","formatter":{"id":"1209"},"major_label_orientation":"horizontal","ticker":{"id":"1191"}},"id":"1190","type":"LinearAxis"},{"attributes":{"line_color":"#1f77b3","line_width":2,"x":{"field":"year"},"y":{"field":"Real interest rate (%)"}},"id":"1214","type":"Line"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"1176"},{"id":"1194"},{"id":"1195"},{"id":"1196"},{"id":"1197"},{"id":"1198"}]},"id":"1200","type":"Toolbar"},{"attributes":{"axis":{"id":"1186"},"grid_line_color":null,"ticker":null},"id":"1189","type":"Grid"},{"attributes":{"below":[{"id":"1186"}],"center":[{"id":"1189"},{"id":"1193"},{"id":"1227"}],"left":[{"id":"1190"}],"margin":null,"min_border_bottom":10,"min_border_left":10,"min_border_right":10,"min_border_top":10,"plot_height":300,"plot_width":700,"renderers":[{"id":"1217"}],"sizing_mode":"fixed","title":{"id":"1178"},"toolbar":{"id":"1200"},"x_range":{"id":"1174"},"x_scale":{"id":"1182"},"y_range":{"id":"1175"},"y_scale":{"id":"1184"}},"id":"1177","subtype":"Figure","type":"Plot"},{"attributes":{"children":[{"id":"1173"},{"id":"1177"},{"id":"1301"}],"margin":[0,0,0,0],"name":"Row02702","tags":["embedded"]},"id":"1172","type":"Row"},{"attributes":{},"id":"1194","type":"SaveTool"},{"attributes":{},"id":"1184","type":"LinearScale"},{"attributes":{"margin":[5,5,5,5],"name":"HSpacer02706","sizing_mode":"stretch_width"},"id":"1173","type":"Spacer"},{"attributes":{},"id":"1226","type":"UnionRenderers"},{"attributes":{"axis_label":"year","bounds":"auto","formatter":{"id":"1207"},"major_label_orientation":"horizontal","ticker":{"id":"1187"}},"id":"1186","type":"LinearAxis"}],"root_ids":["1172"]},"title":"Bokeh Application","version":"2.2.1"}};
  var render_items = [{"docid":"e31088f1-8ad7-4a0b-be6a-c5428742a059","root_ids":["1172"],"roots":{"1172":"091439f9-04fe-4d0b-b3a1-695501347591"}}];
  root.Bokeh.embed.embed_items_notebook(docs_json, render_items);
  }
if (root.Bokeh !== undefined) {
    embed_document(root);
  } else {
    var attempts = 0;
    var timer = setInterval(function(root) {
      if (root.Bokeh !== undefined) {
        clearInterval(timer);
        embed_document(root);
      } else if (document.readyState == "complete") {
        attempts++;
        if (attempts > 100) {
          clearInterval(timer);
          console.log("Bokeh: ERROR: Unable to run BokehJS code because BokehJS library is missing");
        }
      }
    }, 10, root)
  }
})(window);</script>




```python
gdp_plot
```






<div id='1343'>





  <div class="bk-root" id="9eedc115-39c0-4563-a676-96b40f805cc0" data-root-id="1343"></div>
</div>
<script type="application/javascript">(function(root) {
  function embed_document(root) {
  var docs_json = {"0655b42b-25e8-44be-a1b3-ba9f85e4a8b5":{"roots":{"references":[{"attributes":{"end":2009.0,"reset_end":2009.0,"reset_start":1994.0,"start":1994.0,"tags":[[["year","year",null]]]},"id":"1345","type":"Range1d"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"1347"},{"id":"1365"},{"id":"1366"},{"id":"1367"},{"id":"1368"},{"id":"1369"}]},"id":"1371","type":"Toolbar"},{"attributes":{"line_color":"#1f77b3","line_width":2,"x":{"field":"year"},"y":{"field":"GDP per capita (current US$)"}},"id":"1385","type":"Line"},{"attributes":{"end":6454.956982291917,"reset_end":6454.956982291917,"reset_start":2142.9424612316843,"start":2142.9424612316843,"tags":[[["GDP per capita (current US$)","GDP per capita (current US$)",null]]]},"id":"1346","type":"Range1d"},{"attributes":{"data_source":{"id":"1382"},"glyph":{"id":"1385"},"hover_glyph":null,"muted_glyph":{"id":"1387"},"nonselection_glyph":{"id":"1386"},"selection_glyph":null,"view":{"id":"1389"}},"id":"1388","type":"GlyphRenderer"},{"attributes":{},"id":"1355","type":"LinearScale"},{"attributes":{},"id":"1369","type":"ResetTool"},{"attributes":{},"id":"1383","type":"Selection"},{"attributes":{"callback":null,"renderers":[{"id":"1388"}],"tags":["hv_created"],"tooltips":[["country","@{country}"],["year","@{year}"],["GDP per capita (current US$)","@{GDP_per_capita_left_parenthesis_current_US_right_parenthesis}"]]},"id":"1347","type":"HoverTool"},{"attributes":{},"id":"1353","type":"LinearScale"},{"attributes":{},"id":"1378","type":"BasicTickFormatter"},{"attributes":{"text":"GDP per Capita","text_color":{"value":"black"},"text_font_size":{"value":"12pt"}},"id":"1349","type":"Title"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"1370","type":"BoxAnnotation"},{"attributes":{"margin":[5,5,5,5],"name":"HSpacer02919","sizing_mode":"stretch_width"},"id":"1472","type":"Spacer"},{"attributes":{"line_alpha":0.2,"line_color":"#1f77b3","line_width":2,"x":{"field":"year"},"y":{"field":"GDP per capita (current US$)"}},"id":"1387","type":"Line"},{"attributes":{"source":{"id":"1382"}},"id":"1389","type":"CDSView"},{"attributes":{"axis_label":"GDP per capita (current US$)","bounds":"auto","formatter":{"id":"1380"},"major_label_orientation":"horizontal","ticker":{"id":"1362"}},"id":"1361","type":"LinearAxis"},{"attributes":{"items":[]},"id":"1398","type":"Legend"},{"attributes":{"data":{"GDP per capita (current US$)":{"__ndarray__":"xZ16y3XqqkDQHn5qtU+tQHNd1mLETKtADH2G3Si7q0D4LzujCqSoQGRVcM0jE6hAbtfbsdqwp0CK4WYn89SkQCHOjdONjKNA83/gzJBOrUBdqe2koOGyQDKAKxOoB7VAf0vE0wLitUBmXydYn8+3QHjwTSfOgLZA6fR8HszmtkA=","dtype":"float64","order":"little","shape":[16]},"GDP_per_capita_left_parenthesis_current_US_right_parenthesis":{"__ndarray__":"xZ16y3XqqkDQHn5qtU+tQHNd1mLETKtADH2G3Si7q0D4LzujCqSoQGRVcM0jE6hAbtfbsdqwp0CK4WYn89SkQCHOjdONjKNA83/gzJBOrUBdqe2koOGyQDKAKxOoB7VAf0vE0wLitUBmXydYn8+3QHjwTSfOgLZA6fR8HszmtkA=","dtype":"float64","order":"little","shape":[16]},"country":["South Africa","South Africa","South Africa","South Africa","South Africa","South Africa","South Africa","South Africa","South Africa","South Africa","South Africa","South Africa","South Africa","South Africa","South Africa","South Africa"],"year":[1994,1995,1996,1997,1998,1999,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009]},"selected":{"id":"1383"},"selection_policy":{"id":"1397"}},"id":"1382","type":"ColumnDataSource"},{"attributes":{"axis":{"id":"1361"},"dimension":1,"grid_line_color":null,"ticker":null},"id":"1364","type":"Grid"},{"attributes":{"below":[{"id":"1357"}],"center":[{"id":"1360"},{"id":"1364"},{"id":"1398"}],"left":[{"id":"1361"}],"margin":null,"min_border_bottom":10,"min_border_left":10,"min_border_right":10,"min_border_top":10,"plot_height":300,"plot_width":700,"renderers":[{"id":"1388"}],"sizing_mode":"fixed","title":{"id":"1349"},"toolbar":{"id":"1371"},"x_range":{"id":"1345"},"x_scale":{"id":"1353"},"y_range":{"id":"1346"},"y_scale":{"id":"1355"}},"id":"1348","subtype":"Figure","type":"Plot"},{"attributes":{},"id":"1362","type":"BasicTicker"},{"attributes":{},"id":"1397","type":"UnionRenderers"},{"attributes":{},"id":"1366","type":"PanTool"},{"attributes":{"children":[{"id":"1344"},{"id":"1348"},{"id":"1472"}],"margin":[0,0,0,0],"name":"Row02914","tags":["embedded"]},"id":"1343","type":"Row"},{"attributes":{},"id":"1365","type":"SaveTool"},{"attributes":{"margin":[5,5,5,5],"name":"HSpacer02918","sizing_mode":"stretch_width"},"id":"1344","type":"Spacer"},{"attributes":{},"id":"1380","type":"BasicTickFormatter"},{"attributes":{},"id":"1358","type":"BasicTicker"},{"attributes":{"overlay":{"id":"1370"}},"id":"1368","type":"BoxZoomTool"},{"attributes":{"line_alpha":0.1,"line_color":"#1f77b3","line_width":2,"x":{"field":"year"},"y":{"field":"GDP per capita (current US$)"}},"id":"1386","type":"Line"},{"attributes":{},"id":"1367","type":"WheelZoomTool"},{"attributes":{"axis":{"id":"1357"},"grid_line_color":null,"ticker":null},"id":"1360","type":"Grid"},{"attributes":{"axis_label":"year","bounds":"auto","formatter":{"id":"1378"},"major_label_orientation":"horizontal","ticker":{"id":"1358"}},"id":"1357","type":"LinearAxis"}],"root_ids":["1343"]},"title":"Bokeh Application","version":"2.2.1"}};
  var render_items = [{"docid":"0655b42b-25e8-44be-a1b3-ba9f85e4a8b5","root_ids":["1343"],"roots":{"1343":"9eedc115-39c0-4563-a676-96b40f805cc0"}}];
  root.Bokeh.embed.embed_items_notebook(docs_json, render_items);
  }
if (root.Bokeh !== undefined) {
    embed_document(root);
  } else {
    var attempts = 0;
    var timer = setInterval(function(root) {
      if (root.Bokeh !== undefined) {
        clearInterval(timer);
        embed_document(root);
      } else if (document.readyState == "complete") {
        attempts++;
        if (attempts > 100) {
          clearInterval(timer);
          console.log("Bokeh: ERROR: Unable to run BokehJS code because BokehJS library is missing");
        }
      }
    }, 10, root)
  }
})(window);</script>




```python
money_plot
```






<div id='1514'>





  <div class="bk-root" id="1694787b-d5e1-47a5-aba2-13dd3ec04f25" data-root-id="1514"></div>
</div>
<script type="application/javascript">(function(root) {
  function embed_document(root) {
  var docs_json = {"79795c55-139f-4f6b-aa8f-58f7af708e57":{"roots":{"references":[{"attributes":{"data":{"Broad money (% of GDP)":{"__ndarray__":"2OE6NmDOR0AjGcw3Kk9IQJ/Ovg0Cr0hArZKbQ0g/SkAb1pSBtYlLQLVV6jQA3ktA3Fi8hPFaSkDw5p9rZKdMQFvaBif+IE1AlYT/rslQTkClMzeaaMxOQMpogk0VvlBAGMfXqthLUkBrdI06gMVTQEdbgl0xM1RAq1Z/zGJrU0A=","dtype":"float64","order":"little","shape":[16]},"Broad_money_left_parenthesis_percent_of_GDP_right_parenthesis":{"__ndarray__":"2OE6NmDOR0AjGcw3Kk9IQJ/Ovg0Cr0hArZKbQ0g/SkAb1pSBtYlLQLVV6jQA3ktA3Fi8hPFaSkDw5p9rZKdMQFvaBif+IE1AlYT/rslQTkClMzeaaMxOQMpogk0VvlBAGMfXqthLUkBrdI06gMVTQEdbgl0xM1RAq1Z/zGJrU0A=","dtype":"float64","order":"little","shape":[16]},"country":["South Africa","South Africa","South Africa","South Africa","South Africa","South Africa","South Africa","South Africa","South Africa","South Africa","South Africa","South Africa","South Africa","South Africa","South Africa","South Africa"],"year":[1994,1995,1996,1997,1998,1999,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009]},"selected":{"id":"1554"},"selection_policy":{"id":"1568"}},"id":"1553","type":"ColumnDataSource"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"1541","type":"BoxAnnotation"},{"attributes":{"line_color":"#1f77b3","line_width":2,"x":{"field":"year"},"y":{"field":"Broad money (% of GDP)"}},"id":"1556","type":"Line"},{"attributes":{"below":[{"id":"1528"}],"center":[{"id":"1531"},{"id":"1535"},{"id":"1569"}],"left":[{"id":"1532"}],"margin":null,"min_border_bottom":10,"min_border_left":10,"min_border_right":10,"min_border_top":10,"plot_height":300,"plot_width":700,"renderers":[{"id":"1559"}],"sizing_mode":"fixed","title":{"id":"1520"},"toolbar":{"id":"1542"},"x_range":{"id":"1516"},"x_scale":{"id":"1524"},"y_range":{"id":"1517"},"y_scale":{"id":"1526"}},"id":"1519","subtype":"Figure","type":"Plot"},{"attributes":{},"id":"1568","type":"UnionRenderers"},{"attributes":{"axis":{"id":"1528"},"grid_line_color":null,"ticker":null},"id":"1531","type":"Grid"},{"attributes":{"items":[]},"id":"1569","type":"Legend"},{"attributes":{},"id":"1538","type":"WheelZoomTool"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"1518"},{"id":"1536"},{"id":"1537"},{"id":"1538"},{"id":"1539"},{"id":"1540"}]},"id":"1542","type":"Toolbar"},{"attributes":{"margin":[5,5,5,5],"name":"HSpacer03131","sizing_mode":"stretch_width"},"id":"1643","type":"Spacer"},{"attributes":{"axis_label":"Broad money (% of GDP)","bounds":"auto","formatter":{"id":"1551"},"major_label_orientation":"horizontal","ticker":{"id":"1533"}},"id":"1532","type":"LinearAxis"},{"attributes":{"axis":{"id":"1532"},"dimension":1,"grid_line_color":null,"ticker":null},"id":"1535","type":"Grid"},{"attributes":{},"id":"1533","type":"BasicTicker"},{"attributes":{"axis_label":"year","bounds":"auto","formatter":{"id":"1549"},"major_label_orientation":"horizontal","ticker":{"id":"1529"}},"id":"1528","type":"LinearAxis"},{"attributes":{"data_source":{"id":"1553"},"glyph":{"id":"1556"},"hover_glyph":null,"muted_glyph":{"id":"1558"},"nonselection_glyph":{"id":"1557"},"selection_glyph":null,"view":{"id":"1560"}},"id":"1559","type":"GlyphRenderer"},{"attributes":{},"id":"1536","type":"SaveTool"},{"attributes":{"text":"Broad Money","text_color":{"value":"black"},"text_font_size":{"value":"12pt"}},"id":"1520","type":"Title"},{"attributes":{"end":84.11864570346484,"reset_end":84.11864570346484,"reset_start":44.29355346574407,"start":44.29355346574407,"tags":[[["Broad money (% of GDP)","Broad money (% of GDP)",null]]]},"id":"1517","type":"Range1d"},{"attributes":{"line_alpha":0.1,"line_color":"#1f77b3","line_width":2,"x":{"field":"year"},"y":{"field":"Broad money (% of GDP)"}},"id":"1557","type":"Line"},{"attributes":{},"id":"1529","type":"BasicTicker"},{"attributes":{"children":[{"id":"1515"},{"id":"1519"},{"id":"1643"}],"margin":[0,0,0,0],"name":"Row03126","tags":["embedded"]},"id":"1514","type":"Row"},{"attributes":{"source":{"id":"1553"}},"id":"1560","type":"CDSView"},{"attributes":{},"id":"1524","type":"LinearScale"},{"attributes":{"line_alpha":0.2,"line_color":"#1f77b3","line_width":2,"x":{"field":"year"},"y":{"field":"Broad money (% of GDP)"}},"id":"1558","type":"Line"},{"attributes":{},"id":"1551","type":"BasicTickFormatter"},{"attributes":{"margin":[5,5,5,5],"name":"HSpacer03130","sizing_mode":"stretch_width"},"id":"1515","type":"Spacer"},{"attributes":{},"id":"1554","type":"Selection"},{"attributes":{"callback":null,"renderers":[{"id":"1559"}],"tags":["hv_created"],"tooltips":[["country","@{country}"],["year","@{year}"],["Broad money (% of GDP)","@{Broad_money_left_parenthesis_percent_of_GDP_right_parenthesis}"]]},"id":"1518","type":"HoverTool"},{"attributes":{},"id":"1549","type":"BasicTickFormatter"},{"attributes":{"end":2009.0,"reset_end":2009.0,"reset_start":1994.0,"start":1994.0,"tags":[[["year","year",null]]]},"id":"1516","type":"Range1d"},{"attributes":{},"id":"1537","type":"PanTool"},{"attributes":{"overlay":{"id":"1541"}},"id":"1539","type":"BoxZoomTool"},{"attributes":{},"id":"1540","type":"ResetTool"},{"attributes":{},"id":"1526","type":"LinearScale"}],"root_ids":["1514"]},"title":"Bokeh Application","version":"2.2.1"}};
  var render_items = [{"docid":"79795c55-139f-4f6b-aa8f-58f7af708e57","root_ids":["1514"],"roots":{"1514":"1694787b-d5e1-47a5-aba2-13dd3ec04f25"}}];
  root.Bokeh.embed.embed_items_notebook(docs_json, render_items);
  }
if (root.Bokeh !== undefined) {
    embed_document(root);
  } else {
    var attempts = 0;
    var timer = setInterval(function(root) {
      if (root.Bokeh !== undefined) {
        clearInterval(timer);
        embed_document(root);
      } else if (document.readyState == "complete") {
        attempts++;
        if (attempts > 100) {
          clearInterval(timer);
          console.log("Bokeh: ERROR: Unable to run BokehJS code because BokehJS library is missing");
        }
      }
    }, 10, root)
  }
})(window);</script>



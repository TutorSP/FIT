# FIT
FIT: Forecasting Intensity and Track of Tropical Cyclones in Bay of Bengal using Deep Learning on INSAT-3D Satellite Images
<br><br>
<i>Description</i>: Using INSAT-3D satellite images of tropical cyclones in BoB and values of estimated central pressure (ECP) and maximum sustained wind speed (MSW) from RSMC reports by India Meteorological Department (IMD), a 2-way hybrid deep learning model is trained to forecast the intensity and track of TCs simultaneously.
<br><br>
<i>Code files</i>:
<br>
<ul>
  <li>Python file (.py) containing the function definition of the custom loss function based on Euclidean distance for track (latitude and longitude)</li>
  <li>Python file (.py) containing the code for simultaneously forecasting the intensity and track of TCs for a lead time of 3 hrs using 12 hrs of past inputs with a cadence of 30 minutes</li>
  <li>Python file (.py) containing the code for simultaneously forecasting the intensity and track of TCs for a lead time of 6 hrs using 24 hrs of past inputs with a cadence of 30 minutes</li>
  <li>Python file (.py) containing the code for simultaneously forecasting the intensity and track of TCs for a lead time of 6 hrs using 24 hrs of past inputs with a cadence of 1 hr</li>
</ul>
<br>
<i>Data files</i>:
<br>
<ul>
  <li>Directory of BoB TC images of dimensions 64 x 64</li>
  <li>Directory of interpolated numerical data of BoB TCs in the form of CSV files </li>
</ul>

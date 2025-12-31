# FIT
FIT: Forecasting Intensity and Track of Tropical Cyclones in Bay of Bengal using Deep Learning on INSAT-3D Satellite Images
<br><br>
<i>Description</i>: Using INSAT-3D satellite images of tropical cyclones in BoB and values of estimated central pressure (ECP) and maximum sustained wind speed (MSW) from RSMC reports by India Meteorological Department (IMD), a 2-way hybrid deep learning model is trained to forecast the intensity and track of TCs simultaneously.
<br><br>
<i>Code files</i>:
<br>
<ul>
  <li>Python file (.py) containing the custom loss function based on Euclidean distance for track (latitude and longitude); <b>filename=custom_loss.py</b></li>
  <li>Python file (.py) containing the code for simultaneously forecasting the intensity and track of TCs for a lead time of 3 hrs using 12 hrs of past inputs with a cadence of 30 minutes; <b>filename=0.5hr_12hr_3hr.py</b></li>
  <li>Python file (.py) containing the code for simultaneously forecasting the intensity and track of TCs for a lead time of 6 hrs using 24 hrs of past inputs with a cadence of 30 minutes; <b>filename=0.5hr_24hrs_6hrs.py</b></li>
  <li>Python file (.py) containing the code for simultaneously forecasting the intensity and track of TCs for a lead time of 6 hrs using 24 hrs of past inputs with a cadence of 1 hr; <b>filename=1hr_24hrs_6hrs.py</b></li>
</ul>
<br>
<i>Data files</i>:
<br>
<ul>
  <li>Directory of BoB TC images of dimensions 64 x 64; <b>filename= "FIT images.zip"</b></li>
  <li>Directory of interpolated numerical data of BoB TCs in the form of CSV files; <b>filename= "Interpolated CSV files for track prediction.zip"</b></li>
</ul>

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
  <li>Original numerical data of BoB TCs; <b>filename= "Original EXCEL files for track prediction.zip"</b></li>
</ul>
<i>Data preparation</i>:
<br>
The numerical data (LAT, LON, ECP, MSW) have a cadence of 3hrs or 6hrs whereas images are available every 30 minutes. To resolve this inconsistency in the cadence, the numerical variables are independently linearly interpolated. Depending on the input length, output length and cadence, consecutive images are grouped together to form an image data tuple. For instance, if the experiment is the forecast of intensity and track of TCs for a lead time of 3hrs using 12hrs of input and 0.5hr cadence, images of 24 consecutive timestamps are stacked to form one input image data tuple. Similarly, the numerical variables are also stacked together to form one input numerical data tuple. Numerical data of next 6 consecutive timestamps are stacked together to form the corresponding ground truth. If an image data tuple contains one or more blank images, both the image data tuple and the corresponding numerical data tuple are discarded.

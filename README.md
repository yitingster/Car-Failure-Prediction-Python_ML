# Car Failure Prediction 

<h1>Introduction</h1>

<p>
Hello everyone, this is a self-practice python project using pseudo data. Through this project, I want to familiar myself with data cleaning and exploratory data analysis.  

The dataset <b>failure.xslx</b> can be found in the respository. 

There is an interactive python plot that can be view by downloading the script and running it on Jupyter.  
</p>

There will be a second part to this project where a machine learning model would be build to predict car failure (in progress).

<h1>Dataset</h1>

|Field      |Description    |
| ------------- | -------------  |
| Car ID        | Car unique ID  |
| Model  | Car model 3,5 and 7 (in the order of increasing specifications)  |
|Color   | Color of the car|
|Temperature| Average 30 days temperature of the car engine before failure detected|
|RPM   | Average 30 days maximum torque speed at maximum torque before failure detected|
|Factory| Origin of the car manufacturing |
|Usage| usage frequency of the car |
|Fuel consumption| Fuel consumption of the car in Litre per kilometre |
|Membership| Type of membership subscripted by the car owner with the automotive company. “Normal” membership offers subscribers two complimentary car servicin per year. “Premium” membership offers subscribers unlimited car servicing.|
|Failure A-E| A type of car failure: “0” = Corresponding fault not identified, “1” = Corresponding fault identified|

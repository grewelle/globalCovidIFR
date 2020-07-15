# globalCovidIFR
Software to calculate infection fatality rate for COVID-19 using data for cases, tests, and deaths. Associated with Grewelle et al. 2020 "Estimating the Global Infection Fatality Rate of COVID-19".

Primary code use requires a csv file.  Template data showing worldometer.org COVID-19 data at April 21, 2020 allows for simple editing.  4 columns can be edited as follows:
-column 1: country name/code
-column 2: rho (sample prevalence = cases/tests)
-column 3: log(CFR) = natural log of the deaths/cases
-column 4: number of tests performed

Edit primary code to include file path and filename in line 29.  Run python program using chosen python IDE (e.g. pycharm) and necessary dependencies/packages available with Anaconda https://www.anaconda.com/.


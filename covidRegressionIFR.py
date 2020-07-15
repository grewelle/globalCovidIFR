import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns; sns.set(style="white", color_codes=True)
import csv
from scipy.optimize import minimize
from scipy import stats
sns.set(font_scale=3)


def impData(filename): #import csv file of raw data
    abs_file_path = filename
    with open(abs_file_path, newline='') as csvfile:
        totalData = list(csv.reader(csvfile))
    return totalData


def inv_2(x, t, y, w): #weighted regression minimization

    reg = sum(w[i] * (x[0] + x[1]*t[i] - y[i]) ** 2 for i in range(len(t)))

    return reg


def tss(t, y, w): #R-squared calculation
    meanY = sum(w[i] * y[i] for i in range(len(t)))
    return sum(w[i] * (meanY - y[i]) ** 2 for i in range(len(t)))

def main():
    file = "C:/filepath/filename"  # import data file
    data = impData(file)
    country = list(np.array(data)[:, 0]) #record country name
    effort = list(np.array(data)[:, 1]) #record rho
    fatality = list(np.array(data)[:, 2]) #record CFR
    numTests = list(np.array(data)[:, 3]) #record sample size as number of tests
    weightSum = 0
    weight = []

    for i in range(len(country)):  # this loop converts values to floats and calculates weights for each data point
        weight.append(float(numTests[i].replace(',', '')))
        numTests[i] = float(numTests[i].replace(',', ''))
        effort[i] = float(effort[i])
        fatality[i] = float(fatality[i])#linear
        weightSum += weight[i]

    for e in range(len(weight)): #normalize weights
        weight[e] = weight[e]/weightSum

    sizes= np.array(weight)*10000
    x0 =np.array([1,1]) #arbitrary starting values for parameters
    res = minimize(inv_2, x0, args=(effort, fatality, weight), method='BFGS', tol=1e-20,
                   options={'maxiter': 10000}) #weighted regression optimized, see inv_2

    params = res.x #store parameter estimates (slope and intercept)
    rss = inv_2(params, effort, fatality, weight) / (len(effort) - len(params))
    rsquare = 1 - inv_2(params, effort, fatality, weight)/tss(effort, fatality, weight) #R-square calculation, see tss
    print('r-square = ' + str(rsquare))
    print(res.hess_inv)
    stdError = np.sqrt(np.diag(res.hess_inv)) * np.sqrt(rss)
    cov = res.hess_inv[0][1]*rss #variance-covariance matrix
    print(params)
    print(stdError)
    ifr = np.exp(params[0]) #calculate IFR


    """Calculate linear and nonlinear confidence intervals around regression mean"""
    nonlinearFunc = str(ifr) + '*np.exp(' + str(params[1]) + '/(t))'
    linearFunc =  str(params[0]) +'+'+ str(params[1]) + '*(t)'
    linearError = 'np.sqrt(' + str(stdError[0])+ '**2 + (t)**2*' + str(stdError[1]) + '**2 + 2*(t)*' + str(cov) + ')'
    lowerLinear = linearFunc + '-1.96*' + linearError
    upperLinear = linearFunc + '+1.96*' + linearError

    nonlinear95lower_a = np.exp(params[0])-np.exp(params[0]-1.96*stdError[0])
    nonlinear95upper_a = -np.exp(params[0]) + np.exp(params[0] + 1.96 * stdError[0])
    print(nonlinear95lower_a, nonlinear95upper_a)

    nonlinear95lower_all = 'np.sqrt(' + str(nonlinear95lower_a)+ '**2*np.exp(2*' + str(params[1]) + '/t) +(' + str(stdError[1]) + '*1.96)**2*(' + str(ifr) + '/t)**2*np.exp(2*' + str(params[1]) + '/t) + 2*(' + str(ifr) + '/t)*np.exp(2*1.96**2*' + str(params[1]) + '/t)*.0125*' + str(cov) + ')'
    nonlinear95upper_all = 'np.sqrt(' + str(nonlinear95upper_a)+ '**2*np.exp(2*' + str(params[1]) + '/t) +(' + str(stdError[1]) + '*1.96)**2*(' + str(ifr) + '/t)**2*np.exp(2*' + str(params[1]) + '/t) + 2*(' + str(ifr) + '/t)*np.exp(2*1.96**2*' + str(params[1]) + '/t)*.0125*' + str(cov) + ')'

    lowerNonlinear = nonlinearFunc + '-' + nonlinear95lower_all
    upperNonlinear = nonlinearFunc + '+' + nonlinear95upper_all


    curvePrediction = []
    errorPrediction = []
    empiricalError = []
    for h in range(len(country)):
        t = effort[h]
        curvePrediction.append(eval(linearFunc))
        errorPrediction.append(eval(linearError))
        empiricalError.append(np.sqrt(70000/(numTests[h]-1))) #calculate calibrated stdError for each country

    t_stat = []
    df = []
    for i in range(len(country)): #perform Welch's t test for each country

        t_val = (fatality[i] - curvePrediction[i]) / np.sqrt(
            (empiricalError[i]*np.sqrt((numTests[i]-1)/numTests[i])) ** 2  + (errorPrediction[i]*np.sqrt((len(country)-1)/len(country))) ** 2 )
        t_stat.append(t_val)
        v = ((empiricalError[i] * np.sqrt((numTests[i] - 1) / numTests[i])) ** 2 + (
                    errorPrediction[i] * np.sqrt((len(country) - 1) / len(country))) ** 2) ** 2 / (
                        (empiricalError[i] * np.sqrt((numTests[i] - 1) / numTests[i])) ** 4/(numTests[i] - 1) + (
                            errorPrediction[i] * np.sqrt((len(country) - 1) / len(country))) ** 4/(len(country) - 1))
        df.append(v)

    pVals = []
    for d in range(len(df)):
        pVals.append(stats.t.sf(t_stat[d], df[d])) #store p-vals

    count = 0
    for y in range(len(country)):
        print(pVals[y])
        if pVals[y] < 0.975 and pVals[y] > 0.025:
            count += 1
    print(count/len(pVals))


    t = np.linspace(0, 1, 100) # linear
    #t = np.linspace(1, 1500, 15000)  # toggle non-linear axis


    fig, ax = plt.subplots()
    ax.scatter(effort, fatality, s=sizes, alpha=0.5)

    """for i, txt in enumerate(country):
        ax.annotate(txt, (effort[i], fatality[i]))"""


    ax.plot(t, eval(linearFunc)) # use for linear plotting
    ax.plot(t, eval(upperLinear), color='k', linestyle='--')
    ax.plot(t, eval(lowerLinear), color='k', linestyle='--')


    """ax.plot(t, eval(nonlinearFunc)) # use for nonlinear plotting
    ax.plot(t, eval(upperNonlinear), color='k', linestyle='--')
    ax.plot(t, eval(lowerNonlinear), color='k', linestyle='--') # nonlinear"""

    #plt.xscale('log') #toggle log x-axis
    plt.show()


main()
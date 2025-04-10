import pandas
import numpy 
from matplotlib import pyplot as plt
from scipy import stats
from statsmodels.api import OLS
from statsmodels.tsa.stattools import acf
from statsmodels.graphics.gofplots import qqplot
from statsmodels.graphics.tsaplots import plot_acf

# Testing ACF for original and absolute values of residuals
# and the quantile-quantile plot vs normal for residuals
def plots(data, label):
    plot_acf(data, zero = False)
    plt.title(label + '\n ACF for Original Values')
    plt.xlabel('Lags')
    plt.show()
    plot_acf(abs(data), zero = False)
    plt.title(label + '\n ACF for Absolute Values')
    plt.xlabel('Lags')
    plt.show()
    qqplot(data, line = 's')
    plt.title(label + '\n Quantile-Quantile Plot vs Normal')
    plt.show()

# Testing residuals for normality by computation of skewness, kurtosis
# Jarque-Bera and Shapiro-Wilk tests, and summation of ACF values
def analysis(data):
    print('Stdev = ', round(numpy.std(data), 5))
    print('Skewness = ', round(stats.skew(data), 3))
    print('Kurtosis = ', round(stats.kurtosis(data), 3))
    print('ACF L1 original values = ', round(sum(abs(acf(data, nlags = 5)[1:6])), 3))
    print('ACF L1 absolute values = ', round(sum(abs(acf(abs(data), nlags = 5)[1:6])), 3))
    print('Shapiro-Wilk p = ', round(stats.shapiro(data)[1], 5))
    print('Jarque-Bera p = ', round(stats.jarque_bera(data)[1], 5))

# Reading and preprocessing the data
DFrates = pandas.read_excel('annualBofA.xlsx', sheet_name = 'rates')
rates = DFrates.values[:, 1:]/100
DFwealth = pandas.read_excel('annualBofA.xlsx', sheet_name = 'wealth')
wealth = DFwealth.values[:, 1:]
DFcommon = pandas.read_excel('annualBofA.xlsx', sheet_name = 'common')
vix = DFcommon.values[:, 1].astype(float)
trate = DFcommon.values[:, 2].astype(float)
N = len(vix)
allratings = ['Corporate', 'AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC']
returns = numpy.diff(numpy.log(wealth), axis = 0)

for ratings in range(8):
    plt.plot(rates[:, ratings], label = allratings[ratings])
plt.legend()
plt.xlabel('Year')
plt.title('Corporate bond rates')
plt.show()

# Plot of wealth process for seven ratings
for ratings in range(8):
    plt.plot(wealth[:, ratings]/wealth[0, ratings], label = allratings[ratings])
plt.legend()
plt.title('Wealth process for corporate rated bonds, 1996 = 1$')
plt.xlabel('Year')
plt.show()

# Log Heston model: Autoregression of order 1 for annual averaged VIX
lvix = numpy.log(vix)
logHeston = stats.linregress(lvix[:-1], numpy.diff(lvix))
print('Autoregression Log Heston')
print('log(V(t)) = a + b * log(V(t-1)) + Z(t)')
sl = logHeston.slope
it = logHeston.intercept
R2 = logHeston.rvalue**2
print('Slope = ', round(sl, 3), 'Intercept = ', round(it, 3))
print('R^2 = ', round(R2, 3), 'p-value = ', round(logHeston.pvalue, 3))
vixres = numpy.array([lvix[k+1] - lvix[k] * (sl + 1) - it for k in range(N-1)])
analysis(vixres)
plots(vixres, ' AR log VIX')

# Analysis for each ratings
for ratings in range(8):
    print('Ratings ', allratings[ratings])
    
    # Relabeling the data, picking the series for this ratings
    tr = returns[:, ratings]
    rate = rates[:, ratings]
    drate = numpy.diff(rate)
    
    # Simple autoregression for rates
    print('AR for rates')
    print('R(t) - R(t-1) = a + b * R(t-1) + Z(t)')
    Reg = stats.linregress(rate[:-1], drate)
    s = Reg.slope
    i = Reg.intercept
    
    # Analysis of innovations for normality and white noise
    residuals = numpy.array([drate[k] - s * rate[k] - i for k in range(N-1)])
    print('slope = ', s, ' intercept = ', i, ' p = ', Reg.pvalue)
    plots(residuals, 'Bond Rates, Simple Regression for Rating ' + allratings[ratings])
    analysis(residuals)
    
    # Enhanced autoregression for rates with volatility
    print('AR of rates with VIX')
    print('R(t) - R(t-1) = a + b * R(t-1) + c * V(t) + V(t) * Z(t)')
    RegDF = pandas.DataFrame({'const' : 1/vix[1:], 'lag' : rate[:-1]/vix[1:], 'vix' : 1})
    Reg = OLS(drate/vix[1:], RegDF).fit()
    print(Reg.summary())
    print(Reg.params)
    
    # Analysis of innovations for normality and white noise
    plots(Reg.resid, 'Bond Rates, Full Regression for Rating ' + allratings[ratings])
    ratesResid = Reg.resid
    analysis(ratesResid)
    print('Correlation = ', stats.pearsonr(ratesResid, vixres))
    
    # Comparison of this autoregression with VIx with vs without intercept term cV(t)
    print('Cut AR of rates with VIX')
    print('R(t) - R(t-1) = a + b * R(t-1) + V(t) * Z(t)')
    RegDF = pandas.DataFrame({'const' : 1/vix[1:], 'lag' : rate[:-1]/vix[1:]})
    Reg = OLS(drate/vix[1:], RegDF).fit()
    print('R^2 of cut autoregression with VIX = ', Reg.rsquared)
    
    # Simple regression of returns vs rate change: How to find duration
    print('Returns minus rate')
    print('Simple regression vs rate change')
    print('Q(t) - R(t-1) = a + b * (R(t) - R(t-1)) + Z(t)')
    ntr = tr - rate[:-1]
    Reg = stats.linregress(drate, ntr)
    s = Reg.slope
    i = Reg.intercept
    
    # Analysis of residuals for normality and white noise 
    residuals = numpy.array([ntr[k] - s * drate[k] - i for k in range(N-1)])
    print(Reg)
    plots(residuals, 'Bond Returns, Simple Regression for Rating ' + allratings[ratings])
    analysis(residuals)
    
    # Regression of returns minus rate vs rate change with normalization by VIX
    print('Same regression with VIX')
    print('Q(t) - R(t-1) = a + b * (R(t) - R(t-1)) + c * V(t) + Z(t) * V(t)')
    RegCutDF = pandas.DataFrame({'const' : 1/vix[1:], 'duration' : drate/vix[1:], 'vix' : 1})
    Reg = OLS(ntr/vix[1:], RegCutDF).fit()
    print(Reg.summary())
    
    # Analysis of residuals for normality and white noise
    returnsResid = Reg.resid
    plots(returnsResid, 'Bond Returns, Full Regression for Rating ' + allratings[ratings])
    analysis(returnsResid)
    
    # Computation of the covariance matrix of 3 series of residuals
    allResid = numpy.stack([vixres, ratesResid, returnsResid])
    print('Covariance Matrix for Residuals = \n', numpy.cov(allResid))
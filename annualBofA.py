import pandas
import numpy 
from matplotlib import pyplot as plt
from scipy import stats
from statsmodels.api import OLS
from statsmodels.tsa.stattools import acf
from statsmodels.graphics.gofplots import qqplot
from statsmodels.graphics.tsaplots import plot_acf

def plots(data, label):
    plot_acf(data, zero = False)
    plt.title(label + '\n ACF for Original Values')
    plt.show()
    plot_acf(abs(data), zero = False)
    plt.title(label + '\n ACF for Absolute Values')
    plt.show()
    qqplot(data, line = 's')
    plt.title(label + '\n Quantile-Quantile Plot vs Normal')
    plt.show()
    
def analysis(data):
    print('Stdev = ', round(numpy.std(data), 3))
    print('Skewness = ', round(stats.skew(data), 3))
    print('Kurtosis = ', round(stats.kurtosis(data), 3))
    print('ACF L1 original values = ', round(sum(abs(acf(data, nlags = 5)[1:6])), 3))
    print('ACF L1 absolute values = ', round(sum(abs(acf(abs(data), nlags = 5)[1:6])), 3))
    print('Shapiro-Wilk p = ', round(stats.shapiro(data)[1], 3))
    print('Jarque-Bera p = ', round(stats.jarque_bera(data)[1], 3))

DFrates = pandas.read_excel('annualBofA.xlsx', sheet_name = 'rates')
rates = DFrates.values[:, 1:]/100
DFwealth = pandas.read_excel('annualBofA.xlsx', sheet_name = 'wealth')
wealth = DFwealth.values[:, 1:]
DFcommon = pandas.read_excel('annualBofA.xlsx', sheet_name = 'common')
vix = DFcommon.values[:, 1].astype(float)
trate = DFcommon.values[:, 2].astype(float)
N = len(vix)
allratings = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC']
returns = numpy.diff(numpy.log(wealth), axis = 0)

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

for ratings in range(7):
    print('Ratings ', allratings[ratings])
    tr = returns[:, ratings]
    rate = rates[:, ratings]
    drate = numpy.diff(rate)
    print('AR for rates')
    print('R(t) - R(t-1) = a + b * R(t-1) + Z(t)')
    Reg = stats.linregress(rate[:-1], drate)
    s = Reg.slope
    i = Reg.intercept
    residuals = numpy.array([drate[k] - s * rate[k] - i for k in range(N-1)])
    print('slope = ', s, ' intercept = ', i, ' p = ', Reg.pvalue)
    plots(residuals, 'Bond Rates, Simple Regression')
    analysis(residuals)
    print('AR of rates with VIX')
    print('R(t) - R(t-1) = a + b * R(t-1) + c * V(t) + V(t) * Z(t)')
    RegDF = pandas.DataFrame({'const' : 1/vix[1:], 'lag' : rate[:-1]/vix[1:], 'vix' : 1})
    Reg = OLS(drate/vix[1:], RegDF).fit()
    print(Reg.summary())
    print(Reg.params)
    plots(Reg.resid, 'Bond Rates, Full Regression')
    analysis(Reg.resid)
    print('Correlation = ', stats.pearsonr(Reg.resid, vixres))
    print('Returns minus rate')
    print('Simple regression vs rate change')
    print('Q(t) - R(t-1) = a + b * (R(t) - R(t-1)) + Z(t)')
    ntr = tr - rate[:-1]
    Reg = stats.linregress(drate, ntr)
    s = Reg.slope
    i = Reg.intercept
    residuals = numpy.array([ntr[k] - s * drate[k] - i for k in range(N-1)])
    print(Reg)
    plots(residuals, 'Bond Returns, Simple Regression')
    analysis(residuals)
    print('Same regression with VIX')
    print('Q(t) - R(t-1) = a + b * (R(t) - R(t-1)) + c * V(t) + Z(t) * V(t)')
    RegCutDF = pandas.DataFrame({'const' : 1/vix[1:], 'duration' : drate/vix[1:], 'vix' : 1})
    Reg = OLS(ntr/vix[1:], RegCutDF).fit()
    plots(Reg.resid, 'Bond Returns, Full Regression')
    print(Reg.summary())
    analysis(Reg.resid)
    print('Correlation = ', stats.pearsonr(Reg.resid, vixres))
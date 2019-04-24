import cfe
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## Choose dataset ##
USE = "ICRISAT"
#USE = "Tanzania"

# Results can be made persistent by saving them, and then re-loading...
if USE=="ICRISAT":
    DIR = "./IndianICRISAT/"
    r = cfe.from_dataset(DIR+'indian_icrisat.ds')
    USE_GOOD = 'Milk' #'Bengalgram dhal'
elif USE=="Tanzania":
    DIR = "./TanzanianNPS/"
    r = cfe.from_dataset(DIR+'tanzanian_nps.ds')
    USE_GOOD = 'Ripe Bananas'
else:
    raise ValueError("No such value of USE")

fct = pd.read_pickle(DIR+'fct.df')
rda = pd.read_pickle(DIR+"rda.df")

# Use prices, distribution of budgets from first round, first market:
t = r.firstround  # First round
m = r.coords['m'][0] # First market

# Distribution of predicted total expenditures
xhat = r.get_predicted_expenditures().sum('i')
xhat = xhat.where(xhat>0,np.nan)

# Note selection of prices for first period and first market
p = r.prices.sel(t=t,m=m).fillna(1).copy()

def my_prices(p0,p=p,i=USE_GOOD):
    p = p.copy()
    p.loc[i] = p0
    return p

# Now fix up FCT

# Change some labels in fct
d={'protein':'Protein',
   'protein g':'Protein',
   'fat':'Fat',
   'energy_kcal':'Calories',
   'energy kcal':'Calories',
   'calcium':'Calcium',
   'ca mg':'Calcium',
   'betacarotene':'Betacarotene',
   'thiamine':'Thiamine',
   'riboflavin':'Riboflavin',
   'niacin':'Niacin',
   'iron':'Iron',
   'fe mg':'Iron',
   'ascorbic_total':'Ascorbic Acid',
   'vit a ug':'Vitamin A',
   'vit b6 mg':'Vitamin B6',
   'vit b12 ug':'Vitamin B12',
   'vit d ug':'Vitamin D',
   'vit e ug':'Vitamin E',
   'vit c mg':'Vitamin C',
   'mg mg':'Magnesium',
   'zn mg':'Zinc'}

fct = fct.rename(columns=d) #[list(d.values())]

# Fix capitalization (to match food labels)
fct.index = fct.reset_index()['Item name'].str.title()

# Replace missing with zeros
fct = fct.fillna(0)

try:
    fct.index = fct.index.droplevel('unit')
except AttributeError: pass # No units?

def nutrient_demand(x,p,z=None):
    c = r.demands(x,p,z=z)
    fct0,c0 = fct.align(c,axis=0,join='inner')
    N = fct0.T@c0

    return N

# In first round, averaged over households and villages
zbar = r.z.sel(t=r.firstround).mean(['j','m'])[:-1] # Leave out log HSize

# This matrix product gives minimum nutrient requirements for average
# household in first round
hh_rda = rda.replace('',0).T@zbar

def nutrient_adequacy_ratio(x,p):
    return nutrient_demand(x,p)/hh_rda

UseNutrients = ['Protein','Calories','Iron','Calcium']

# A quantile of 0.5 is the median.  Play with quantile value, or just assign.
x0 = xhat.sel(t=t,m=m).quantile(0.01,'j') # Budget (median household)
x0 = 50

X = np.linspace(x0/10,x0*5,50)


# Choose reference (t,m) for reference good
ref_price = r.prices.sel(i=USE_GOOD,t=t,m=m)
P = np.linspace(ref_price/10,ref_price*5,50)

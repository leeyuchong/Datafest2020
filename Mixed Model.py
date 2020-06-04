import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm


data = pd.read_csv('/Users/ani1705/Downloads/Dataframe_OrdinalPlans.csv')

# Cleaning the dataframe
data = data.drop(data.columns[0], axis=1)
data = data[['plan', 'state', 'Application', 'Endowment', 'distcrs', 'alloncam','board',
          'number_locations', 'stufacr', 'county.cases', 'saeq9at_2',
          'sanin04', 'sanin05', 'sanin08', 'Sent_Score']]

# Converting to ordinal
data.plan = data.plan/2
data.plan

# Cleaning the dataframe
data.dropna(inplace=True)
new_index = range(data.shape[0])
data.index = new_index

# First run our linear regression with those predictors
model = LinearRegression(fit_intercept=True)
x = data[['Application', 'Endowment', 'distcrs', 'alloncam','board',
          'number_locations', 'stufacr', 'county.cases', 'saeq9at_2',
          'sanin04', 'sanin05', 'sanin08']]
y = data.plan

x = sm.add_constant(x) ## let's add an intercept (beta_0) to our model

# Fitting the model
model = sm.OLS(y, x).fit()
predictions = model.predict(x)
model.summary()

# Check for multicollinearity
sns1 = sns.pairplot(x)
sns1.savefig("output.png")

# Firstly see if there is a difference
grid = sns.lmplot(x = "distcrs", y = "plan", col = "state", sharex=False,
                  col_wrap = 4, data = data, height=4)

# We do see a difference
data['county'] = 0
data['county'] = data['county.cases']

# construct our model, with our state now shown as a group in a Random
# intercept model
md = smf.mixedlm("plan ~ distcrs + Application + Endowment + distcrs + alloncam + board + number_locations + stufacr + county + saeq9at_2 + sanin04 + sanin05 + sanin08", data, groups=data["state"])
mdf = md.fit(reml=False)
print(mdf.summary())

# Random Intercept + Randome Slope: SentScore
md = smf.mixedlm("plan ~ distcrs+ Application + Endowment + distcrs + alloncam + board + number_locations + stufacr + county + saeq9at_2 + sanin04 + sanin05 + sanin08", data, groups=data["state"], re_formula="~Sent_Score")
mdf = md.fit(reml=False)
print(mdf.summary())

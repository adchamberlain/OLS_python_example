#==============================================================================
# OLS Example in Python
# by Andrew Chamberlain, Ph.D.
# achamberlain.com
#==============================================================================

import pandas as pd
import numpy as np
import statsmodels.api as sm

# Generate some fake data.
np.random.seed(3)

c1 = np.random.normal(0,3,1000)

c2 = np.random.uniform(0,9,1000)

c3 = np.random.uniform(1,1,1000) # Approach one.
c3 = np.ones(1000) # Approach two. 

y = 2*c1 + 4*c2 + 3 + np.random.normal(0,1,1000)

# Stack vectors into a matrix.
x = np.column_stack((c1, c2, c3))

# View top 10 rows of x.
x[:10]

# Create data frame from x matrix. 
df = pd.DataFrame(data = x)

# Create column names for data frame.
df.columns = ['a','b','c']

# View top 10 rows of data frame.
df.head(10)

# View first 10 rows of data frame
df[:10]

# View entry i,j in data frame (0 is first row, column).
df.iloc[0,1] # row 1, column 2.
df.iloc[0:3, 0:2] # first 3 rows (0 1 2 ), first 2 columns (0 1).
df.iloc[0:1,0:1] # first row, first column
df.loc[0, 'a'] # First element in column a.
df.ix[0] # Get first row of data. 


# Get dimensions of data frame.
df.shape
df.shape[0] # Number of rows of df.


# Manually calculate OLS coefficients.
b_ols = np.dot( np.linalg.inv( np.dot(x.transpose(),x)), np.dot( x.transpose(), y) )

# Generate fitted values manually. 
y_hat = np.dot( x, b_ols)

# Manually calculate homoskedastic SEs.
epsilon = y - y_hat
sigma_squared = np.var(epsilon)
sigma = np.sqrt(sigma_squared)
sand = np.linalg.inv( np.dot(x.transpose(), x) ) # 3x3 var-cov martrix.
se_homo = sigma * np.sqrt( np.diag(sand) )

print(b_ols)
print(se_homo)


#Run OLS and print regression results via statsmodels package (robust standard errors).
model = sm.OLS(y, x)
results = model.fit(cov_type='HC1')
print(results.summary())

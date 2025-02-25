#!/usr/bin/env python
# coding: utf-8

# # Activity: Perform multiple linear regression
# 

# ## Introduction

# As you have learned, multiple linear regression helps you estimate the linear relationship between one continuous dependent variable and two or more independent variables. For data science professionals, this is a useful skill because it allows you to compare more than one variable to the variable you're measuring against. This provides the opportunity for much more thorough and flexible analysis. 
# 
# For this activity, you will be analyzing a small business' historical marketing promotion data. Each row corresponds to an independent marketing promotion where their business uses TV, social media, radio, and influencer promotions to increase sales. They previously had you work on finding a single variable that predicts sales, and now they are hoping to expand this analysis to include other variables that can help them target their marketing efforts.
# 
# To address the business' request, you will conduct a multiple linear regression analysis to estimate sales from a combination of independent variables. This will include:
# 
# * Exploring and cleaning data
# * Using plots and descriptive statistics to select the independent variables
# * Creating a fitting multiple linear regression model
# * Checking model assumptions
# * Interpreting model outputs and communicating the results to non-technical stakeholders

# ## Step 1: Imports

# ### Import packages

# Import relevant Python libraries and modules.

# In[1]:


# Import libraries and modules.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import ols
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats


# ### Load dataset

# `Pandas` was used to load the dataset `marketing_sales_data.csv` as `data`, now display the first five rows. The variables in the dataset have been adjusted to suit the objectives of this lab. As shown in this cell, the dataset has been automatically loaded in for you. You do not need to download the .csv file, or provide more code, in order to access the dataset and proceed with this lab. Please continue with this activity by completing the following instructions.

# In[2]:


# RUN THIS CELL TO IMPORT YOUR DATA.

### YOUR CODE HERE ### 
data = pd.read_csv('marketing_sales_data.csv')

# Display the first five rows.

data.head()


# ## Step 2: Data exploration

# ### Familiarize yourself with the data's features
# 
# Start with an exploratory data analysis to familiarize yourself with the data and prepare it for modeling.
# 
# The features in the data are:
# 
# * TV promotional budget (in "Low," "Medium," and "High" categories)
# * Social media promotional budget (in millions of dollars)
# * Radio promotional budget (in millions of dollars)
# * Sales (in millions of dollars)
# * Influencer size (in "Mega," "Macro," "Micro," and "Nano" categories)
# 

# **Question:** What are some purposes of EDA before constructing a multiple linear regression model?

# EDA helps identify relationships between variables, detect outliers, assess variable distributions, determine which independent variables show linear relationships with the dependent variable, and identify potential multicollinearity issues before modeling.

# ### Create a pairplot of the data
# 
# Create a pairplot to visualize the relationship between the continous variables in `data`.

# In[3]:


# Create a pairplot of the data.

sns.pairplot(data.select_dtypes(include=[np.number]))
plt.show()


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Refer to [the content where creating a pairplot is demonstrated](https://www.coursera.org/learn/regression-analysis-simplify-complex-data-relationships/item/dnjWm).
# 
# </details>
# 

# <details>
# <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# Use the function in the `seaborn` library that allows you to create a pairplot showing the relationships between variables in the data.
# 
# </details>
# 

# <details>
# <summary><h4><strong>Hint 3</strong></h4></summary>
# 
# Use the `pairplot()` function from the `seaborn` library and pass in the entire DataFrame.
# 
# </details>
# 

# **Question:** Which variables have a linear relationship with `Sales`? Why are some variables in the data excluded from the preceding plot?
# 
# 

# Both "Radio" and "Social Media" show linear relationships with "Sales".
# 
# Some variables from the dataset are excluded from this pairplot because it only shows continuous numerical variables (selected using data.select_dtypes(include=[np.number])). Categorical variables like "TV" (with Low, Medium, High categories) and "Influencer" (with Mega, Macro, Micro, Nano categories) are excluded since they cannot be meaningfully represented in a scatterplot or histogram. The pairplot specifically visualizes relationships between continuous variables to check for linearity and correlation patterns.

# ### Calculate the mean sales for each categorical variable

# There are two categorical variables: `TV` and `Influencer`. To characterize the relationship between the categorical variables and `Sales`, find the mean `Sales` for each category in `TV` and the mean `Sales` for each category in `Influencer`. 

# In[4]:


# Calculate the mean sales for each TV category. 
tv_mean_sales = data.groupby('TV')['Sales'].mean()
print("Mean Sales by TV Category:")
print(tv_mean_sales)



# Calculate the mean sales for each Influencer category. 

influencer_mean_sales = data.groupby('Influencer')['Sales'].mean()
print("\nMean Sales by Influencer Category:")
print(influencer_mean_sales)



# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Find the mean `Sales` when the `TV` promotion is `High`, `Medium`, or `Low`.
#     
# Find the mean `Sales` when the `Influencer` promotion is `Macro`, `Mega`, `Micro`, or `Nano`.  
# 
# </details>

# <details>
# <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# Use the `groupby` operation in `pandas` to split an object (e.g., data) into groups and apply a calculation to each group.
# 
# </details>

# <details>
# <summary><h4><strong>Hint 3</strong></h4></summary>
# 
# To calculate the mean `Sales` for each `TV` category, group by `TV`, select the `Sales` column, and then calculate the mean. 
#     
# Apply the same process to calculate the mean `Sales` for each `Influencer` category.
# 
# </details>

# **Question:** What do you notice about the categorical variables? Could they be useful predictors of `Sales`?
# 
# 

# Looking at the categorical variables:
# 
# For TV categories:
# 
# • There's a very clear, strong pattern where sales increase substantially across the Low → Medium → High categories
# • The differences are large: High TV spending shows over 3 times the sales of Low TV spending
# • The progression appears ordinal and consistent
# 
# For Influencer categories:
# 
# • The differences between categories are much smaller (only about a 7% difference between highest and lowest)
# • The pattern doesn't follow a clear ordinal relationship - Nano influencers outperform Micro influencers despite presumably being smaller
# • Mega influencers do perform best, but not by a large margin
# 
# As predictors of Sales:
# 
# • TV categories would likely be very useful predictors given the strong, consistent relationship with sales
# • Influencer categories might add some predictive value, but the relationship is weaker and less consistent
# • Both would be worth including in the model, but I would expect TV categories to show stronger statistical significance

# ### Remove missing data
# 
# This dataset contains rows with missing values. To correct this, drop all rows that contain missing data.

# In[5]:


# Drop rows that contain missing data and update the DataFrame.

data = data.dropna()
print(f"\nShape after dropping missing values: {data.shape}")


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Use the `pandas` function that removes missing values.
# 
# </details>

# <details>
# <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# The `dropna()` function removes missing values from an object (e.g., DataFrame).
# 
# </details>

# <details>
# <summary><h4><strong>Hint 3</strong></h4></summary>
# 
# Use `data.dropna(axis=0)` to drop all rows with missing values in `data`. Be sure to properly update the DataFrame.
# 
# </details>

# ### Clean column names

# The `ols()` function doesn't run when variable names contain a space. Check that the column names in `data` do not contain spaces and fix them, if needed.

# In[6]:


# Rename all columns in data that contain a space. 

data.columns = data.columns.str.replace(' ', '_')
print("\nUpdated column names:")
print(data.columns)



# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# There is one column name that contains a space. Search for it in `data`.
# 
# </details>

# <details>
# <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# The `Social Media` column name in `data` contains a space. This is not allowed in the `ols()` function.
# 
# </details>

# <details>
# <summary><h4><strong>Hint 3</strong></h4></summary>
# 
# Use the `rename()` function in `pandas` and use the `columns` argument to provide a new name for `Social Media`.
# 
# </details>

# ## Step 3: Model building

# ### Fit a multiple linear regression model that predicts sales
# 
# Using the independent variables of your choice, fit a multiple linear regression model that predicts `Sales` using two or more independent variables from `data`.

# In[9]:


# Define the OLS formula with the correct column name
formula = "Sales ~ Social_Media + Radio + C(TV) + C(Influencer)"

# Create an OLS model
model = ols(formula, data=data)

# Fit the model
results = model.fit()

# Save the results summary
summary = results.summary()

# Display the model results
print(summary)


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Refer to the content that discusses [model building](https://www.coursera.org/learn/regression-analysis-simplify-complex-data-relationships/lecture/zd74V/interpret-multiple-regression-coefficients) for linear regression.
# 
# </details>

# <details>
# <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# Use the `ols()` function imported earlier—which creates a model from a formula and DataFrame—to create an OLS model.
# 
# </details>

# <details>
# <summary><h4><strong>Hint 3</strong></h4></summary>
# 
# You previously learned how to specify in `ols()` that a feature is categorical. 
#     
# Be sure the string names for the independent variables match the column names in `data` exactly.
# 
# </details>

# **Question:** Which independent variables did you choose for the model, and why?
# 
# 

# I chose Social_media, Radio, TV, and Influencer because: 
# 1) Social_media and Radio show linear relationships with Sales in the scatterplots, and 
# 2) The categorical variables TV and Influencer show different mean sales values across categories, suggesting predictive value.

# ### Check model assumptions

# For multiple linear regression, there is an additional assumption added to the four simple linear regression assumptions: **multicollinearity**. 
# 
# Check that all five multiple linear regression assumptions are upheld for your model.

# ### Model assumption: Linearity

# Create scatterplots comparing the continuous independent variable(s) you selected previously with `Sales` to check the linearity assumption. Use the pairplot you created earlier to verify the linearity assumption or create new scatterplots comparing the variables of interest.

# In[11]:


# Create a scatterplot for each independent variable and the dependent variable.

# Check model assumptions
# Linearity assumption
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(data['Social_Media'], data['Sales'])
axes[0].set_xlabel('Social Media Budget')
axes[0].set_ylabel('Sales')
axes[0].set_title('Sales vs Social Media')

axes[1].scatter(data['Radio'], data['Sales'])
axes[1].set_xlabel('Radio Budget')
axes[1].set_ylabel('Sales')
axes[1].set_title('Sales vs Radio')
plt.tight_layout()
plt.show()
 


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Use the function in the `seaborn` library that allows you to create a scatterplot to display the values for two variables.
# 
# </details>

# <details>
# <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# Use the `scatterplot()` function in `seaborn`.
# 
# </details>

# <details>
# <summary><h4><strong>Hint 3</strong></h4></summary>
#     
# Pass the independent and dependent variables in your model as the arguments for `x` and `y`, respectively, in the `scatterplot()` function. Do this for each continous independent variable in your model.
# 
# </details>

# **Question:** Is the linearity assumption met?
# 

# Based on the scatterplots, I'd say the linearity assumption is reasonably met, but with some nuances:
# 
# For Radio vs Sales (right plot):
# 
# -There's a clear positive linear relationship
# -The points follow a fairly consistent upward trend
# -The relationship appears strong and linear across the entire range
# 
# For Social Media vs Sales (left plot):
# 
# -There is a general positive relationship
# -The scatter is more pronounced than with Radio
# -There's some clustering pattern visible, possibly influenced by the TV categories
# 
# Overall, the linearity assumption is better satisfied for Radio than for Social Media, but both show sufficient linear patterns to proceed with the linear regression model. The wider spread in the Social Media plot suggests there may be other factors (like TV category) influencing the relationship, which is why including categorical variables in your multiple regression model is appropriate.

# ### Model assumption: Independence

# The **independent observation assumption** states that each observation in the dataset is independent. As each marketing promotion (i.e., row) is independent from one another, the independence assumption is not violated.

# ### Model assumption: Normality

# Create the following plots to check the **normality assumption**:
# 
# * **Plot 1**: Histogram of the residuals
# * **Plot 2**: Q-Q plot of the residuals

# In[12]:


# Calculate the residuals.

residuals = results.resid


# Create a histogram with the residuals. 

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(residuals, bins=20)
plt.title('Histogram of Residuals')


# Create a Q-Q plot of the residuals.

plt.subplot(1, 2, 2)
stats.probplot(residuals, plot=plt)
plt.title('Q-Q Plot of Residuals')
plt.tight_layout()
plt.show()



# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Access the residuals from the fit model object.
# 
# </details>

# <details>
# <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# Use `model.resid` to get the residuals from a fit model called `model`.
# 
# </details>

# <details>
# <summary><h4><strong>Hint 3</strong></h4></summary>
# 
# For the histogram, pass the residuals as the first argument in the `seaborn` `histplot()` function.
#     
# For the Q-Q plot, pass the residuals as the first argument in the `statsmodels` `qqplot()` function.
# 
# </details>

# **Question:** Is the normality assumption met?
# 
# 

# Based on these plots, the normality assumption is reasonably met, though not perfectly:
# Looking at the histogram of residuals:
# 
# -The distribution appears roughly bell-shaped
# -It's approximately symmetric around zero
# -There might be a slight rightward shift/skew, but it's not dramatic
# 
# Examining the Q-Q plot:
# 
# -Most points follow the red reference line fairly well, especially in the middle range
# -There are some minor deviations at both extremes (particularly at the very low and high ends)
# -The tails show slight departures from normality, with the lower tail being a bit heavier than expected
# 
# While the normality isn't perfect, the deviations aren't severe enough to invalidate your model. In multiple regression with a reasonably large sample size, slight departures from normality are generally tolerable due to the Central Limit Theorem. The model should still provide reliable inferences about the relationships between your variables.

# ### Model assumption: Constant variance

# Check that the **constant variance assumption** is not violated by creating a scatterplot with the fitted values and residuals. Add a line at $y = 0$ to visualize the variance of residuals above and below $y = 0$.

# In[13]:


# Create a scatterplot with the fitted values from the model and the residuals.

plt.figure(figsize=(8, 6))
plt.scatter(results.fittedvalues, residuals)
plt.axhline(y=0, color='r', linestyle='-')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted Values')
plt.show() 


# Add a line at y = 0 to visualize the variance of residuals above and below 0.

### YOUR CODE HERE ### 


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Access the fitted values from the model object fit earlier.
# 
# </details>

# <details>
# <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# Use `model.fittedvalues` to get the fitted values from a fit model called `model`.
# 
# </details>

# <details>
# <summary><h4><strong>Hint 3</strong></h4></summary>
# 
# 
# Call the `scatterplot()` function from the `seaborn` library and pass in the fitted values and residuals.
#     
# Add a line to a figure using the `axline()` function.
# 
# </details>

# **Question:** Is the constant variance assumption met?
# 
# 
# 

# The constant variance assumption (homoscedasticity) is not well met:
# 
# The plot shows a distinct pattern where the residuals appear to be clustered into three vertical bands corresponding to different fitted value ranges (around 75-100, 175-225, and 275-325). Within each band:
# 
# 1.The spread of residuals is roughly similar in magnitude (similar vertical dispersion)
# 2.The residuals are distributed both above and below zero
# 
# This clustering pattern likely corresponds to the three TV categories (Low, Medium, High), which have distinctly different sales levels. While the variance within each group seems relatively constant, the overall pattern does not show the ideal random scatter we'd want for perfect homoscedasticity.
# 
# This suggests that while your model captures the main effects of the variables, there might be some group-specific variance or interaction effects not fully accounted for. The model is still useful, but you should be somewhat cautious when interpreting prediction intervals.

# ### Model assumption: No multicollinearity

# The **no multicollinearity assumption** states that no two independent variables ($X_i$ and $X_j$) can be highly correlated with each other. 
# 
# Two common ways to check for multicollinearity are to:
# 
# * Create scatterplots to show the relationship between pairs of independent variables
# * Use the variance inflation factor to detect multicollinearity
# 
# Use one of these two methods to check your model's no multicollinearity assumption.

# In[15]:


# Create a pairplot of the data.

sns.pairplot(data[['Social_Media', 'Radio']])
plt.show()


# In[16]:


# Calculate the variance inflation factor (optional).

X = data[['Social_Media', 'Radio']]
X = sm.add_constant(X)  # Add constant term
vif_data = pd.DataFrame()
vif_data["Variable"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print("\nVariance Inflation Factors:")
print(vif_data)


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Confirm that you previously created plots that could check the no multicollinearity assumption.
# 
# </details>

# <details>
# <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# The `pairplot()` function applied earlier to `data` plots the relationship between all continous variables  (e.g., between `Radio` and `Social Media`).
# 
# </details>

# <details>
# <summary><h4><strong>Hint 3</strong></h4></summary>
# 
# The `statsmodels` library has a function to calculate the variance inflation factor called `variance_inflation_factor()`. 
#     
# When using this function, subset the data to only include the continous independent variables (e.g., `Radio` and `Social Media`). Refer to external tutorials on how to apply the variance inflation factor function mentioned previously.
#  
# 
# </details>

# **Question 8:** Is the no multicollinearity assumption met?
# 
# Looking at the pairplot and VIF values, the no multicollinearity assumption is met:
# 
# From the scatterplot (bottom left), Social_Media and Radio show only a moderate positive correlation. There isn't a tight linear pattern that would indicate high collinearity.
# The VIF (Variance Inflation Factor) values confirm this:
# 
# Both Social_Media and Radio have VIF values of 1.66
# VIF values below 5 are generally considered acceptable
# Values below 2 indicate very low multicollinearity
# 
# 
# 
# A VIF of 1.66 suggests that the variance of the coefficient estimate is inflated by only 66% due to correlation with other predictors, which is well within acceptable limits.
# The constant term has a higher VIF (4.71), but this is normal and not concerning as it doesn't affect the interpretation of your predictor variables.
# In conclusion, multicollinearity is not a significant issue in your model, and this assumption is satisfied.
# 
# 

# ## Step 4: Results and evaluation

# ### Display the OLS regression results
# 
# If the model assumptions are met, you can interpret the model results accurately.
# 
# First, display the OLS regression results.

# In[17]:


# Display the model results summary.

print(summary)


# **Question:** What is your interpretation of the model's R-squared?
# 

# The R-squared value of 0.904 indicates that 90.4% of the variance in Sales is explained by your model. This is an excellent result that demonstrates strong predictive power. The adjusted R-squared of 0.903 is nearly identical to the R-squared, which confirms that the model isn't over-fitted with unnecessary predictors.
# 
# With such a high R-squared value, your model captures the vast majority of what influences sales variation in this dataset. This is further supported by the extremely low p-value for the F-statistic (1.82e-282), which indicates that the model as a whole is highly statistically significant.
# 
# For a marketing effectiveness analysis, an R-squared above 0.9 is exceptional and suggests that the variables you've included (TV category, Radio spending, Social Media spending, and Influencer category) collectively do an excellent job of explaining what drives sales performance.
# 
# This strong explanatory power gives the business confidence that focusing on the significant factors identified in your model will lead to reliable sales predictions and effective marketing budget allocation.

# ### Interpret model coefficients

# With the model fit evaluated, you can look at the coefficient estimates and the uncertainty of these estimates.
# 
# Again, display the OLS regression results.

# In[18]:


# Display the model results summary.

print(summary)


# **Question:** What are the model coefficients?
# 
# 

# Intercept: 217.48, TV[Low]: -154.57, TV[Medium]: -75.59, Influencer[Mega]: 2.49, Influencer[Micro]: 2.94, Influencer[Nano]: 0.80, Social_Media: -0.14, Radio: 2.97

# **Question:** How would you write the relationship between `Sales` and the independent variables as a linear equation?
# 
# 

# Sales = 217.48 - 154.57×TV[Low] - 75.59×TV[Medium] + 2.49×Influencer[Mega] + 2.94×Influencer[Micro] + 0.80×Influencer[Nano] - 0.14×Social_Media + 2.97×Radio

# **Question:** What is your intepretation of the coefficient estimates? Are the coefficients statistically significant?
# 
# 

# TV categories are highly significant (p<0.001) with substantial negative coefficients for Low and Medium compared to High (baseline). Radio is significant (p<0.001) with a positive coefficient (2.97). Social_Media coefficient is negative but not significant (p=0.837). Influencer categories show minimal impact and aren't statistically significant.
# 

# **Question:** Why is it important to interpret the beta coefficients?
# 
# 

# Beta coefficients reveal which marketing channels drive sales most effectively, allowing for optimal budget allocation. They quantify expected sales impact per unit spending, enabling ROI comparison across channels and informing strategic decisions about where to invest marketing dollars.

# **Question:** What are you interested in exploring based on your model?
# 
# 

# I'd explore potential interaction effects between TV and Radio, investigate why Social_Media shows no significant impact despite correlation in scatterplots, and analyze diminishing returns by testing non-linear relationships with higher-order terms.

# **Question:** Do you think your model could be improved? Why or why not? How?

# Yes. Improvements could include: adding interaction terms between TV and other variables; transforming Social_Media to better capture its non-linear relationship with Sales; addressing the heteroscedasticity visible in residual plots; and incorporating time-based variables to account for seasonal effects.

# ## Conclusion

# **What are the key takeaways from this lab?**
# 
# TV category is the strongest predictor of sales, with High TV outperforming Medium by $75.6M and Low by $154.6M. Radio advertising shows significant positive impact ($2.97M per unit). Social Media surprisingly shows no significant effect. The model explains 90.4% of sales variance, providing a reliable framework for marketing decisions.
# 
# **What results can be presented from this lab?**
# 
# A data-driven marketing allocation model that explains 90.4% of sales variation; quantified impact of each marketing channel; evidence that TV category and Radio spending are the most effective sales drivers; categorical ranking of marketing channels by ROI; and residual analysis validating model reliability.
# 
# **How would you frame your findings to external stakeholders?**
# 
# "Our analysis shows that TV category is the strongest sales driver, with High TV campaigns generating approximately $75.6M more than Medium and $154.6M more than Low campaigns. Every additional $1M spent on Radio increases sales by nearly $3M. We recommend prioritizing High TV campaigns and Radio advertising while carefully evaluating Social Media strategy, as it currently shows no significant sales impact. This model explains over 90% of sales variation, providing confidence in these recommendations."
# 

# #### **References**
# 
# Saragih, H.S. (2020). [*Dummy Marketing and Sales Data*](https://www.kaggle.com/datasets/harrimansaragih/dummy-advertising-and-sales-data).

# **Congratulations!** You've completed this lab. However, you may not notice a green check mark next to this item on Coursera's platform. Please continue your progress regardless of the check mark. Just click on the "save" icon at the top of this notebook to ensure your work has been logged.

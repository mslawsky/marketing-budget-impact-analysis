#!/usr/bin/env python
# coding: utf-8

# # Activity: Hypothesis testing with Python

# ## **Introduction**
# 

# As you've been learning, analysis of variance (commonly called ANOVA) is a group of statistical techniques that test the difference of means among three or more groups. It's a powerful tool for determining whether population means are different across groups and for answering a wide range of business questions.
# 
# In this activity, you are a data professional working with historical marketing promotion data. You will use the data to run a one-way ANOVA and a post hoc ANOVA test. Then, you will communicate your results to stakeholders. These experiences will help you make more confident recommendations in a professional setting. 
# 
# In your dataset, each row corresponds to an independent marketing promotion, where your business uses TV, social media, radio, and influencer promotions to increase sales. You have previously provided insights about how different promotion types affect sales; now stakeholders want to know if sales are significantly different among various TV and influencer promotion types.
# 
# To address this request, a one-way ANOVA test will enable you to determine if there is a statistically significant difference in sales among groups. This includes:
# * Using plots and descriptive statistics to select a categorical independent variable
# * Creating and fitting a linear regression model with the selected categorical independent variable
# * Checking model assumptions
# * Performing and interpreting a one-way ANOVA test
# * Comparing pairs of groups using an ANOVA post hoc test
# * Interpreting model outputs and communicating the results to nontechnical stakeholders

# ## **Step 1: Imports** 
# 

# Import pandas, pyplot from matplotlib, seaborn, api from statsmodels, ols from statsmodels.formula.api, and pairwise_tukeyhsd from statsmodels.stats.multicomp.

# In[1]:


# Import libraries and packages.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import numpy as np


# `Pandas` was used to load the dataset `marketing_sales_data.csv` as `data`, now display the first five rows. The variables in the dataset have been adjusted to suit the objectives of this lab. As shown in this cell, the dataset has been automatically loaded in for you. You do not need to download the .csv file, or provide more code, in order to access the dataset and proceed with this lab. Please continue with this activity by completing the following instructions.

# In[2]:


# RUN THIS CELL TO IMPORT YOUR DATA.

### YOUR CODE HERE ### 
data = pd.read_csv('marketing_sales_data.csv')

# Display the first five rows.

print(data.head())


# The features in the data are:
# * TV promotion budget (in Low, Medium, and High categories)
# * Social media promotion budget (in millions of dollars)
# * Radio promotion budget (in millions of dollars)
# * Sales (in millions of dollars)
# * Influencer size (in Mega, Macro, Nano, and Micro categories)

# **Question:** Why is it useful to perform exploratory data analysis before constructing a linear regression model?

# Exploratory data analysis helps understand the data structure, identify patterns, detect outliers, and assess relationships between variables. By visualizing the data, we can determine which variables might be appropriate for our model, check for potential violations of assumptions, and make informed decisions about model specification. This helps build more accurate and meaningful models.

# ## **Step 2: Data exploration** 
# 

# First, use a boxplot to determine how `Sales` vary based on the `TV` promotion budget category.

# In[3]:


# Create a boxplot with TV and Sales.

plt.figure(figsize=(10, 6))
sns.boxplot(x='TV', y='Sales', data=data)
plt.title('Sales by TV Promotion Budget')
plt.xlabel('TV Promotion Budget')
plt.ylabel('Sales (in millions)')
plt.show()


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# There is a function in the `seaborn` library that creates a boxplot showing the distribution of a variable across multiple groups.
# 
# </details>

# <details>
# <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# Use the `boxplot()` function from `seaborn`.
# 
# </details>

# <details>
# <summary><h4><strong>Hint 3</strong></h4></summary>
# 
# Use `TV` as the `x` argument, `Sales` as the `y` argument, and `data` as the `data` argument.
# 
# </details>

# **Question:** Is there variation in `Sales` based off the `TV` promotion budget?

# Based on the boxplot, there appears to be substantial variation in Sales across different TV promotion budget categories. The High TV budget category shows higher median sales compared to Medium and Low categories, suggesting that TV promotion budget may have a significant impact on sales performance.

# Now, use a boxplot to determine how `Sales` vary based on the `Influencer` size category.

# In[4]:


# Create a boxplot with Influencer and Sales.

plt.figure(figsize=(10, 6))
sns.boxplot(x='Influencer', y='Sales', data=data)
plt.title('Sales by Influencer Size')
plt.xlabel('Influencer Size')
plt.ylabel('Sales (in millions)')
plt.show()


# **Question:** Is there variation in `Sales` based off the `Influencer` size?

# The boxplot shows variation in Sales across different Influencer size categories, although this variation might be less pronounced than with TV budget. Different influencer categories show different median sales values, with potentially higher sales for certain influencer sizes.

# ### Remove missing data
# 
# You may recall from prior labs that this dataset contains rows with missing values. To correct this, drop these rows. Then, confirm the data contains no missing values.

# In[5]:


# Drop rows that contain missing data and update the DataFrame.

data = data.dropna()
print("Any missing values?", data.isnull().any().any())





# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# There is a `pandas` function that removes missing values.
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
# Verify the data is updated properly after the rows containing missing data are dropped.
# 
# </details>

# ## **Step 3: Model building** 
# 

# Fit a linear regression model that predicts `Sales` using one of the independent categorical variables in `data`. Refer to your previous code for defining and fitting a linear regression model.

# In[6]:


# Define the OLS formula.

formula = 'Sales ~ C(TV)'

# Create an OLS model.

model = ols(formula, data=data)


# Fit the model.

results = model.fit()

# Save the results summary.

summary = results.summary()


# Display the model results.

print(summary)


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Refer to code you've written to fit linear regression models.
# 
# </details>

# <details>
# <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# Use the `ols()` function from `statsmodels.formula.api`, which creates a model from a formula and DataFrame, to create an OLS model.
# 
# </details>
# 

# <details>
# <summary><h4><strong>Hint 3</strong></h4></summary>
# 
# Use `C()` around the variable name in the ols formula to indicate a variable is categorical.
#     
# Be sure the variable string names exactly match the column names in `data`.
# 
# </details>

# **Question:** Which categorical variable did you choose for the model? Why?

# I chose TV as the categorical variable for the model because it shows clearer separation between groups in the exploratory boxplots. The variation in Sales across TV budget categories is more distinct, suggesting that TV budget may have a stronger relationship with Sales, making it a better predictor for our analysis.

# ### Check model assumptions

# Now, check the four linear regression assumptions are upheld for your model.

# **Question:** Is the linearity assumption met?

# For ANOVA with categorical predictors, the linearity assumption is automatically satisfied as we're not modeling a continuous relationship. The model simply estimates different means for each category.

# The independent observation assumption states that each observation in the dataset is independent. As each marketing promotion (row) is independent from one another, the independence assumption is not violated.

# Next, verify that the normality assumption is upheld for the model.

# In[9]:


# Calculate the residuals.

residuals = results.resid

# Create a histogram of the residuals
plt.figure(figsize=(10, 6))
plt.hist(residuals, bins=20)
plt.title('Histogram of Residuals')
plt.xlabel('Residual Value')
plt.ylabel('Frequency')
plt.show()

# Create a QQ plot of the residuals
plt.figure(figsize=(10, 6))
sm.qqplot(residuals, line='45')
plt.title('QQ Plot of Residuals')
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
# For the QQ-plot, pass the residuals as the first argument in the `statsmodels` `qqplot()` function.
# 
# </details>

# **Question:** Is the normality assumption met?

# Looking at both the histogram of residuals and the QQ plot, there are clear signs that the residuals don't follow a normal distribution:
# 
# The histogram shows a somewhat multi-modal distribution rather than the expected bell shape of a normal distribution. It appears to have multiple peaks and an irregular pattern.
# The QQ plot provides even stronger evidence against normality. In a QQ plot, points following the red 45-degree line would indicate normally distributed residuals. Instead, we see the blue points forming an almost vertical line that significantly deviates from the reference line. This indicates that the empirical distribution of the residuals is very different from a theoretical normal distribution.
# 
# This violation of the normality assumption may affect the validity of statistical tests in your ANOVA analysis, particularly for smaller sample sizes. You might consider:
# 
# Transforming your dependent variable (Sales)
# Using non-parametric alternatives to ANOVA
# Proceeding with caution, knowing that ANOVA can be somewhat robust to violations of normality with larger sample sizes

# Now, verify the constant variance (homoscedasticity) assumption is met for this model.

# In[10]:


# Create a scatter plot with the fitted values from the model and the residuals.

plt.figure(figsize=(10, 6))
plt.scatter(results.fittedvalues, residuals)
plt.axhline(y=0, color='r', linestyle='-')
plt.title('Residuals vs Fitted Values')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.show()


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Access the fitted values from the model object fit earlier.
# 
# </details>

# <details>
# <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# Use `model.fittedvalues` to get the fitted values from the fit model called `model`.
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

# **Question:** Is the constant variance (homoscedasticity) assumption met?

# Based on this residuals vs. fitted values plot, the constant variance (homoscedasticity) assumption is NOT met.
# The plot shows three distinct vertical columns of residuals, which is typical when modeling with categorical predictors (like your TV promotion budget with Low, Medium, and High categories). Each column represents the residuals for a specific category.
# There are several issues with homoscedasticity visible in this plot:
# 
# The spread of residuals appears to increase slightly as the fitted values increase (from left to right), suggesting potential heteroscedasticity.
# The range of residuals is quite wide (approximately -60 to +60) for each category, indicating substantial variability.
# The residuals don't form a random, even band around the zero line - they form distinct vertical patterns at each fitted value.
# 
# This violation of the homoscedasticity assumption may affect the reliability of standard errors and, consequently, the validity of confidence intervals and p-values in your ANOVA results. When this assumption is violated, you might consider:
# 
# Using robust standard errors
# Applying a variance-stabilizing transformation to your dependent variable
# Using weighted least squares regression
# Considering alternative modeling approaches
# 
# Despite this violation, ANOVA is somewhat robust to moderate violations of homoscedasticity, especially with balanced designs, so you might still proceed with caution while acknowledging this limitation.

# ## **Step 4: Results and evaluation** 

# First, display the OLS regression results.

# In[12]:


# Display the model results summary.

print(summary)


# **Question:** What is your interpretation of the model's R-squared?

# The R-squared value of 0.874 indicates that approximately 87.4% of the variation in Sales is explained by the TV promotion budget categories. This is a very high R-squared value, suggesting that TV promotion budget is strongly associated with Sales outcomes. With both R-squared and adjusted R-squared at 0.874, the model appears to be explaining a substantial portion of the variance in the dependent variable without overfitting.
# 

# **Question:** What is your intepretation of the coefficient estimates? Are the coefficients statistically significant?

# The coefficient estimates show:
# 
# Intercept (300.5296): This represents the estimated mean Sales for the reference category, which is likely "High" TV budget (since Low and Medium are shown as contrasts).
# C(TV)[T.Low] (-208.8133): This indicates that promotions with Low TV budgets have, on average, about $208.8 million less in sales compared to High TV budgets.
# C(TV)[T.Medium] (-101.5061): This indicates that promotions with Medium TV budgets have, on average, about $101.5 million less in sales compared to High TV budgets.
# 
# All coefficients are highly statistically significant with p-values effectively zero (P>|t| = 0.000), and very large t-statistics (124.360, -62.720, and -30.526). The 95% confidence intervals for each coefficient are narrow and do not include zero, further confirming their statistical significance. The results strongly suggest that increasing TV promotion budgets from Low to Medium to High has a substantial positive impact on Sales.

# **Question:** Do you think your model could be improved? Why or why not? How?

# While the model has a high R-squared value, it could potentially be improved in several ways:
# 
# 1.Include additional variables: The model only considers TV promotion budget, but the dataset contains other potential predictors like Social media budget, Radio budget, and Influencer size. A multiple regression model incorporating these variables might capture more complex relationships.
# 2.Address assumption violations: As we observed earlier, the model violates both normality and homoscedasticity assumptions. Transforming the dependent variable (Sales) or using robust regression methods might help address these issues.
# 3.Consider interactions: The effect of TV budget on Sales might vary depending on other factors like Influencer size. Including interaction terms could capture these nuanced relationships.
# 4.Explore non-linear relationships: The current model assumes categorical effects, but exploring non-linear relationships with continuous versions of the variables might provide additional insights.
# 5.Split the dataset: Creating training and testing datasets would allow for proper validation of the model's predictive performance.
# 
# Despite these potential improvements, the current model is already quite strong with its high R-squared value, suggesting that TV promotion budget is indeed a major driver of Sales in this dataset.

# ### Perform a one-way ANOVA test
# 
# With the model fit, run a one-way ANOVA test to determine whether there is a statistically significant difference in `Sales` among groups. 

# In[13]:


# Create an one-way ANOVA table for the fit model.

anova_table = sm.stats.anova_lm(results, typ=2)
print("\nANOVA Table:")
print(anova_table)


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Review what you've learned about how to perform a one-way ANOVA test.
# 
# </details>

# <details>
# <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# There is a function in `statsmodels.api` (i.e. `sm`) that peforms an ANOVA test for a fit linear model.
# 
# </details>

# <details>
# <summary><h4><strong>Hint 3</strong></h4></summary>
# 
# Use the `anova_lm()` function from `sm.stats`. Specify the type of ANOVA test (for example, one-way or two-way), using the `typ` parameter.
#    
# 
# </details>

# **Question:** What are the null and alternative hypotheses for the ANOVA test?

# The null hypothesis (H₀) for the ANOVA test is that there is no difference in mean Sales across the three TV promotion budget categories (Low, Medium, and High). Mathematically, this can be written as:
# H₀: μLow = μMedium = μHigh
# 
# The alternative hypothesis (H₁) is that at least one of the TV promotion budget categories has a mean Sales value that differs significantly from the others. In other words, at least one of the means is different:
# H₁: At least one μi ≠ μj (where i and j represent different TV budget categories)

# **Question:** What is your conclusion from the one-way ANOVA test?

# Based on the ANOVA results, we reject the null hypothesis. The extremely small p-value (8.805550e-256, effectively zero) and very large F-statistic (1971.455737) provide overwhelming evidence that there are statistically significant differences in mean Sales across the different TV promotion budget categories. The evidence against the null hypothesis is extremely strong, suggesting that TV promotion budget has a significant effect on Sales.

# **Question:** What did the ANOVA test tell you?

# The ANOVA test tells us that:
# 
# 1.There are statistically significant differences in mean Sales between at least some of the TV promotion budget categories.
# 2.The relationship between TV promotion budget and Sales is very unlikely to be due to random chance (p-value effectively zero).
# 3.The model with TV promotion budget categories explains a substantial portion of the variation in Sales, as indicated by the large sum of squares for the model (4.052692e+06) compared to the residual sum of squares (5.817589e+05).
# 4.The F-statistic of 1971.46 is extremely large, suggesting a very strong effect of TV promotion budget on Sales.
# 
# However, the ANOVA test alone doesn't tell us which specific TV budget categories differ from each other - for that, we would need to examine the post-hoc tests (like Tukey's HSD) to make pairwise comparisons between the three budget categories.

# ### Perform an ANOVA post hoc test
# 
# If you have significant results from the one-way ANOVA test, you can apply ANOVA post hoc tests such as the Tukey’s HSD post hoc test. 
# 
# Run the Tukey’s HSD post hoc test to compare if there is a significant difference between each pair of categories for TV.

# In[14]:


# Perform the Tukey's HSD post hoc test.

tukey = pairwise_tukeyhsd(endog=data['Sales'],
                          groups=data['TV'],
                          alpha=0.05)
print("\nTukey's HSD Post Hoc Test:")
print(tukey)


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Review what you've learned about how to perform a Tukey's HSD post hoc test.
# 
# </details>

# <details>
# <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# Use the `pairwise_tukeyhsd()` function from `statsmodels.stats.multicomp`.
# 
# </details>

# <details>
# <summary><h4><strong>Hint 3</strong></h4></summary>
# 
# The `endog` argument in `pairwise_tukeyhsd` indicates which variable is being compared across groups (i.e., `Sales`). The `groups` argument in `pairwise_tukeyhsd` tells the function which variable holds the group you’re interested in reviewing.
# 
# </details>

# **Question:** What is your interpretation of the Tukey HSD test?

# The Tukey HSD test results show statistically significant differences in mean Sales between all pairs of TV promotion budget categories. Specifically:
# 
# 1.High vs. Low: The mean difference is -208.8133, indicating that High TV budget promotions result in sales that are about $208.8 million higher than Low TV budget promotions. This difference is statistically significant (p-adj = 0.001) and the confidence interval (-216.637 to -200.9896) does not include zero.
# 2.High vs. Medium: The mean difference is -101.5061, indicating that High TV budget promotions result in sales that are about $101.5 million higher than Medium TV budget promotions. This difference is also statistically significant (p-adj = 0.001) with a confidence interval (-109.3204 to -93.6918) that excludes zero.
# 3.Low vs. Medium: The mean difference is 107.3072, indicating that Medium TV budget promotions result in sales that are about $107.3 million higher than Low TV budget promotions. This difference is statistically significant (p-adj = 0.001) with a confidence interval (99.7063 to 114.908) that does not include zero.
# 
# All comparisons have "reject = True," confirming that we can reject the null hypothesis of equal means for each pair of categories.

# **Question:** What did the post hoc tell you?**

# The post hoc Tukey HSD test provides several important insights:
# 
# 1.All three TV promotion budget categories differ significantly from each other in terms of their effect on Sales, creating a clear hierarchy of effectiveness: High > Medium > Low.
# 2.There is a progressive increase in Sales as TV promotion budget increases, with each step up in budget category resulting in substantial and statistically significant gains.
# 3.The largest difference in Sales occurs between High and Low TV budget categories (approximately $208.8 million), while the difference between consecutive categories (High vs. Medium and Medium vs. Low) is roughly $100-107 million in each case.
# 4.The adjusted p-values of 0.001 for all comparisons indicate that these differences are highly unlikely to be due to random chance, even after controlling for multiple comparisons.
# 5.The narrow confidence intervals suggest high precision in our estimates of the mean differences between categories.
# 
# These results strongly support the conclusion that higher TV promotion budgets are associated with significantly higher Sales, with each level of increase providing substantial returns.

# ## **Considerations**
# 
# **What are some key takeaways that you learned during this lab?**
# 
# During this lab, several important insights emerged:
# 
# 1.The power of one-way ANOVA for comparing means across multiple groups, allowing us to statistically test differences between three TV promotion budget categories simultaneously.
# 2.The importance of checking model assumptions before interpreting results. In this case, we observed violations of both normality and homoscedasticity, which should be considered when evaluating the reliability of our findings.
# 3.The value of post-hoc tests (like Tukey HSD) in providing specific pairwise comparisons after finding significant overall effects in ANOVA. These tests identified exactly which budget categories differed from each other.
# 4.The strong relationship between TV promotion budget and Sales, as evidenced by the high R-squared value (87.4%), indicating that TV promotion strategy is a crucial driver of sales performance.
# 5.The clear hierarchical effect of TV promotion budgets, with each step up in budget category (Low to Medium to High) associated with significant increases in Sales.
# 6.The complementary nature of different statistical tools (boxplots, regression, ANOVA, post-hoc tests) in building a comprehensive understanding of data relationships.
# 7.The practical application of statistical methods to real-world marketing questions, demonstrating how data analysis can inform business decisions.
# 
# 
# **What summary would you provide to stakeholders? Consider the statistical significance of key relationships and differences in distribution.**
# 
# Summary for Stakeholders:
# Our analysis of marketing promotion data reveals compelling evidence that TV promotion budget has a substantial and statistically significant impact on sales performance. Here are the key findings:
# 
# 1.TV promotion budget explains approximately 87% of the variation in sales figures, making it an exceptionally strong predictor of sales outcomes.
# 2.We found clear, statistically significant differences in sales performance across all three TV budget categories (High, Medium, and Low), with each level providing distinct sales advantages.
# 3.High TV budget promotions outperform Medium budget promotions by approximately $101.5 million in sales on average, while Medium budget promotions outperform Low budget promotions by approximately $107.3 million.
# 4.The total sales advantage of High over Low TV budget promotions is approximately $208.8 million, representing a substantial return on promotional investment.
# 5.These findings are highly statistically significant, with p-values effectively zero, indicating that these differences are extremely unlikely to be due to random chance.
# 
# Based on these findings, we recommend:
# 
# 1.Prioritizing High TV budget allocations where possible, as they consistently deliver the strongest sales performance.
# 2.When budget constraints exist, even increasing from Low to Medium TV budgets can yield significant sales improvements.
# 3.Developing a tiered marketing strategy that optimizes TV promotion spending based on expected sales returns.
# 4.Conducting further analysis that incorporates other marketing channels (social media, radio, influencers) to develop a more comprehensive promotional strategy.
# 
# While our model is statistically robust, we should note some limitations, including violations of certain statistical assumptions that suggest additional factors may be at play. Future analyses could address these limitations and provide even more refined insights.
# 

# #### **Reference**
# [Saragih, H.S. *Dummy Marketing and Sales Data*](https://www.kaggle.com/datasets/harrimansaragih/dummy-advertising-and-sales-data)

# **Congratulations!** You've completed this lab. However, you may not notice a green check mark next to this item on Coursera's platform. Please continue your progress regardless of the check mark. Just click on the "save" icon at the top of this notebook to ensure your work has been logged.

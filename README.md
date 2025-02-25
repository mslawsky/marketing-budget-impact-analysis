# Marketing Budget Impact Analysis 📊

**Marketing Channel Optimization Analysis**  
This project demonstrates how data-driven analysis can identify which marketing channels deliver the strongest sales impact, providing evidence-based guidance for budget allocation decisions.

![Analysis Overview](distribution-of-sales.png)

---

## Executive Summary & Key Findings 📄

Our analysis revealed a remarkably strong relationship between TV advertising and sales performance:

1. **TV Advertising ROI (R-squared: 0.999)**: ✨
   - **Impact Quantification**: Each $1M invested yields $3.56M in sales
   - **Statistical Significance**: Extremely high confidence (p < 0.001)
   - **Strategic Insight**: Clear evidence supporting TV as primary channel

2. **Channel Comparison**: 📈
   - **TV**: Strongest sales predictor with clear linear relationship
   - **Radio**: Moderate positive correlation with sales
   - **Social Media**: Weaker association with overall sales
   - **Resource Focus**: Prioritize TV while testing complementary channel strategies

3. **Model Validation**: ✅
   - Linearity assumption confirmed 
   - Normality of residuals verified
   - Homoscedasticity requirements satisfied
   - **Reliability**: Statistical foundation for decision-making

---

## Data Exploration: Understanding Sales Patterns 📊

![Sales Distribution](distribution-of-sales.png)

Initial data exploration revealed a relatively balanced sales distribution ranging from approximately 50 to 375 million dollars, with no strong skew in either direction. This well-structured data provided an excellent foundation for regression analysis.

---

## Statistical Analysis: Marketing Channel Impact Assessment 📈

### Relationship Identification
![Pairwise Relationships](pairwise-relationships.png)

The pairplot visualization clearly shows that TV has the strongest linear relationship with Sales compared to Radio and Social Media. This initial finding guided our selection of TV as the primary independent variable for our regression model.

### Model Development
![TV Budget vs Sales](linearity.png)

The scatterplot confirms an exceptionally strong linear relationship between TV Budget and Sales. The tight clustering of points along a virtually perfect line demonstrates the predictive power of TV advertising expenditure.

### Model Validation
![Residuals Analysis](histogram-of-residuals.png)

Statistical testing confirms the validity of our model:
- The histogram and Q-Q plot show that residuals follow an approximately normal distribution
- The predicted vs. actual values show minimal deviation

![Homoscedasticity Check](residuals-vs-fitted-values.png)

The consistent band of points across all fitted values indicates that the error variance is stable, satisfying the homoscedasticity assumption required for reliable linear regression.

---

## Business Applications & Strategic Impact 💡

The insights revealed through this analysis suggest several powerful strategic approaches:

1. **Investment Prioritization**: The data provides clear evidence for prioritizing TV advertising in the marketing mix

2. **ROI Forecasting**: With every $1M in TV advertising corresponding to $3.56M in sales, organizations can accurately project returns on marketing investments

3. **Testing Framework**: A structured approach to testing TV advertising strategies can further refine understanding of performance drivers

4. **Market Monitoring**: Regular analysis of the TV advertising-sales relationship can identify shifts in effectiveness over time

5. **Channel Integration**: Exploring synergies between TV and other marketing channels might reveal multiplicative effects

---

## Analysis Methodology 🔍

This project follows a rigorous analytical approach:

1. **Problem Definition**: Identifying which marketing channels most effectively drive sales growth

2. **Data Exploration**: Examining distributions and relationships in marketing campaign data

3. **Statistical Modeling**: Using simple linear regression to quantify precise relationships

4. **Assumption Verification**: Testing for linearity, independence, normality, and homoscedasticity 

5. **Performance Measurement**: Evaluating model accuracy using R-squared, p-values, and confidence intervals

6. **Insight Generation**: Converting statistical findings into actionable business strategies

---

## Technical Resources 📁

- **Analysis Tools**: Python, Pandas, Matplotlib, Seaborn, Statsmodels
- **Statistical Methods**: Linear Regression, Assumption Testing, Hypothesis Testing
- **Key Files**: 
  - [Marketing Budget Analysis Notebook](https://github.com/mslawsky/marketing-budget-impact-analysis/blob/main/Marketing_Budget_Analysis.ipynb)
  - [Marketing Campaign Dataset](https://github.com/mslawsky/marketing-budget-impact-analysis/blob/main/marketing_and_sales_data_evaluate_lr.csv)

---

## The Power of Data-Driven Decisions ⚡

This project illustrates how statistical analysis reveals patterns that transform business decision-making:

- **Objectivity**: Reducing the influence of assumptions and preconceptions
- **Precision**: Quantifying exact relationships between marketing and revenue 
- **Evidence**: Building strategy on empirical foundations
- **Optimization**: Allocating resources for maximum impact

While this analysis focuses on marketing budget allocation, the data-driven approach demonstrated here applies across virtually every business domain.

---

## Contact ✉️

For inquiries about this analysis:
- [LinkedIn Profile](https://www.linkedin.com/in/melissaslawsky/)
- [Client Results](https://melissaslawsky.com/portfolio/)
- [Email](mailto:melissa@melissaslawsky.com)

---

© Melissa Slawsky 2025. All Rights Reserved.

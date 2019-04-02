# 1. Overview.
<\br>
 The analysis of credit risk and the decision making for granting loans is one of the most important operations for money lenders/investor.
Based on tha historic data and borrowers profile, identify the risk and likelyhood of defaulter. Also, suggest interest rate to achieve the desired
profit for the company based on the model outcomes.

# 2. Assumptions
2.1 I have considered loan status as a dependent variable which has 4 classes. I have transformed into 2 classes as "Fully Paid" and "Charged Off".
I have replaced "Does not meet the credit policy. Status:Charged Off" with "Charged Off" and simillary for "Fully Paid"

2.2 For the model building and calculation of credit score, I have considered all the important features such as loan_amt, interest_rate, grade etc.

2.3 From data, I assumed min and maximum credit score as 1 and 999 with total range of 1-999.

2.4 I have drop certain features in feature selection phase such as emp_title, emp_length, purpose (in future scope I would consider), 
last_payment_d and other features listed in code considering their significance level. I have also dropped highly correlated featrures.

# 3. Recommendations and Findings
3.1 Grade is important factor in deciding approval of loan and interst rate. For instance, borrower of Grade A are more likely 
to pay their loans on time and people of Grade G are more risky as they dont pay the loan on time.

3.2 Target more borrowers of Grade C, D and E as there is less risk and high interest rate compared to borrowers of Grade A and Grade G. 
This will ensure to get the desired profit for the company.

3.3 From map, CA and NY state has highest number with higher rate. A company can target more customer to achieve more profit as
there is less risk and high ROI.


# 4. Future Score
4.1 When the business model is understood in more deatiled way then the credit score calulation can be done in more accurate way.

4.2 Use of descriptive statistics of columns such as find its weight of evidence and Information value (IV) to generate the credit score.

4.3 Use other robust models such as SVM, and boosting and techniques to improve performance of models and anticipate more accurate result to risk less.
Other visualization would also help to get insights of data.

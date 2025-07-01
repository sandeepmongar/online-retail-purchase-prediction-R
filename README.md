# online-retail-purchase-prediction-R
This project analyzes the Online Retail II dataset using R to predict customer purchases. It uses Ridge Regression and Decision Tree models, achieving over 98% accuracy. It includes data preprocessing, cross-validation, visualizations, and evaluation to provide actionable business insights.

The project explores predictive modeling on the "Online Retail II" dataset to identify customer purchasing behavior. Implemented in R, it includes Ridge Regression and Decision Tree models, achieving over 98% accuracy. It demonstrates full pipeline preprocessing, binary classification, cross-validation, and business insights.

## Dataset

- **Source:** [UCI Machine Learning Repository - Online Retail II](https://archive.ics.uci.edu/ml/datasets/Online+Retail+II)
- **Records:** 541,910
- **Fields:** Invoice, StockCode, Description, Quantity, InvoiceDate, Price, Customer.ID, Country

##  Technologies Used

- R
- caret
- glmnet
- rpart
- ggplot2

## Preprocessing

- Handled 135,000+ missing values
- Transformed `Quantity` to binary: `1` (purchased), `0` (not purchased)
- Encoded categorical variables (`Country`) as factors
- Converted `InvoiceDate` to datetime
- Visualized `Price` distribution and outliers

## Models

### 1. Ridge Regression
- Regularized model using `glmnet`
- Cross-validated lambda tuning
- Accuracy: **98.05%**
- Insights: Strong recall, smooth regularization path

### 2. Decision Tree
- Implemented with `caret` and `rpart`
- Tuned with 5-fold cross-validation
- Accuracy: **98.08%**
- Insights: Excellent precision and interpretability

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- RMSE (for Ridge)

##  Visualizations

- Histogram and boxplot of `Price`
- Ridge regularization path
- Tree structure for decision model
- Confusion matrices


![image](https://github.com/user-attachments/assets/35a407b3-7d29-41e2-8afe-3dd2546a0b5d)


![image](https://github.com/user-attachments/assets/8548f261-b8fb-49a0-961e-7fa1e0783162)


## Outcome

Both models performed well, but Decision Tree had slightly better overall performance. This approach helps businesses forecast customer behavior and improve inventory and marketing strategies.

##  Author

Sandeep Monger

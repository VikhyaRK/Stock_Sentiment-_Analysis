# **Stock Sentiment Analysis and Price Prediction**

This project explores the relationship between social media sentiment and stock price movements. By analyzing stock-related tweets and historical stock data, the project predicts stock price trends and closing prices using machine learning models. Below is a step-by-step explanation of the workflow and methodology:

## **Project Workflow**

### **1. Data Collection**
- **Tweet Dataset (`stock_tweets.csv`)**: Contains stock-related tweets with timestamps, stock tickers, and company names.
- **Stock Data (`stock_yfinance_data.csv`)**: Includes historical stock prices (open, close, high, low, adjusted close) and trading volume for various stocks.

---

### **2. Data Preprocessing**
- **Tweets**:
  - Cleaned tweets by removing mentions, hashtags, links, special characters, and converting text to lowercase.
  - Performed sentiment analysis using VADER (SentimentIntensityAnalyzer) to assign sentiment scores:
    - **Positive Sentiment**: Scores > 0.
    - **Negative Sentiment**: Scores < 0.
    - **Neutral Sentiment**: Scores ≈ 0.
- **Stock Data**:
  - Converted dates to a consistent format (`YYYY-MM-DD`).
  - Engineered new features:
    - **Fluctuation**: Difference between daily high and low prices.
    - **Price Gain**: Difference between close and open prices.
    - **Total Valuation (EOD)**: Total traded value (`Volume × Close`).

---

### **3. Dataset Alignment**
- Created a unique `anchor` column by combining `Date` and `Stock Name` to align tweets with corresponding stock data.
- Merged the tweet dataset and stock dataset on this `anchor`.

---

### **4. Sentiment Analysis and Visualization**
- Analyzed the distribution of positive, negative, and neutral tweets.
- Created visualizations:
  - **Pie Charts**: Sentiment distribution.
  - **Line Plots**: Daily trends of positive/negative tweets and their correlation with stock valuation.

---

### **5. Company-Specific Analysis**
- Focused on individual companies (Tesla, Apple, and Taiwan Semiconductor):
  - Split datasets into positive and negative sentiment subsets.
  - Plotted the impact of sentiment on daily stock valuation for each company.
  - Generated correlation heatmaps to understand relationships between features (e.g., sentiment, price gain, volume).

---

### **6. Machine Learning Models**
- **Goal**: Predict stock prices and trends based on historical prices, tweet sentiments, and tweet volumes.
  
#### **Sliding Window Approach**
- Used a **3-day sliding window** to capture sequential dependencies in data.
  - **Features**: Historical prices, tweet sentiments, and tweet volumes.
  - **Target**: Stock closing price for the next day.

#### **Models Implemented**
1. **Support Vector Regression (SVR)**:
   - Captures non-linear relationships between features and stock prices.
   - Achieved reasonable accuracy for predicting closing prices.
   
2. **Linear Regression**:
   - Simple and interpretable model used for comparison.
   - Provided baseline results for regression tasks.

3. **Artificial Neural Networks (ANNs)**:
   - Built deep learning models for:
     - **Regression**: Predict closing prices.
     - **Classification**: Predict stock price trends (up or down) relative to previous prices.
   - Achieved high accuracy with multiple hidden layers and activation functions.

---

### **7. Evaluation**
- **Performance Metrics**:
  - **R² Score**: Assessed the accuracy of regression models.
  - **Slope Matching**: Compared the predicted and actual price trends (slopes) to calculate trend prediction accuracy.
- **Visual Comparison**:
  - Plotted actual vs. predicted stock prices to visualize model performance.

---

## **Key Takeaways**
- Public sentiment significantly impacts stock price movements.
- Machine learning models (especially ANNs) effectively leverage sentiment and historical data for stock price prediction.
- Sentiment analysis adds value to stock market forecasting by capturing investor sentiment in real-time.

---

## **Technologies Used**
- **Programming Language**: Python
- **Libraries**: 
  - **Data Manipulation**: `pandas`, `numpy`
  - **Visualization**: `matplotlib`, `seaborn`
  - **Natural Language Processing**: `nltk`
  - **Machine Learning**: `sklearn`, `keras`

---
layout: default
title: Proposal
description: 
---

## Introduction and Background

### Literature Review
We will be looking at data from all past airplane departures to extract meaningful patterns about airline and flight delays. There has been a lot of past research done on this topic and many have proposed applying decision tree models such as AdaBoost and as well as combining both logistic regression and decision tree models to make predictions [3,4].

### Dataset Description
We will be using data from the Bureau of Transportation Statistics to ensure that we have the most accurate and up-to-date data (June 2024). We chose January 2005 as our starting date as we found that “passenger travel by commercial airlines did not recover until March 2004 when the number of passengers enplaned returned to the August 2001 level” [1]. This dataset has 100+ features and includes information about the airline, weather, airport and has over 500,000 data points per month. 

### Dataset Link
View it [here↗](https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FGJ&QO_fu146_anzr=b0-gvzr)

## Problem Definition
### Problem
Flight delays are a significant problem in the aviation industry, impacting millions of passengers and resulting in economic losses and operational challenges. In 2007, the overall direct cost on airlines and passengers amounted to $28.9 billion, with an additional impact of $4 billion on the GDP [2]. 

### Motivation
There is a clear need for machine learning solutions that can learn from historical data and identify patterns to provide early warning of potential delays. A reliable prediction model could help airlines optimize their operations, allow passengers to plan their journeys more effectively, and ultimately reduce the economic impact of delays.

## Methods

### Data Preprocessing Methods
Data cleaning will replace any negative delay time (denoting early departure/arrival) as 0. We will also fill in any missing delay time (in minutes) with 0. PCA will extract features that are the most important in determining flight delays. We’ll accordingly filter out features that are unnecessary in prediction of on-time and delayed flights. Then, we’ll downsample on-time flights and upsample delayed flights to rebalance the dataset and mitigate bias.

### ML Algorithms and Models
The random forest algorithm is an ensemble learning method that combines multiple decision trees to improve accuracy and reduce overfitting. Next, the flexibility in kernel functions allows support vector machines (SVMs) to model complex, non-linear relationships between features, including weather, carrier, and delay types, contributing to better predictions. Lastly, given the large dataset, which includes flights from 2003 to 2024, neural networks could outperform random forest and SVMs by capturing deeper patterns. We will use the scikit-learn and TensorFlow libraries to implement the algorithms. For random forest and SVMs, we will use RandomForestClassifier and SVC from scikit-learn. In TensorFlow, the core classes for neural networks are tf.keras.Sequential, tf.keras.Model, and tf.keras.layers.

## Results and Discussion

We’ll use cross-validation to test our model by grouping unused data points (~20% of the dataset) from our random forest into a testing dataset. The ratio of the number of correct to incorrect predictions can evaluate the accuracy of our model. We will also use precision to express the testing results in terms of the positive classifications independently (e.g. ratio of flights correctly declared with a delay to all flights predicted to have a delay).  Finally, we’ll use recall to express how many of the delayed flights were correctly identified as delayed.

### Project Goals
Our goal is to maximize the recall value in our model evaluation (>90%). Incorrectly identifying a flight to have no delay could pose planning issues for customers. However, incorrectly identifying flights to have a delay also poses economic problems for the airline. We’ll also optimize for a high precision value, but not at the expense of our customers.


## Gantt Chart
View it [here↗](https://docs.google.com/spreadsheets/d/1IldWabWzaao4a45LP-cbYl0E8GFJnbYj/edit?gid=1317918749#gid=1317918749)

## Contribution Table

| Name        | Proposal Contributions          |
|:-------------|:------------------|
| Rishi Borra           | Introduction & Background |
| My Phung | Problem Definition   |
| Long Lam           | Methods: Data Preprocessing      |
| Joseph Thomas           | Methods: ML Algorithms |
| Aziz Albahar           | Results and Discussions |

## References
[1] “Twenty Years Later, How Does Post-9/11 Air Travel Compare to the Disruptions of COVID-19? | Bureau of Transportation Statistics,” www.bts.gov, Sep. 10, 2021. https://www.bts.gov/data-spotlight/twenty-years-later-how-does-post-911-air-travel-compare-disruptions-covid-19

[2] M. O. Ball, C. Barnhart, M. Dresner, and Augusto Voltes, “Total Delay Impact Study: A Comprehensive Assessment of the Costs and Impacts of Flight Delay in the...,” ResearchGate, Oct. 2010. https://www.researchgate.net/publication/272202358_Total_Delay_Impact_Study_A_Comprehensive_Assessment_of_the_Costs_and_Impacts_of_Flight_Delay_in_the_United_States (accessed Oct. 04, 2024).
‌
[3] S. Choi, Y. J. Kim, S. Briceno and D. Mavris, "Prediction of weather-induced airline delays based on machine learning algorithms," 2016 IEEE/AIAA 35th Digital Avionics Systems Conference (DASC), Sacramento, CA, USA, 2016, pp. 1-6, doi: 10.1109/DASC.2016.7777956. keywords: {Meteorology;Delays;Atmospheric modeling;Data models;Predictive models;Training;Schedules}

[4] V. Natarajan, S. Meenakshisundaram, G. Balasubramanian and S. Sinha, "A Novel Approach: Airline Delay Prediction Using Machine Learning," 2018 International Conference on Computational Science and Computational Intelligence (CSCI), Las Vegas, NV, USA, 2018, pp. 1081-1086, doi: 10.1109/CSCI46756.2018.00210. keywords: {Delays;Airports;Atmospheric modeling;Logistics;Predictive models;Meteorology;Mathematical model;Flight delay prediction;Logistic regression;Decision tree algorithm;Analytical modeling;Delay evaluation}

[back ⏎](./)

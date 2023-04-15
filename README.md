# UmojaHack-Africa-2022-Begineer-Challenge

This notebook contains a code + some modifications to the solution i submitted while competing on Umoja Hack Africa(A Biggest inter-universities Data science competition in Africa) 2022 Beginner challenge organised by zindi Africa.

# OVERVIEW

AirQo’s air quality sensing network has more than 120 low-cost devices deployed across Uganda; in most cases, these devices are deployed in unmonitored or perilous environments. These low-cost electronic devices are susceptible to breakdown caused by communication malfunction, aging, wear and tear, manufacturing deficiencies, incorrect calibration, mishandling and other external environmental factors. Faults lead to data inaccuracies and data loss, which impacts decisions and policies that could significantly impact people’s lives.

Device failure detection and monitoring is critical to AirQo’s business; faulty devices need to be identified, isolated and fixed or replaced with urgency. Data received from a device can be used to identify whether the device is working correctly or not.

In this challenge, the task is to develop a classification model to identify a device has an off set fault or not, regardless of the device. The model can be used by AirQo to automatically flag a device that is returning faulty data.

# DATASET
The Dataset is split into training and testing datasets.The train file contains approximately 300,000 readings and the test contains approximately 100,000 readings.Each row in the training set consist of the following informations;

1. ID
2. Datetime
3. Sensor1 PM2.5 
4. Sensor2 PM2.5 
5. Temperature
6. Relative Humidity
7. Offset fault (Target)

The testing set has the same features as the training set, except that it does not include the offset fault feature, as this is the target variable that we are to predict.

# APPROACH

To solve this problem, I used a simple Gradient Boosting model. I first cleaned the data by removing missing values and outliers. Then,extract some features,used Label binarizer encoding to convert categorical variables into numerical ones. Finally, I split the training set into a training set and a validation set and use stratified shuffle split to them into 10 folds and trained the Gradient Boosting model on the training set. I evaluated the model on the validation set using the F1 score.

# Results
My final model achieved an f1 score of 0.9169 on the validation set and 0.9155 on training set. I submitted my predictions for the test set to Zindi and achieved an F1 score of 0.8834 on the leaderboard.

# Conclusion
Participating in this challenge was a great learning experience for me. I gained valuable hands-on experience in building a machine learning model and working with real-world datasets. I am grateful for the opportunity and look forward to participating in more challenges in the future.

Feel free to check out my code in this repository and let me know your thoughts!

# Building Automated MLOps Pipeline with Spark Integration for News Category Prediction
Welcome to the Spark-NewsCategoryPrediction-PickleModel repository
A pretrained model in Pickle format is executed in Spark to predict the News category.

         This project is designed to predict the News category.

The purpose of my project is to develop a predictive spark model that can accurately categorize news articles
into different categories. The model is implemented using a pretrained model in Pickle format and executed
in a Spark environment. By utilizing the power of Spark's distributed computing capabilities, we aim to
efficiently process large volumes of news data and provide real-time predictions on the category of news articles.
---------------------------------------------------------------------------------------------------------------------
Project Details:
1.MLOps Pipeline:Developed a robust MLOps pipeline that periodically updates the training dataset 
  from MongoDB collections. The pipeline automates data preprocessing,model training 
  using Logistic Regression, testing,model evaluation and deployment using Python libraries.
2.Data Update Schedule:Implemented a dynamic data update schedule to update the training
  data every 6 months to ensure the model stays up-to-date with fresh information.It improved model performance.
3.Data Preprocessing:Implemented data preprocessing techniques like lowercasing,
 punctuation removal, lemmatization, and stop-word removal to clean the input text data
4.Implemented  feature extraction
4.Model Training :Trained model using Logistic Regression, achieving accurate categorization results
5. Conducted rigorous model testing and evaluation, resulting in highly accurate predictions.
6.Model Deployment:Deployed the trained model and vectorizer as pickle files for easy integration 
                 into a Spark environment.
7.Spark Integration: Integrated the trained model and vectorizer into a Spark application,
enabling scalable and distributed predictions on new data.
8.CI/CD Workflow: Configured a CI/CD workflow to trigger automatic updates to 
the MLOps pipeline when code is merged into the main branch,ensuring model updates are 
automatically propagated.
9.Developed a Django Rest API to enable real-time predictions of news article categories
 using the trained ML model.
----------------------------------------------------------------------------------------------
If you want to learn more about ML Models With Spark, please refer below posts.
https://neptune.ai/blog/apache-spark-tutorial

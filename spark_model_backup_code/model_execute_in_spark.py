#########################################################################################
#             Team consists of three members working on a project:
# Team member 1: Responsible for creating an API using Django REST Framework and testing it with Postman.
# Team member 2: In charge of developing a model and saving it in Pickle format.
# Me (team member 3): Tasked with executing the pretrained model, saved in Pickle format, using Spark.

#My task : model in Pickle format and execute in Spark to predict the category of news paragraphs.
# 1. Modle creation and creating a pickle file of the model are not my tasks.
#     (1.1), but I created a dummy file called news_categories.csv."
#     (1.2) Train the logistic regression model, and after that, save the model in a pickle format file.
#     called "second_classification_model.pkl"   

#2. A pretrained model in Pickle format is executed in Spark.
# 2.1) Creates a Spark session and Spark context.
# 2.2) Reads the model file using sc.binaryFiles and unpickles the data to retrieve the models.
# 2.3) Splits the input data into sentences and removes leading and trailing spaces.
# 2.4) Joins the cleaned sentences back into a single string.
# 2.5) Processes the data using the process_data function.
# 2.6) Creates a spark DataFrame from the processed data.
# 2.7) Loads the pickled vectorizer from the vectorizer file.
# 2.8) Defines a user-defined function (UDF) called predict that applies the vectorizer and the broadcast 
#       models to predict the category for each sentence.
# 2.9) Applies the UDF to the sparkDataFrame to add a "Predicted Category" column.
# 2.10) Selects the necessary columns for prediction.
# 2.11) Converts the spark DataFrame to JSON format.
# 2.12) Stops the Spark session.

 
#######################################################################################################
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import json

from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.types import StringType
from pyspark.ml import PipelineModel

# Additional imports for text processing
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

def process_data(data):    #Preprocesses the input data by performing the following steps:

    # Download stopwords and lemmatization data
    nltk.download('stopwords')  #after download disable this 
    nltk.download('wordnet')    #after download disable this 
    nltk.download('punkt')      #after download disable this 

    print("            PREPROCESSES THE INPUT DATA BY PERFORMING THE FOLLOWING STEPS: ")
    # 1. Converts the text to lowercase.
    text = data.lower()

    # Replace special characters and punctuation followed by a letter with a space and the letter itself
    text = re.sub(r"[^\w\s](?=[a-zA-Z])", " ", text) 
    print("Text after removing dots, commas, special characters, and punctuation with space: ---", text)

    # Remove remaining special characters and punctuation
    text = re.sub(r"[^\w\s]", "", text)
    print("Text after removing remaining special characters and punctuation: ---", text)

    # Remove numbers
    text = re.sub(r"\d+", "", text)
    print("Text after removing numbers: ---", text)

    # Tokenize the text into individual words or tokens
    tokens = word_tokenize(text)
    print("Text after tokenization: ---", tokens)

    # Remove stop words
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words]
    print("Text after removing stop words: ---", tokens)

    # Lemmatize the words to their base form
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    print("Text after lemmatization: ---", tokens)

    # Remove duplicate tokens
    tokens = list(set(tokens))
    print("Text after removing duplicate tokens: ---", tokens)

    # Join the processed tokens back into a clean text
    processed_text = " ".join(tokens)
   
    print("Processed text: ---", processed_text)
    print("type of process text",type(processed_text))
    return processed_text

#spark codes write inside spark_model() function
def spark_model(model_filepath,vectorizer_filepath,data):  #spark codes write inside spark_model() function
    # Create spark session
    spark = SparkSession.builder.getOrCreate()
    sc = spark.sparkContext

    # Unpickle, pkl file
    model_rdd_pkl = sc.binaryFiles(model_filepath)
    model_rdd_data = model_rdd_pkl.collect() 

    # Load and broadcast python object over spark nodes
    models = pickle.loads(model_rdd_data[0][1])
    broadcast_model = sc.broadcast(models)

    sentences = data.split('.')
 
    # Store sentences in the new_paragraphs list, again split " "
    new_sentences = [sentence.strip(" ") for sentence in sentences]
    print("New sentences: ",new_sentences)
    print("data type of new_sentences:",type(new_sentences))

    # Join the cleaned sentences back into a single string
    data = '.'.join(new_sentences)
    orginal_data=[data]

    # Process the data using the process_data function
    process_sentences=[process_data(data)]

    # Zip the lists together
    list_collect = list(zip(orginal_data,process_sentences))

    # Create a spark DataFrame from the zipped data
    df_pred = spark.createDataFrame(list_collect, ["Content","Process_content"])
    print("    SPARK DATA FRAME ")
    df_pred.show()

    # Load the pickled vectorizer
    with open(vectorizer_filepath, 'rb') as f:    
        vectorizer = pickle.load(f)

    # Create udf and call predict method on broadcasted model
    @udf(returnType=StringType())
    def predict(*new_sentences):
        for new_sentence in new_sentences:
            new_sentence_features = vectorizer.transform([new_sentence])
            predicted_category = broadcast_model.value.predict(new_sentence_features)[0]
            return predicted_category

    # Create predictions using spark udf
    df_pred = df_pred.withColumn("Predicted Category", predict(df_pred['Process_content']))
    print("    MODEL PREDICTION USING SPARK ")
    df_pred.show()

    # Select the necessary columns for prediction
    df_pred = df_pred.select("Content","Predicted Category")

    # Convert df_pred to JSON format
    df_pred_json = df_pred.toJSON()
    return df_pred_json
    
    # Stop the SparkSession
    sc.stop()

#main() function is used to call spark_model() function & pass argument as  user data & filepath of model & vectorizer
def main():  
    model_filepath = '/home/shan/spark_model_backup_code/second_classification_model.pkl' 
    vectorizer_filepath = '/home/shan/spark_model_backup_code/vectorizer.pkl'

    # Prompt the user to enter the news in a paragraph separated by dots
    userdata = input("Enter the news in paragraph (separated by dot): ")

    # Call the spark_model function with the provided file paths and user input
    output_json = spark_model(model_filepath, vectorizer_filepath, userdata)

    # Loop over the JSON output and print each row
    for row in output_json.collect():
        print(row)
main()  #it call main() function

#________________________________________________________________________________________________________________


#         #####     MODEL CREATION ...pickle file creation for spark model############################



# def model_creation(model_filepath,vectorizer_filepath,data):
# # def model_creation():    
#     # Read the data from the CSV file
#     df = pd.read_csv('/home/shan/news_categories.csv')

#      # Split the data into features and labels
#     x_train = df['unique_words']
#     y_train = df['category']

#     # Convert text data into numerical features using TF-IDF vectorization
#     vectorizer = TfidfVectorizer()
#     x_train = vectorizer.fit_transform(x_train)

#      # Split the dataset into training and testing sets
#     x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

#     # Train the logistic regression model
#     model = LogisticRegression()
#     model.fit(x_train, y_train)

#    # Save the trained model as a pickle file
#     with open('second_classification_model.pkl', 'wb') as f:
#         pickle.dump(model, f)

#     # Save the vectorizer as a pickle file
#     with open('vectorizer.pkl', 'wb') as f:
#         pickle.dump(vectorizer, f)

#      ##predicit output
#     ################
    
#     # with open(model_filepath, 'rb') as f:    
#     #     model = pickle.load(f)      

#     # with open(vectorizer_filepath, 'rb') as f:    
#     #     vectorizer = pickle.load(f)    
   
#     data = "We need a healthy life.healthy surroundings we make for our future"

#     # Process the input data and make predictions
#     l = {}
#     sentences = data.split('.')
    
#     # Remove leading and trailing spaces from each sentence
#     new_sentences = [sentence.strip(" ") for sentence in sentences]
#     print("New sentences:",new_sentences)
        
#     # Join the cleaned sentences back into a single string
#     data = '.'.join(new_sentences)

#     # Process the data using the process_data function
#     joined_sentences= process_data(data)

#     new_paragraph_features = vectorizer.transform([joined_sentences]) 
#     predicted_category = model.predict(new_paragraph_features)[0]
#     l.update({"Content":data,"Predicted category":predicted_category})
#     json_data = json.dumps(l)
#     # print(json_data)
#     return json_data
# # model_creation()

# def mainn():
#     model_filepath = '/home/shan/second_classification_model.pkl'
#     vectorizer_filepath='/home/shan/vectorizer.pkl'

#     data = input("Enter the news in paragraph (separated by dot): ")
#     # data = "Finance of india raise gobally.we need a healthy life.in sports sachin hit century.In technology 6G phone launch new week.technology india made progress.\
#     #                      Shewag score century.Us navy attack india"
#     output_json=model_creation(model_filepath,vectorizer_filepath,data)

#     # Print the JSON data
#     print(output_json)
# # mainn() 

# ###################################################################################################################

########################    use this function instead of spark udf function ##############################################
# # def predict(*new_sentences):
# #     l={}
# #     for new_sentence in new_sentences:
# #             new_sentence_features = vectorizer.transform([new_sentence])
# #             predicted_category = broadcast_model.value.predict(new_sentence_features)[0]
# #             # print(f"Predicted Category: {predicted_category}")
# #             l.update({new_sentence:predicted_category})
# #             return predicted_category
# #     for i,j in l.items():
# #         print(i ,":",j)
# # predict_udf = udf(predict, StringType())   
# # Create predictions using spark udf
# # df_pred = df_pred.withColumn("Predicted Category", predict_udf(df_pred['Content']))
# # df_pred.show()

#               #OR  use this function instead of spark udf function
# # def predict(*new_sentences):
# #     l={}
# #     for new_sentence in new_sentences:
# #             new_sentence_features = vectorizer.transform([new_sentence])
# #             predicted_category = broadcast_model.value.predict(new_sentence_features)[0]
# #             # print(f"Predicted Category: {predicted_category}")
# #             l.update({new_sentence:predicted_category})
# #             # return predicted_category
# #     # return l
# #     df = pd.DataFrame(l.items(), columns=['Content', 'Predicted Category'])
# #     print(df)
# #     # for i,j in l.items():
# #     #     print(i ,":",j)
# # predictionlist = predict(*new_sentences)
# ###################################################

###################create datset######################################################
# def test():
#     import csv

#     data = {
#         'sports': [
#             'victory', 'championship', 'athlete', 'tournament', 'score', 'goal', 'record', 'team', 'coach', 'match', 'medal', 'penalty', 'injury', 'title', 'league', 'defeat', 'spectator', 'stadium', 'rivalry', 'comeback', 'final', 'referee', 'training', 'transfer', 'celebration', 'fans', 'performance', 'doping', 'contract', 'highlight'
#         ],
#         'politics': [
#             'election', 'candidate', 'government', 'policy', 'voting', 'campaign', 'legislation', 'president', 'congress', 'senate', 'debate', 'corruption', 'diplomacy', 'reform', 'bill', 'inauguration', 'partisan', 'coalition', 'speech', 'opposition', 'vote', 'cabinet', 'protest', 'policy', 'constituency', 'justice', 'campaign trail', 'ballot', 'statecraft'
#         ],
#         'technology': [
#             'innovation', 'startup', 'artificial intelligence', 'digital', 'internet', 'software', 'hardware', 'gadget', 'robotics', 'cybersecurity', 'blockchain', 'automation', 'data', 'cloud', 'algorithm', 'virtual reality', 'augmented reality', 'nanotechnology', 'biotechnology', 'e-commerce', 'social media', 'encryption', 'prototype', 'tech industry', 'tech giant', 'tech startup', 'internet of things', 'computer science', 'cybercrime'
#         ],
#         'entertainment': [
#             'celebrity', 'movie', 'music', 'actor', 'actress', 'red carpet', 'premiere', 'box office', 'award', 'performance', 'album', 'television', 'director', 'producer', 'screenplay', 'pop culture', 'festival', 'entertainment industry', 'celebrity gossip', 'cinema', 'comedy', 'drama', 'musical', 'reality TV', 'blockbuster', 'chart-topping', 'hit song', 'star-studded'
#         ],
#         'business': [
#             'economy', 'company', 'stock market', 'entrepreneur', 'investment', 'financial', 'startup', 'marketplace', 'shareholder', 'merger', 'acquisition', 'profit', 'loss', 'revenue', 'CEO', 'market trend', 'banking', 'trade', 'industry', 'stock exchange', 'product launch', 'global market', 'business strategy', 'innovation', 'corporate', 'business development', 'business news', 'market analysis'
#         ],
#         'science': [
#             'research', 'discovery', 'experiment', 'scientist', 'technology', 'innovation', 'theory', 'space', 'biology', 'chemistry', 'physics', 'environment', 'genetics', 'astronomy', 'scientific study', 'data analysis', 'climate change', 'scientific breakthrough', 'scientific community', 'scientific research', 'quantum mechanics', 'STEM', 'scientific method', 'laboratory', 'scientific journal', 'science news', 'scientific evidence', 'scientific theory'
#         ],
#         'health': [
#             'wellness', 'disease', 'medical', 'healthcare', 'pandemic', 'vaccine', 'nutrition', 'fitness', 'mental health', 'public health', 'healthcare system', 'health policy', 'medical research', 'hospital', 'doctor', 'patient', 'pharmaceutical', 'health condition', 'health crisis', 'medical breakthrough', 'medical study', 'healthy'
#         ]
#     }

#     # Define the CSV file path
#     csv_file = 'news_categories.csv'

#     # Write the dictionary to the CSV file
#     with open(csv_file, 'w', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow(['category', 'unique Words'])
#         for category, words in data.items():
#             writer.writerow([category, ','.join(words)])

#     print(f"Data written to {csv_file} successfully.")    
   
# # test() 


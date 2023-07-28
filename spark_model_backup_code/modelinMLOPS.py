
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

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

import time,datetime
# # pip install pymongo
import pymongo 


def check_database():
    # Connect to MongoDB
    myclient = pymongo.MongoClient('mongodb+srv://shanvas786:toor@cluster0.un4re1j.mongodb.net/')
    mydb = myclient['modeldatabase']

    L =mydb.list_collection_names()
  
    d={}
    # d[1]=L[0]
    d[1]= {time.ctime(time.time()):L}
    print("inital collection:",d)
    i=1
    flag = 0
    while True:
        
        # Retrieve documents with a processed_message_id value of None
        documents=mydb.list_collection_names()
        
        for document in documents:
            i=i+1
            if document not in L:
                current_time = datetime.datetime.now()
                formatted_time = current_time.strftime("%I:%M %p, %Y-%m-%d")
                d[i] = {formatted_time:document}
                # d[i] = document
                L.append(document)
                
                mycol =mydb['info']
                mylist=[]
                mylist.append({"Time":formatted_time,"Updated_collection":document})
                x = mycol.insert_many(mylist)
                print(x.inserted_ids)
                while(True):
                    if document in mydb.list_collection_names() :
                        if mydb[document].estimated_document_count() == 0 :
                            print("New table: ",document," is found.. Please wait server database data is uploading..... ")
                        else:
                            print("server database data is uploading Start..... ")
                            flag = 1
                            time.sleep(2 * 60)  # Sleep for 3 minutes (3 * 60 seconds)
                            break
                    else:
                        myquery = { "Updated_collection":document }
                        mycol.delete_many(myquery)
                        break
            else:
                i = i -1                  
        print("-----updated database details------")
        print("Collection details in List:",L)
        print("Collection details in Dictionary:",d)
        if flag ==1:
            # return L[-1]
            break
        time.sleep(10)
    # return L[-1]  
# p=check_database() 
# print(p)
#--------------------------------------------------------------------------------------------------

# Step 1: Dataset Update
def update_dataset():
    # Connect to MongoDB
    myclient = pymongo.MongoClient('mongodb+srv://shanvas786:toor@cluster0.un4re1j.mongodb.net/')
    mydb = myclient['modeldatabase']
    
    def read_updated_table():
        mycol = mydb['info'] 
        mydoc = mycol.find().sort("_id",-1)[0] 
        # print(mydoc["Updated_collection"])
        return mydoc["Updated_collection"]

    def download_data(table):
        import json
        from bson import ObjectId

        mycol = mydb[table]    
        # Retrieve all documents from the collection
        documents = mycol.find()

        class JSONEncoder(json.JSONEncoder):
            def default(self, o):
                if isinstance(o, ObjectId):
                    return str(o)
                return super().default(o)

        # Export the documents to a JSON file
        with open('collection_data.json', 'w') as file:
            for document in documents:
                json.dump(document, file, cls=JSONEncoder)
                file.write('\n')
        print("Collection downloaded and saved as 'collection_data.json'")
        
    table= read_updated_table()
    download_data(table) 
#-------------------------------------------------------------------------------------------
def process_data(data):    #Preprocesses the input data by performing the following steps:

    # Download stopwords and lemmatization data
    # nltk.download('stopwords')  #after download disable this 
    # nltk.download('wordnet')    #after download disable this 
    # nltk.download('punkt')      #after download disable this 

    # print("            PREPROCESSES THE INPUT DATA BY PERFORMING THE FOLLOWING STEPS: ")
    sentences=data
    # Split the sentences using the dot as a delimiter
    sentence_list = sentences.split(".")
    # print("sentence_list: ",sentence_list)

    # Strip the spaces from each sentence
    stripped_sentences = [sentence.strip(" ") for sentence in sentence_list]
    # print("stripped_sentences: ",stripped_sentences)

    # Rejoin the sentences with the dot
    rejoin_data = ".".join(stripped_sentences)
    # print("result :",rejoin_data)

    # 1. Converts the text to lowercase.
    text = rejoin_data.lower()

    # Replace special characters and punctuation followed by a letter with a space and the letter itself
    text = re.sub(r"[^\w\s](?=[a-zA-Z])", " ", text) 
    # print("Text after removing dots, commas, special characters, and punctuation with space: ---", text)

    # Remove remaining special characters and punctuation
    text = re.sub(r"[^\w\s]", "", text)
    # print("Text after removing remaining special characters and punctuation: ---", text)

    # Remove numbers
    text = re.sub(r"\d+", "", text)
    # print("Text after removing numbers: ---", text)

    # Tokenize the text into individual words or tokens
    tokens = word_tokenize(text)
    # print("Text after tokenization: ---", tokens)

    # Remove stop words
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words]
    # print("Text after removing stop words: ---", tokens)

    # Lemmatize the words to their base form
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    # print("Text after lemmatization: ---", tokens)

    # Remove duplicate tokens
    tokens = list(set(tokens))
    # print("Text after removing duplicate tokens: ---", tokens)

    # Join the processed tokens back into a clean text
    processed_text = " ".join(tokens)
   
    # print("Processed text: ---", processed_text)
    # print("type of process text",type(processed_text))
    # Assign different values to caller based on the calling function
    if caller == 'readdata_call':
        return processed_text
    elif caller == 'modelload_call':
        return rejoin_data,processed_text
    # return rejoin_data,processed_text
    # return processed_text

def read_data():
    global caller
    caller = 'readdata_call' 
    # # df = pd.read_csv("E:\\jangoclass\\internship\\myproject\\spark_model_backup_code\\news_categories.csv")
    # # # print(df.head(10))
    # # print(df["unique_words"])

    df = pd.read_json("E:\\Projectss\\mlproject\\collection_data.json", lines=True)
   
    #****************************************************    
    # #  Show Top 5 Records
    # df.head()

    # # # Shape of the dataset
    # df.shape           #(209527, 6)

    # # Dataset information
    # print(df.columns)  #['link', 'headline', 'category', 'short_description', 'authors', 'date']
    #---------------------------------------    
#     # 3. Data Checks to perform
#    * Check missing values
#    * Check Duplicates
#    * Chek data type
#    * Check the number of unique values of each column
#    * Check statistics of data set
#    * Check various categories present in the different categorical column

    # #3.1 Check Missing values
        # df.isnull().sum()
    #     # OR
    # df.isna().sum()  

    # # if missing value present,then delete missing values
    # df.dropna( inplace=True) #remove null values

    # # 3.2 Check Duplicates
    # df.duplicated().sum()

    # # if duplicates present then remove duplicate
    # ndf=df.drop_duplicates()
    # ndf.duplicated().sum()

    # # 3.3 Check data types
    # ndf.info()

    # # 3.4 Checking the number of unique values of each column
    # df.nunique()

    # # 3.5 Check statistics of data set
    # df.describe()
    #----------------------------------------------
    # # 3.7 Exploring Data
    # print("Categories in 'link' variable:  ",end=" ")
    # print(df["link"].unique())

    # print("Categories in 'headline' variable:  ",end=" ")
    # print(df["headline"].unique())

    # print("Categories in 'category' variable:  ",end=" ")
    # print(df["category"].unique())

    # print("Categories in 'short_description' variable:  ",end=" ")
    # print(df["short_description"].unique())

    # print("Categories in 'authors' variable:  ",end=" ")
    # print(df["authors"].unique())

    # print("Categories in 'date' variable:  ",end=" ")
    # print(df["date"].unique())
    #-----------------------------------------------------
    # # define numerical and categorical columns
    # numeric_features =[feature for feature in df.columns if df[feature].dtype != 'O']
    # categorical_features =[feature for feature in df.columns if df[feature].dtype == 'O']

    # print("we have ",len(numeric_features) ,"numeric_features: ",numeric_features)
    # print("we have ",len(categorical_features) ,"categorical_features: ",categorical_features)
    #**********************************************************
    # Drop columns from the DataFrame
    columns_to_drop = ['link', 'headline', 'authors','date']  # Replace with the actual column names you want to drop
    n_df = df.drop(columns=columns_to_drop)
    # print(n_df.columns)    #['category', 'short_description']

    x = n_df["short_description"]
    
    processed_sentences = [process_data(sentence) for sentence in x]
    
    df_pred = pd.DataFrame(processed_sentences, columns=["short_description"])
    # OR
    # list_collect = list(zip(processed_sentences))
    # df_pred = pd.DataFrame(list_collect, columns=["short_description"])

    x = df_pred["short_description"]
 
    # print(x.unique())


    y =n_df["category"]
    # # Print the updated DataFrame
    print(y.unique())

    # n=n_df['category'].isnull().sum()
    # print(n)


    # df.dropna(subset=['category', 'short_Description'], inplace=True) #remove null values
    # print(n_df)
    return(x,y)

#----------------------------------------------------------------------------------------------
# Step 2: Model Training
def train_model(x,y):
    # Split the dataset into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # # Fit and transform the training data ,Convert text data into numerical features using TF-IDF vectorization 
    vectorizer = TfidfVectorizer()
    x_train = vectorizer.fit_transform(x_train)

    # Train the logistic regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, y_train)
    return(vectorizer,model,x_test,y_test)
# Step 3: Model Testing
def test_model(vectorizer,model,x_test,y_test):
    # # Predictions on the test set using the best model

    # ## Transform the testing data. Vectorize the test data
    x_test  = vectorizer.transform(x_test)
    y_pred = model.predict(x_test)
    
    # # y_pred = best_model.predict(x_test)
    
    # # #  Evaluate the Model
    accuracy = accuracy_score(y_test, y_pred)
    # confusion_mtx = confusion_matrix(y_test, y_pred)
    # classification_rep = classification_report(y_test, y_pred)

    print("Accuracy:", accuracy)
    # print("Confusion Matrix:\n", confusion_mtx)
    # print("Classification Report:\n", classification_rep)
#---------------------
# Step 4: Model Deployment
def deploy_model(vectorizer,model):
    # Save the trained model as a pickle file
    with open('second_classification_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    # Save the vectorizer as a pickle file
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)   

# Step 8: CI/CD Workflow
def mlops_pipeline():
    print("staring MLOPS pipeline function")
    check_database()
    # print("a:",a)
    
    # Step 1: Dataset Update
    update_dataset()

    x,y=read_data()

    # Step 2: Model Training
    vectorizer,model,x_test,y_test=train_model(x,y)
    
    # Step 3: Model Testing
    test_model(vectorizer,model,x_test,y_test)
    
    # Step 4: Model Deployment
    deploy_model(vectorizer,model)


def call_mlops_pipeline():
    # pip install GitPython
    import git
    repo = git.Repo(search_parent_directories=True)
    git_branch = repo.active_branch.name
    print(git_branch)

    if git_branch == "main":
        try:
            import schedule,time
        #    # #   Schedule the task to run every 5 days:
        #         # schedule.every(5).days.do(mlops_pipeline())
                # Schedule the task to run now
            # schedule.every(3).days.at("22:33").do(mlops_pipeline())  


            # Calculate the number of seconds in 6 months
            interval =  6 * 30 * 24 * 60 * 60  # 6 months * 30 days * 24 hours * 60 minutes * 60 seconds

            # Schedule the task to run after 6 months 
            schedule.every(interval).seconds.do(mlops_pipeline())

            
            # # Schedule the task to run after 6 months at 22:33
            # schedule.every(interval).seconds.at("22:33").do(mlops_pipeline())

        #     # #Run the scheduler continuously:
            while True:
                schedule.run_pending()
                # schedule.run_all()
                time.sleep(10)
            
        #     # Run the pending tasks once
            # schedule.run_all()
            # mlops_pipeline()
        except Exception as e:
            print("An error occurred:", str(e))    
    else:
        print("Not on the main branch. Skipping the CI/CD workflow.")    

    #create branch    -----  git branch sub-branch1
    #                 ------ git checkout sub-branch1

# call_mlops_pipeline()
#------------------------------------------------------------------------
def model_load(model_filepath,vectorizer_filepath,data):
    import pickle
    global caller
    caller = 'modelload_call'
     #predicit output
    ###############
    
    with open(model_filepath, 'rb') as f:    
        model = pickle.load(f)      

    with open(vectorizer_filepath, 'rb') as f:    
        vectorizer = pickle.load(f)    
   
    l = {}

     # Process the data using the process_data function
    rejoin_data,process_sentences= process_data(data)

    new_paragraph_features = vectorizer.transform([process_sentences]) 
    predicted_category = model.predict(new_paragraph_features)[0]
    l.update({"Content":rejoin_data,"Predicted_category":predicted_category})
    json_data = json.dumps(l)
    j_data=json.loads(json_data)
    # print(json_data)
    return j_data

def predict():
    model_filepath = 'E:\\Projectss\\mlproject\\second_classification_model.pkl'
    vectorizer_filepath='E:\\Projectss\\mlproject\\vectorizer.pkl'
    data = input("Enter the news in paragraph (separated by dot): ")
    # data ="sachin score 100.he played well.he get a trophy"
    output_json=model_load(model_filepath,vectorizer_filepath,data)
    for i,j in output_json.items():
        print(i,":",j)
# predict() 
# Main execution
if __name__ == "__main__":
    call_mlops_pipeline()
    predict()

# ###################################################################################################################
# #spark codes write inside spark_model() function
# def spark_model(model_filepath,vectorizer_filepath,data):  #spark codes write inside spark_model() function
#     global caller
#     caller = 'sparkmodel_call'
#     # Create spark session
#     spark = SparkSession.builder.getOrCreate()
#     sc = spark.sparkContext

#     # Unpickle, pkl file
#     model_rdd_pkl = sc.binaryFiles(model_filepath)
#     model_rdd_data = model_rdd_pkl.collect() 

#     # Load and broadcast python object over spark nodes
#     models = pickle.loads(model_rdd_data[0][1])
#     broadcast_model = sc.broadcast(models)


#     # sentences=data
#     # # Split the sentences using the dot as a delimiter
#     # sentence_list = sentences.split(".")
#     # print("sentence_list: ",sentence_list)

#     # # Strip the spaces from each sentence
#     # stripped_sentences = [sentence.strip(" ") for sentence in sentence_list]
#     # print("stripped_sentences: ",stripped_sentences)

#     # # Rejoin the sentences with the dot
#     # rejoin_data = ".".join(stripped_sentences)
   
#     # orginal_data=[rejoin_data]

#     # # Process the data using the process_data function
#     # process_sentences=[process_data(rejoin_data)]

#     # Process the data using the process_data function
#     rejoin_data,process_sentences= process_data(data)

#     # Zip the lists together
#     list_collect = list(zip([rejoin_data],[process_sentences]))

#     # Create a spark DataFrame from the zipped data
#     df_pred = spark.createDataFrame(list_collect, ["Content","Process_content"])
#     print("    SPARK DATA FRAME ")
#     df_pred.show()

#     # Load the pickled vectorizer
#     with open(vectorizer_filepath, 'rb') as f:    
#         vectorizer = pickle.load(f)

#     # Create udf and call predict method on broadcasted model
#     @udf(returnType=StringType())
#     def predict(*new_sentences):
#         for new_sentence in new_sentences:
#             new_sentence_features = vectorizer.transform([new_sentence])
#             predicted_category = broadcast_model.value.predict(new_sentence_features)[0]
#             return predicted_category

#     # Create predictions using spark udf
#     df_pred = df_pred.withColumn("Predicted Category", predict(df_pred['Process_content']))
#     print("    MODEL PREDICTION USING SPARK ")
#     df_pred.show()

#     # Select the necessary columns for prediction
#     df_pred = df_pred.select("Content","Predicted Category")

#     # Convert df_pred to JSON format
#     df_pred_json = df_pred.toJSON()
#     return df_pred_json
    
#     # Stop the SparkSession
#     sc.stop()

# #main() function is used to call spark_model() function & pass argument as  user data & filepath of model & vectorizer
# def main():  
#     model_filepath = '/home/shan/spark_model_backup_code/second_classification_model.pkl' 
#     vectorizer_filepath = '/home/shan/spark_model_backup_code/vectorizer.pkl'

#     # Prompt the user to enter the news in a paragraph separated by dots
#     userdata = input("Enter the news in paragraph (separated by dot): ")

#     # Call the spark_model function with the provided file paths and user input
#     output_json = spark_model(model_filepath, vectorizer_filepath, userdata)

#     # Loop over the JSON output and print each row
#     for row in output_json.collect():
#         print(row)

    
#     # # Loop over the JSON output and print each row
#     # for row in output_json.collect():
#     #     row_dict = json.loads(row)
#     #     for key, value in row_dict.items():
#     #         print(key, ":", value)  

       
# main()  #it call main() function


#________________________________________________________________________________________________________________
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


#####################################################
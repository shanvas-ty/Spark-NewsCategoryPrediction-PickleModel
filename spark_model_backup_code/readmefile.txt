(1).change the filepath of  model_filepath & vectorizer_filepath   inside main() function
(2) After the program"modelinMLOPS.py" runs, it automatically downloads stopwords and lemmatization data. 
   Once it's downloaded, we comment out the code for downloading stopwords and lemmatization data inside the process data() function.

(3).To install the dependencies listed in a requirements.txt file, you can use the pip command. Here's how you can install the requirements:
----------------------------------------------------------------------------------------------------------------------------------------
   1.Open your command-line interface or terminal.

   2.Navigate to the directory where your project is located or where the requirements.txt file is located.

   3.Activate your project's virtual environment (if applicable). This step ensures that the packages are installed within the desired environment.

   4.Run the following command to install the requirements:
                                                      pip install -r requirements.txt

         This command will read the requirements.txt file and install all the listed packages along with their specific versions.


##############################################################################

(4).To run the Spark program model_execute_in_spark.py after installing the requirements, you can follow these steps:
-----------------------------------------------------------------------------------------------------------------

   1.Ensure that you have a Spark cluster or Spark installation set up and running. Make sure that you have the necessary Spark binaries and environment variables configured.

   2.Open your command-line interface or terminal.

   3.Navigate to the directory where your model_execute_in_spark.py file is located.

   4.Activate your project's virtual environment (if applicable). This step ensures that the installed packages are used within the desired environment.

   5.Run the following command to execute the Spark program:         
                                                  spark-submit model_execute_in_spark.py
                                #Note :- here we mention the filepath of the spark program
                                    eg :-spark-submit filepath/model_execute_in_spark.py                  
        The spark-submit command is used to submit a Spark application to a cluster or local Spark installation. It will automatically handle the Spark setup and execution.

    6.Wait for the program to run. Spark will execute the code in model_execute_in_spark.py using the configured Spark cluster or local installation.

Make sure that your Spark program is correctly written and refers to the necessary input data or models. Adjust the command as needed, depending on your specific Spark setup and requirements.

Note: Ensure that you have the appropriate Spark dependencies installed, as specified in your requirements.txt file.     

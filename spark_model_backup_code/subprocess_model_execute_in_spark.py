import subprocess

# Define the command to execute the programpy
# command = ["/usr/bin/python3", "/home/shan/spark_model_backup_code/model_execute_in_spark.py"]
command = ["python3", "/home/shan/spark_model_backup_code/model_execute_in_spark.py"]

# Execute the program using subprocess
subprocess.run(command, check=True)




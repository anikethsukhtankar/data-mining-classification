# Data Mining Classification
The goal of the project is to increase familiarity with the classification packages, available in R to do data mining analysis on real-world problems. Several different classification methods were used on the given Life Expectancy dataset. The dataset was obtained from the Wikipedia website. The continent column was added as per the requirements to be used as class label. kNN, Support Vector Machine, C4.5 and RIPPER were the classification methods used on the data set.

Authors

Aniketh Sukhtankar (UF ID 7819 9584)
-------------------------------------------------------
 INTRO TO DATA MINING - PROJECT 1 CLASSIFICATION 
-------------------------------------------------------

CONTENTS OF THIS FILE 
---------------------
    
 * Introduction
 * Requirements
 * Installation and Configuration
 * Developer information


INTRODUCTION
------------
The main project folder contains the following files:
```
          - Project1_Classification.R
          - Project1_Classification.xlsx
	  - Project1_Classification_Report.pdf
          - Readme.txt
```


REQUIREMENTS
------------
RStudio (latest version)

The following need to be installed to run the project:
* rJava
* RWeka
* class
* zeallot
* gmodels
* caret
* e1071

These packages can be installed by running the following command 
-> install.packages(c("rJava","RWeka","class","zeallot","gmodels","caret","e1071"))

Once the "rJava" package is installed, a 'jre' folder is created on your computer under C: > Program Files > Java. Replace {jre_folder_name} in the command below with this folder name found on your computer, and execute the command in R console.
Sys.setenv(JAVA_HOME='C:\\Program Files\\Java\\{jre_folder_name}')

INSTALLATION AND CONFIGURATION
------------------------------
1. Set the DATASET_FILEPATH parameter in Project1_Classification.R to the complete file path of the input dataset (shared in the zipped folder).
2. Execute the functions in the beginning of the Project1_Classification.R file to load them into the R environment.
3. Set the seed to the following values : 1707, 1234, 1111, 2222 and 3333 to obtain the results in the report.
4. After setting each seed value run the script immediately after the functions in Project1_Classification.R to get the classification results.(Note : RIPPER and C4.5 might take some time to generate the optimal values)

DEVELOPER INFORMATION
---------------------

  Aniketh Sukhtankar (UF ID 7819 9584) asukhtankar@ufl.edu

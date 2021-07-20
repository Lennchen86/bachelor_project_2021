# Studying behavior through GPS analysis - a bachelor thesis project 
(Author: Yong Lenn Chen, Date: 20.07.2021)
***
Mobile devices are everywhere. We take them to every place we go and with the different sensors that have been built within each device, it enables us to track different daily activities, like sport or sleep. That means, everyday, millions and millions of data are being stored and analysed in order to enhance our experience with these applications. In this paper, we focus on the geographical location data and try to discover their behavior pattern by using a combination of the probabilistic topic model and document clustering. To do so, we collected data from 21 individuals and generated location labels according to the location places that they went. 
At the end, we validated our findings with silhouette score to see how well the clustering methods have been performed. From this study, some routines have been found from an individual, like "be at home from 0 am - 11 am" or "be at college from 0 am - 7 am and from 2 pm - 5 pm ". Our work has validated that, with the methods by combining topic model with document clustering, we were able to discover individuals' pattern of behaviors. 
***

## 1. How is the project organized? 
This project is organized in the following way. In total there are 21 users with their location data which can be added and seen inside the `./Resources/Parsed_Data` directory. To parse the data along the program, you can define it in the first row inside `main()` function in `main.py`. Within the  `./Resources` directory you can also find the directories `/dataframes` and `/images` for all the dataframes and images that we are able to create from this project. We have created four files, namely  
- createLDA_featurevector.py
- doc_clustering.py
- fineGrainLocation.py
- main.py

where `main.py` holds everything we need to run the project. One thing to note is that, currently, the program reads one data set at a time, which, in other words, means that it takes the geographical location data from one single user. Let me explain the most relevant part within each of the other three files, next.
### The `fineGrainLocation.py` file 
Inside this code file, we have all functions that we need to create fine grain location representation from the given data set. The most relevant functions are 

| Function | Description |
| --- | --- |
| label_generator_words(keyword, label)| The parameters that this function require are the keyword that has been extracted from the location and a list to store the labels. This function can be changed if needed. At the moment, we decided to use the labels: Campus, Home, Leisure, Road, Market, Others. The return of the function is an updated label list.| 
| label_generator_numbers(keyword, label) | The parameters that this function require are the keyword that has been extracted from the location with only a street name and a list to store the labels. Different keywords can be added manually to check and add it to require location labels. The return of the function is an updated label list.|
| set_labels(multipleLoc, location_data) | The function looks through each locations and from each of them it creates the location labels from the keywords. It doesn't return anything.|
| fine_grain_location(location_data)| From the `location_data`, it creates the 30 min. time intervals, fills in the gaps with 'N' when data were missing and looks for the highest occurred location label to set it as the most representative for the given time interval. The outcome is then stored in a dataframe, called `result` which is also the return variable of this function.|


### The `createLDA_featurevector.py` file 
This file includes all functions that we performed for the LDA model (with an arbitary k value to set for k amount of topics) and the feature vector that happens afterwards. 

### The `doc_clustering.py` file 
Here, we have three main functions for the three document clustering that we performed. The three methods are Isolation Forest, Agglomerative Clustering and Gaussian Mixture Model. 

## 2. Dependencies and Installation 
At this page, we have put the lists of dependencies and install instructions for people who wants to explore and work on this project for your use. 
### Dependencies 
To load this project, the following packages have to be installed:
- numpy 
- pandas 
- re
- matplotlib
- collections 
- datetime 
- sklearn 
- gensim 
- pprint
- boltons
### Installation
The package can be cloned using the following commands. 

With HTTPS:
```
git clone https://github.com/Lennchen86/bachelor_project_2021.git
cd bachelor_project_2021 
```
With SSH: 
```
git clone git@github.com:Lennchen86/bachelor_project_2021.git
```

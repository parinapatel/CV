# How to run the code

**In main function in run.py, assign related values to following 3 variables.**

**1. train_model = False**

**2. input_file = "\<the relative location of the file.>"**

**3. output_name = "\<name of file to store the predicted video to. DO NOT PROVIDE EXTENSION.>"**

*Extra notes:*

*For training purposes, I have used Dataset https://web.archive.org/web/20190901190223/http://www.nada.kth.se/cvap/actions/*

*To train, please do the following:*

*1. Download the dataset from the link.*

*2. Create a folder "Data MHI". NOTE THAT THE Folder structure is IMPORTANT.*

*3. Please name the main folder of training videos as "Data MHI" and every action name should be a subfolder.*

*4. Store all the extracted zips in corresponding action folder.*

*5. In main function, in run.py assign value True of variable "train_model".*

* *This will generate "training_dataset" folder with training and testing data.*
* *KNN model will be created in main structure with filename as "knn_trained_model.pkl".*

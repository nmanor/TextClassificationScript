# Text Classification Script
This script is used for **normalizing** text dataset, **extracting features** and **classifying** the text.

## Normalization 
Before extraction features you can normalize the dataset.
Available normalization:
* **C** - Spelling correction.
* **H** - HTML tags removal.
* **L** - Convert to lowercase.
* **P** - Punctuation removal.
* **R** - Repeated characters removal.
* **S** - Stop words removal.
* **T** - Stemming
* **M** - Lemmatizing.

## Features
Supported features:
* N-grams (including skip grams)

## Classifiying Methods
Supported methods:
* mlp - [Multilayer perceptron](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html).
* svc - [Support vector classifier](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html).
* lr - [Logistic regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).
* rf - [Random forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html).
* mnb - [Multinomial naive bayes](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html).


## How to run
#### Before starting:
* make sure that the folder structure is as shown:
	```
	<dataset name>
				├───results
				├───output
				├───cfgs
				├───dataset
				│   ├───training
				│   │   └───<data file>
				│	│	.
				│	│	.
				│	│	.
				│   │   └───<data file>
				│	├───testing (optional)
				│   │   └───<data file>
				│	│	.
				│	│	.
				│	│	.
				│   │   └───<data file>		
	```	

	* **results folder** - the results files (xlsx and data files) will be saved at the end of the run here.
	* **output folder**  - some features data files will be saved here during the run.
	* **cfgs folder** - it is highly recommended to save here your config files
	* **dataset folder** - this is where the dataset and the future normalized datasets will be saved.
	* **\<data file\>** - Each data file is a category, so name it as its category (ex. "Entertainment").  The contents of each file is the examples each in a new line.  So totally, every category examples are in one file in separate lines.
	
* To run the program you need to define each classification you want to run.
	   Each classification is defined using a config file (in JSON format).
	   config file example:
	```json
   {
		"train":"your/training/dataset",
		"test":"your/testing/dataset",
		"output_csv":"your/output/directory",
		"nargs":"CHLPR",
		"features":[ngrams_1000_w_tfidf_2_0,],
		"results":"your/results/folder",
		"methods":["mlp","svc"]
	}
	```
	* **train** - The directory of the training dataset. something like this: ```<YOUR_PATH>/<DATASET_NAME>/dataset/training```
	* **test** - The directory of the testing dataset. something like this: ```<YOUR_PATH>/<DATASET_NAME>/dataset/testing```
	* **output_csv** - The directory of the output folder: ```<YOUR_PATH>/<DATASET_NAME>/output```
	* **nargs** - The normalization arguments (ex. ```CHLPR```)
	* **features** - List of the features you want to extract, for ngrams the template is like this:  ```ngrams_<amount>_<w/c>_<tfidf/tf>_<n>_<skip>```
	 (ex.  ```ngrams_1000_w_tfidf_2_0 - 1000``` =  word bigrams without skip with tfidf.)
	* **results** - The directory of the results folder: ```<YOUR_PATH>/<DATASET_NAME>/results```
	* **methods** - List of the classification methods to execute.

				
* To run the program, just use the command line as so:
	```sh 
    python main.py -c "/dataset/cfgs"
    ```
	
	
## Notes
* You can create config files with the config files creator script
* You can create different xlsx file with the write_xlsx_file script using different baseline than the default (the first one)
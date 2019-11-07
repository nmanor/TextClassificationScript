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

Side note - The normalaized datasets will be created in the training\testing folder and will be named training@<nargs> (ex. training@CHL)

## Features
Supported features:
* N-grams (including skip grams)
* common stylistic features (see the full list at [`stylistic_features.py`](stylistic_features.py))

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
	<dataset_name>
		├───cfgs
		├───output
		├───results
		└───dataset
			├───training
			│		├─ <data_file>
			│		│       .
			│		│       .
			│		│       .
			│		└─ <data_file>		
			└───testing (optional, recommended)
					├─ <data_file>
					│       .
					│       .
					│       .
					└─ <data_file>	
    ```
	* **results folder** - the results files (xlsx and data files) will be saved at the end of the run here.
	* **output folder**  - some features data files will be saved here during the run.
	* **cfgs folder** - it is highly recommended to save here your config files
	* **dataset folder** - this is where the dataset and the future normalized datasets will be saved.
	* **\<data file\>** - Each data file is a category, so name it as its category (ex. "Entertainment").  The contents of each file is the examples each in a new line.  So totally, every category examples are in one file in separate lines.
	
	
 * To create that folder structure you can use the script `init_env.py` like so:
    ```sh
    python init_env.py <destination folder>
    ```
    or run the [`init_env.py`](init_env.py) in any python platform.
    
* To run the program you need to define each classification you want to run.
	   Each classification is defined using a config file (in JSON format).
	   config file example:
	```json
   {
		"train":"your/training/dataset",
		"test":"your/testing/dataset",
		"output_csv":"your/output/directory",
		"nargs":"CHLPR",
		"features":["ngrams_1000_w_tfidf_2_0"],
		"results":"your/results/folder",
		"methods":["mlp","svc"],
	        "measure": ["precision_score", "accuracy_score"],
         "stylistic_features": ["pm2", "scc"]
	}
	```
	* **train** - The directory of the training dataset. something like this: ```<YOUR_PATH>/<DATASET_NAME>/dataset/training```
	* **test** - The directory of the testing dataset. something like this: ```<YOUR_PATH>/<DATASET_NAME>/dataset/testing```
	* **output_csv** - The directory of the output folder: ```<YOUR_PATH>/<DATASET_NAME>/output```
	* **nargs** - The normalization arguments (ex. ```CHLPR```)
	* **features** - List of the features you want to extract, for ngrams the template is like this:  ```ngrams_<amount>_<w/c>_<tfidf/tf>_<n>_<skip>```  (ex.  ```ngrams_1000_w_tfidf_2_0 - 1000``` =  word bigrams without skip with tfidf.)
	* **results** - The directory of the results folder: ```<YOUR_PATH>/<DATASET_NAME>/results```
	* **methods** - List of the classification methods to execute.
	* **measure** - List of the measure methods to invoke.
	* **stylistic_features** - List of the stylistic features you want to extract. You can see full list of the features in the `stylistic_features.py` file

#### Running the software:	
* To run the program from the command line, just use the command line as so:
	```sh 
    python main.py -c "/dataset/cfgs"
    ``` 
   You can also run the program in Python platform (like PyCharm or Anaconda) by putting the configuration path in the variable `cfg_dir` in the
   [`main.py`](main.py) file. The running may take a wile. 
* Look for the xlsx files result files, the wordclouds and the pickle files in your `result` folder:
  * **Xlsx files** - the files of the result of the running. Every measure method has its own file.
  * **WordCloud** - if this option is turn on (optional) you should see folders with images of word cloud for 
  each corpus in your `train` and `test` folders. 
  * **Pickle** - files that compress all the result into one line of text. The software using those files for several uses.
  You can see the content of those by using the `pickle` module.
	
	
## Creating config files
You can create alot of config files with the [`config_creator.py`](config_creator.py) script automatically.

## Writing to XLSX file
You can create different xlsx files for each measure using the `new_xlsx_file.py` (the default is to invoke file for ever measure)

## Generate the WordClouds
You can generate the WordClouds to your corpus with `words_clouds.py`. The software will invoke them automatically, but you can stop it in the `main.py` (its may take some time to generate them, so if you want the software to work faster you better turn that off)

## Notifications
You can get a telegram message from a bot when the program finishes or crashes, just add your telegram chat id in the `USERS` list in the `system_config.py` file.

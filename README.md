Text Classification Script
This script is used for normalizing text dataset, extracting features and classifiying the text.


before starting:
    1. make sure that the folder structure is as shown:
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
		**results folder** - the results files (xlsx and data files) will be saved at the end of the run here.
		**output folder**  - some features data files will be saved here during the run.
		**cfgs** - it is highly recommended to save here your config files
		**dataset** - this is where the dataset and the future normalaized datasets will be saved.
		**<data file>** - Each data file is a category, so name it as its category (ex. "Entertainment"). 
					  The contents of each file is the exmaples each in a new line.
					  So totally, every category exmaples are in one file in seperate lines.
	2.To run the program you need to define each classification you want.
	  Each classification is defined using a config file (in JSON format).
	  You can see an example of a config file in *config_exmaple.json*
	
	
	
- You can create config files with the config files creator script
- You can create different xlsx file with the write_xlsx_file script using different baseline than the default (the first one)
before starting:
    1. make sure that the folder structure is as shown:
			*dataset name*
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
		results folder - the results files (xlsx and data files) will be saved at the end of the run here.
		output folder  - some features data files will be saved here during the run.
		cfgs - it is highly recommended to save here your config files
		dataset - this is where the dataset and the future normalaized datasets will be saved.

- You can create config files with the config files creator script
- You can create different xlsx file with the write_xlsx_file script using different baseline than the default (the first one)
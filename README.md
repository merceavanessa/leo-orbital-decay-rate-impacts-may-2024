leo-orbital-decay-rate-impacts-may-2024
==============================

### Description

This repository contains the code used to analyze the storm-induced orbital decay of eleven LEO satellites during May 1–14, 2024, in relation to space weather. Space weather and orbit data were processed via a Snakemake workflow. Data and results are archived on Zenodo and can be used to reproduce our findings by running the two notebooks in `src/notebooks`.
 
### Project Organization

    ├── LICENSE                                    <- License file for the project.
    ├── README.md                                  <- README for this specific sub-project.
    ├── setup.py                                   <- Setup configuration to make the project pip installable.
    ├── requirements.txt                           <- Defines Python dependencies for the analysis environment.
    ├── .gitignore                                 <- Specifies intentionally untracked files to ignore.
    ├── Snakefile                                  <- Snakemake workflow to automate the processing, modeling, and analysis pipelines:
                                                       * Creates and configures the Python/Conda environment.
                                                       * Downloads the solar wind, proxy, orbit determination (POD), and TLE data.
                                                       * Preprocesses each dataset (OMNI, LATIS, POD, TLE) with specific pipelines defined in the configuration files.
                                                       * Merges the preprocessed data into a single dataset for modeling.
                                                       * Runs grid search for hyperparameter tuning and selects the best regression model.
                                                       * Generates predictions and computes feature importances for orbital decay impacts.
    ├── snakemake-config_03-09_16-45.yaml          <- YAML configuration file for dataset paths, pipelines, and model setup.
    ├── Zenodo/                                    <- Main data and results directory.
        ├── data/                                  <- Contains raw, preprocessed, and processed datasets:
            ├── 2024-05-01_2024-05-14/             <- Datasets and configurations for the specified date range.
                ├── raw/                           <- Raw, original data from the different sources.
                ├── preprocessed/                  <- Transformed intermediate data.
                ├── processed/                     <- Final merged dataset ready for modeling and predictions.
                ├── configs/                       <- Configuration files used by the preprocessing pipelines.
                ├── maneuvers/                     <- Maneuver data associated with the dataset.
                ├── CMEs/                          <- Data related to Coronal Mass Ejections (ICME durations, etc.).
                ├── orbgen/                        <- Orbit data for the LEO satellites.
                └── GRID/                          <- Data specifically related to grid configurations for modeling. 
        └── results/                               <- Different results generated from the workflow, from figures, to 
                                                      intermediate data for modeling, to grid results.
    ├── workflow/                                  <- Contains Snakemake workflow reports and a stylesheet
    └── src/                                       <- Source code directory 
        ├── data/                                  <- Methods for downloading datasets (e.g., OMNI, LATIS, POD, TLE).
        ├── preprocessing/                         <- Scripts for cleaning and transforming individual datasets using configuration pipelines.
        ├── merging/                               <- Scripts to merge datasets into a consolidated form for modeling.
        ├── model_selection/                       <- Code to perform model selection (e.g., lasso regression with grid search).
        ├── postprocessing/                        <- Maneuver postprocessing.
        ├── visualization/                         <- Scripts for with plotting methods for data and results.
        ├── experiments/                           <- Scripts for grid search, model selection, and response time estimation.
        ├── configs/                               <- Configuration data classes.
        ├── utils/                                 <- General-purpose helper functions.
        ├── notebooks/                             <- Notebooks generating all figures used in the manuscript

### Reproducibility

To reproduce our results:

1. Clone the repository
2. Install dependencies:
   e.g., pip install -r requirements.txt
3. Download the data and results from Zenodo and place them in `./Zenodo/`
4. Run the notebooks in `src/notebooks`

Intermediate data processing steps can be reproduced by running the Snakemake workflow (e.g., snakemake --cores 1); Snakemake installation instructions are available [here](https://snakemake.readthedocs.io/en/stable/getting_started/installation.html). Intermediate processed files are provided in the Zenodo repository under `/data`. Model selection and response time estimation were run manually using the scripts in `src/experiments`, with the results archived on Zenodo under `/results`. This repository can also be installed as a Python package (pip install -e .).
> Note: The Snakemake workflow was used to organize data processing pipelines. However, model selection and response time estimations were run manually via the scripts in src/experiments. End-to-end automated reproduction via Snakemake has not been tested beyond the data processing steps. Intermediate processed files are available in Zenodo/data, and modeling results are in Zenodo/results.


### Contact

For questions or feedback, please contact:<br>
**Vanessa Mercea** – [vanessa-maria.mercea@unibe.ch](mailto:vanessa-maria.mercea@unibe.ch)<br>
**ORCID** [0000-0001-5252-9393](https://orcid.org/0000-0001-5252-9393)<br>

[![DOI](https://zenodo.org/badge/1185296166.svg)](https://doi.org/10.5281/zenodo.19115188)


<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>.</small></p>


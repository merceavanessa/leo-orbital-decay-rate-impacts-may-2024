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
                                                       Creates and configures the Python/Conda environment.
                                                       Downloads the solar wind, proxy, orbit determination (POD), and TLE data.
                                                       Preprocesses each dataset (OMNI, LATIS, POD, TLE) with specific pipelines defined in the 
                                                       configuration files. Merges the preprocessed data into a single dataset for modeling.
                                                       Runs grid search for hyperparameter tuning and selects the best regression model.
                                                       Generates predictions and computes feature importances for orbital decay impacts.
    ├── snakemake-config_03-09_16-45.yaml          <- YAML configuration file for dataset paths, pipelines, and model setup.
    ├── Zenodo/                                    <- Main data and results directory.
        ├── data/                                  <- Contains raw and processed datasets and configurations
            ├── 2024-05-01_2024-05-14/             <- 
                ├── raw/                           <- [needed] Raw, original data from the different sources.
                ├── processed/                     <- [needed] Final merged dataset ready for modeling and predictions.
                ├── configs/                       <- [needed] Configuration files used by the preprocessing pipelines.
                ├── maneuvers/                     <- [needed] Maneuver data associated with the dataset.
                ├── CMEs/                          <- [needed] Data related to Coronal Mass Ejections (ICME durations, etc.).
                └── GRID/                          <- [needed] Grid configurations and cross validation / model selection outputs. 
        └── results/                               <- [needed] Different results generated from the workflow, from figures, to 
            ├── 2024-05-01_2024-05-14/             <-
                ├── figures/                       <- [output of /src/notebooks] All figures generated for the manuscript, produced by notebooks in
                                                      /src/notebooks 
                ├── disturbance_annotations/       <- [output of /src/notebooks] All figures with annotations of in-orbit disturbance strengths
                ├── cors/                          <- [output of /src/notebooks] Plots of lags vs. correlation and estimated response times by region )
                ├── satellites/<satellite>/
                                       └── data    <- [needed] Input/Output files for model train/test; model and results files, shap values. 
                                                     with_val and no_val sub-folders contain data and results for two kind of 
                                                     data splits: if splitting the train set further into half train, half validation, 
                                                     or using the entire train set to fit the model. This was an analysis done to assess
                                                     the impact of the dataset length on the model performance. Manuscript results use 
                                                     the full train set i.e. no_val folder, once the best model was selected via cross-validation.
                                       └── plots   <- [output of /src/notebooks] Prediction plots per satellite, as in the manuscript
                ├── shifts                         <- [needed] File containing the best response time estimates, and a .csv per satellite with 
                                                      annotated in-orbit disturbance strengths.
                ├── summary                        <- [needed] Files containing aggregated results and feature importances, including the best model parameters,
                                                      performance metrics, and feature importances; a copy of the reference grid configuration;
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

<p><small>Folders marked with `[needed]` are used and must be populated with the corresponding data to run the notebooks and reproduce the results. Folders marked with `[output of /src/notebooks]` are generated by running the two notebooks, and contain all figures and results used in the manuscript.</small></p>

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
**Vanessa Mercea*– [vanessa-maria.mercea@unibe.ch](mailto:vanessa-maria.mercea@unibe.ch)<br>
**ORCID*[0000-0001-5252-9393](https://orcid.org/0000-0001-5252-9393)<br>

[![DOI](https://zenodo.org/badge/1185296166.svg)](https://doi.org/10.5281/zenodo.19115188)


<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>.</small></p>


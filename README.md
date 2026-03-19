leo-orbital-decay-rate-impacts-may-2024
==============================

Quantifying and modeling the storm-induced orbital decay of low Earth orbit satellites during May 2024

### Project Organization

    ├── leo-orbital-decay-rate-impacts-may-2024  <- Analysis and results specific to May 2024 data.
    ├── LICENSE                               <- License file for the project.
    ├── README.md                             <- README for this specific sub-project.
    ├── setup.py                              <- Setup configuration to make the project pip installable.
    ├── requirements.txt                      <- Defines Python dependencies for the analysis environment.
    ├── .gitignore                            <- Specifies intentionally untracked files to ignore.
    ├── Snakefile                             <- Snakemake workflow to automate the processing, modeling, and analysis pipelines:
                                                 * Creates and configures the Python/Conda environment.
                                                 * Downloads the solar wind, proxy, orbit determination (POD), and TLE data.
                                                 * Preprocesses each dataset (OMNI, LATIS, POD, TLE) with specific pipelines defined in the configuration files.
                                                 * Merges the preprocessed data into a single dataset for modeling.
                                                 * Runs grid search for hyperparameter tuning and selects the best regression model.
                                                 * Generates predictions and computes feature importances for orbital decay impacts.
    ├── snakemake-config_03-09_16-45.yaml     <- YAML configuration file for dataset paths, pipelines, and model setup.
    ├── Zenodo/                               <- Main data and results directory.
        ├── data/                             <- Contains raw, preprocessed, and processed datasets:
            ├── 2024-05-01_2024-05-14/        <- Datasets and configurations for the specified date range.
                ├── raw/                      <- Raw, original data from the sources.
                ├── preprocessed/             <- Transformed intermediate data.
                ├── processed/                <- Final merged dataset ready for modeling and predictions.
                ├── configs/                  <- Configuration files used by the preprocessing pipelines.
                ├── maneuvers/                <- Maneuver data associated with the dataset.
                ├── CMEs/                     <- Data related to Coronal Mass Ejections (ICME durations, etc.).
                ├── orbgen/                   <- Orbit generation data for the LEO satellites.
                └── GRID/                     <- Data specifically related to grid configurations and modeling outputs.
        └── results/                          <- Storage for different results generated from the workflow.
    ├── workflow/                             <- Contains Snakemake workflow reports and a stylesheet
    └── src/                                  <- Source code directory 
        ├── data/                             <- Methods for downloading datasets (e.g., OMNI, LATIS, POD, TLE).
        ├── preprocessing/                    <- Scripts for cleaning and transforming individual datasets using configuration pipelines.
        ├── merging/                          <- Scripts to merge datasets into a consolidated form for modeling.
        ├── model_selection/                  <- Code to perform model selection (e.g., lasso regression with grid search).
        ├── postprocessing/                   <- Maneuver postprocessing.
        ├── visualization/                    <- Scripts for with plotting methods for data and results.
        ├── experiments/                      <- Scripts for grid search, model selection, and response time estimation.
        ├── configs/                          <- Configuration data classes.
        ├── utils/                            <- General-purpose helper functions.
        ├── notebooks/                        <- Notebooks generating all figures used in the manuscript
### Notes

MIT License

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>.</small></p>
#########################################################################################################
# MAIN
#########################################################################################################
configfile: "snakemake-config_03-09_16-45.yaml"
configfile: "snakemake-credentials.yaml"


#########################################################################################################
# SETUP
#########################################################################################################

rule create_environment:
    """
    Create a new conda environment based on the provided configuration.
    """
    params:
        python_interpreter = config["env"]["python_interpreter"],
        python_version = config["env"]["python_version"],
        has_conda = config["env"]["has_conda"],
        conda_env = config["env"]["conda_env"],
        conda_env_path = config["env"]["conda_env_path"],
        raw_data_path = config["data"]["base_paths"]["raw"],
        preprocessed_data_path = config["data"]["base_paths"]["preprocessed"],
        processed_data_path = config["data"]["base_paths"]["processed"],
        data_config_path = config["data"]["base_paths"]["configs"]
    output:
        "logs/create_environment.log"
    shell:
        """
        mkdir -p {params.raw_data_path}
        mkdir -p {params.preprocessed_data_path}
        mkdir -p {params.data_config_path}

        if {params.has_conda}
        then
            echo ">>> Detected conda, creating conda environment."
            conda create -p {params.conda_env_path}{params.conda_env} python=3.9 
            touch {output}
            echo ">>> New conda env created. Activate with:\nsource activate {params.conda_env_path}{params.conda_env}" > {output}
        else
            touch {output}
            echo ">>> No conda detected. Please install conda to manage your environment" > {output}
        fi
        """


rule setup_requirements:
    """
    Install the required packages in the environment.
    """
    input:
        "logs/create_environment.log",
    params:
        python_interpreter = config["env"]["python_interpreter"],
    output:
        "logs/setup_requirements.log"
    shell:
        """
        {params.python_interpreter} -m pip install -U pip setuptools wheel
       	{params.python_interpreter} -m pip install -r requirements.txt
        touch {output}
        echo ">>> Requirements up to date." > {output}
        """

rule check_environment:
    """
    Check if the environment is set up correctly.
    """
    input:
        "logs/setup_requirements.log",
    params:
        python_interpreter = config["env"]["python_interpreter"],
        conda_env = config["env"]["conda_env"],
        conda_env_path = config["env"]["conda_env_path"]
    output:
        "logs/check_environment.log"
    shell:
        """
        echo "checking env"
        {params.python_interpreter} test_environment.py
        touch {output}
        source activate {params.conda_env_path}{params.conda_env}
        echo ">>> Environment up to date."
        echo ">>> Environment up to date." > {output}
        """

#########################################################################################################
# Download DATA
#########################################################################################################
rule download_omni_data:
    """
    Download solar wind data for specified dates
    """
    input:
    params:
        python_interpreter = config["env"]["python_interpreter"],
        start_date = config["data"]["download"]["start_date"],
        end_date = config["data"]["download"]["end_date"],
        folder_path = config["data"]["base_paths"]["raw"],
        solar_wind_ds = config["data"]["datasets"]["omni"],
        proxies_ds = config["data"]["datasets"]["latis"]
    output:
        solar_wind_raw = config["data"]["base_paths"]["raw"] + \
            "/"+config["data"]["datasets"]["omni"] + '.csv'
    shell:
        """
        {params.python_interpreter} src/data/download_datasets.py \
        --folder_path {params.folder_path} \
        --data_source OMNI \
        --dataset {params.solar_wind_ds} \
        --start_date {params.start_date} \
        --end_date {params.end_date}
        """

rule download_latis_data:
    """
    Download latis data for specified dates
    """
    input:
    params:
        python_interpreter = config["env"]["python_interpreter"],
        start_date = config["data"]["download"]["start_date"],
        end_date = config["data"]["download"]["end_date"],
        folder_path = config["data"]["base_paths"]["raw"],
        proxies_ds = config["data"]["datasets"]["latis"]
    output:
        proxies_raw = config["data"]["base_paths"]["raw"] + \
            "/"+config["data"]["datasets"]["latis"] + '.csv'
    shell:
        """
        {params.python_interpreter} src/data/download_datasets.py \
        --folder_path {params.folder_path} \
        --data_source LATIS \
        --dataset {params.proxies_ds} \
        --start_date {params.start_date} \
        --end_date {params.end_date}
        """

rule add_pod_data:
    """
    Download POD data for specified dates from 
    """
    params:
       pod_path_orig = config["data"]["base_paths"]["original_pod"]
    output:
        pod_raw = directory(config["data"]["base_paths"]["raw"] + \
            "/"+config["data"]["datasets"]["pod"])
    shell:
        """
        mkdir -p {output.pod_raw}
        cp {params.pod_path_orig}/*.LST {output.pod_raw}/ 
        """

rule download_tle_data:
    """
    Download TLE data for specified dates from 
    """
    input:
    params:
        python_interpreter = config["env"]["python_interpreter"],
        start_date = config["data"]["download"]["start_date"],
        end_date = config["data"]["download"]["end_date"],
        folder_path = config["data"]["base_paths"]["raw"],
        conf_path = config["data"]["base_paths"]["configs"],
        tle_ds = config["data"]["datasets"]["tle"],
        spacetrack_user = config["credentials"]["spacetrack"]["username"],
        spacetrack_password = config["credentials"]["spacetrack"]["password"]
    output:
        tle_raw = directory(config["data"]["base_paths"]["raw"] + \
            "/"+config["data"]["datasets"]["tle"])
    shell:
        """
        mkdir -p {output.tle_raw}
        {params.python_interpreter} src/data/download_datasets.py \
        --folder_path {params.folder_path} \
        --conf_path {params.conf_path} \
        --data_source TLE \
        --dataset {params.tle_ds} \
        --start_date {params.start_date} \
        --end_date {params.end_date} \
        --username {params.spacetrack_user} \
        --password {params.spacetrack_password}
        """

#########################################################################################################
# PREPROCESS DATA
#########################################################################################################

# Preprocess Solar Wind Data
rule preprocess_omni:
    """
    Preprocess the OMNI data.
    """
    input:
        input_file_path = config["data"]["base_paths"]["raw"] + \
            "/"+config["data"]["datasets"]["omni"] + '.csv',
    params:
        dataset = config["data"]["datasets"]["omni"],
        configuration = config["preprocessing_pipelines"][config["data"]
                                                          ["datasets"]["omni"]],
        python_interpreter = config["env"]["python_interpreter"]
    output:
        preprocessed_file_path = config["data"]["base_paths"]["preprocessed"] + \
            "/"+config["data"]["datasets"]["omni"] + '.csv'
    shell:
        """
        {params.python_interpreter} src/preprocessing/preprocess_dataset.py \
        -input_file_path {input.input_file_path} \
        -dataset {params.dataset} \
        -pipeline_configuration "{params.configuration}" \
        -output_file_path {output.preprocessed_file_path}
        """

# Preprocess Geomagnetic Indices and Proxy Data
rule preprocess_latis:
    """
    Preprocess the geomagnetic indices and Solar proxy data.
    """
    input:
        input_file_path = config["data"]["base_paths"]["raw"] + \
            "/"+config["data"]["datasets"]["latis"] + '.csv',
    params:
        dataset = config["data"]["datasets"]["latis"],
        configuration = config["preprocessing_pipelines"][config["data"]
                                                          ["datasets"]["latis"]],
        python_interpreter = config["env"]["python_interpreter"]
    output:
        preprocessed_file_path = config["data"]["base_paths"]["preprocessed"] + \
            "/"+config["data"]["datasets"]["latis"] + '.csv'
    shell:
        """
        {params.python_interpreter} src/preprocessing/preprocess_dataset.py \
        -input_file_path {input.input_file_path} \
        -dataset {params.dataset} \
        -pipeline_configuration "{params.configuration}" \
        -output_file_path {output.preprocessed_file_path}
        """

# Preprocess TLE Data
rule preprocess_tle:
    """
    Preprocess TLE Data
    """
    input:
        input_file_path = config["data"]["base_paths"]["raw"] + \
            "/"+config["data"]["datasets"]["tle"]
    params:
        dataset = config["data"]["datasets"]["tle"],
        configuration = config["preprocessing_pipelines"][config["data"]
                                                          ["datasets"]["tle"]],
        python_interpreter = config["env"]["python_interpreter"]
    output:
        preprocessed_file_path = directory(config["data"]["base_paths"]["preprocessed"] + \
            "/"+config["data"]["datasets"]["tle"])
    shell:
        """
        mkdir -p {output.preprocessed_file_path}
        {params.python_interpreter} src/preprocessing/preprocess_dataset.py \
        -input_file_path {input.input_file_path} \
        -dataset {params.dataset} \
        -pipeline_configuration "{params.configuration}" \
        -output_file_path {output.preprocessed_file_path}
        """

# Preprocess POD Data
rule preprocess_pod:
    """
    Preprocess POD Data
    """
    input:
        input_file_path = config["data"]["base_paths"]["raw"] + \
            "/"+config["data"]["datasets"]["pod"]
    params:
        dataset = config["data"]["datasets"]["pod"],
        configuration = config["preprocessing_pipelines"][config["data"]
                                                          ["datasets"]["pod"]],
        python_interpreter = config["env"]["python_interpreter"]
    output:
        preprocessed_file_path = directory(config["data"]["base_paths"]["preprocessed"] + \
            "/"+config["data"]["datasets"]["pod"])
    shell:
        """
        mkdir -p {output.preprocessed_file_path}
        {params.python_interpreter} src/preprocessing/preprocess_dataset.py \
        -input_file_path {input.input_file_path} \
        -dataset {params.dataset} \
        -pipeline_configuration "{params.configuration}" \
        -output_file_path {output.preprocessed_file_path}
        """

# we should run this for each individual satellite by updating the config file in the data folder
# Combine Data
rule combine_data_products:
    """
    Combine the preprocessed solar wind, geomagnetic indices, proxy data and satellite data.
    """
    input:
        solar_wind_preprocessed = config["data"]["base_paths"]["preprocessed"] + \
            "/"+config["data"]["datasets"]["omni"] + '.csv',
        proxies_preprocessed = config["data"]["base_paths"]["preprocessed"] + \
            "/"+config["data"]["datasets"]["latis"] + '.csv',
        sat_preprocessed = config["data"]["base_paths"]["preprocessed"] + \
            "/"+config["data"]["datasets"]["pod"],
        tle_preprocessed = config["data"]["base_paths"]["preprocessed"] + \
            "/"+config["data"]["datasets"]["tle"]
    params:
        python_interpreter = config["env"]["python_interpreter"],
        start_date = config["data"]["download"]["start_date"],
        end_date = config["data"]["download"]["end_date"],
    output:
        combined_data_path = directory(config["data"]["base_paths"]["processed"])
    shell:
        """
        mkdir -p {output.combined_data_path}
        {params.python_interpreter} src/merging/merge_datasets.py \
        -input_file_paths {input.solar_wind_preprocessed} \
        -input_file_paths {input.proxies_preprocessed} \
        -satellite_base_path {input.sat_preprocessed} \
        -tle_base_path {input.tle_preprocessed} \
        -output_base_path {output.combined_data_path}
        """


#########################################################################################################
# MODEL SELECTION AND PREDICTION
#########################################################################################################

# Select Best Model and Predict
rule select_best_and_predict:
    """
    Select best model and predict scores and importances.
    """
    input:
        input_base_path = config["data"]["base_paths"]["base"],
        input_sub_path = config["data"]["base_paths"]["processed"],
        current_grid_filepath = config["data"]["base_paths"]["base"] + config["data"]["grid_paths"]["output"] + config["data"]["grid_paths"]["current_grid_filename"]
    params:
        python_interpreter = config["env"]["python_interpreter"]
    shell:
        """
        {params.python_interpreter} src/model_selection/select_best_and_predict.py \
        -input_base_path {input.input_base_path} \
        -input_sub_path {input.input_sub_path} \
        -current_grid_filepath {input.current_grid_filepath} 
        """

# Run Grid Search
rule run_grid_search:
    """
    Run Lasso regression experiment with grid search.
    """
    input:
        input_base_path = config["data"]["base_paths"]["base"],
        combined_data_path = config["data"]["base_paths"]["processed"]
    params:
        python_interpreter = config["env"]["python_interpreter"]
    output:
        current_grid_filepath = config["data"]["base_paths"]["base"] + config["data"]["grid_paths"]["output"] + config["data"]["grid_paths"]["current_grid_filename"]
    shell:
        """
        {params.python_interpreter} src/experiments/run_lasso_experiment.py \
        -input_base_path {input.input_base_path} \
        -input_sub_path {input.combined_data_path} \
        -current_grid_filepath {output.current_grid_filepath}
        """


#########################################################################################################
# END RULE TO GENERATE A PIPELINE REPORT
#########################################################################################################

rule analysis:
    """
    Run the analysis pipeline.
    """
    input:
        combined_data_path = config["data"]["base_paths"]["processed"]
    params:
        report_name= config["env"]["report_name"],
        reports_path = config["env"]["reports_path"],
        report_file = config["env"]["reports_path"] + "/" + config["env"]["report_name"],
    shell:
        """
        mkdir -p {params.reports_path}
        # snakemake --report > {params.report_file} --cores all
        """
# Condition-Treatment Relation Extraction on Disease-related Social Media Data

## Introduction

This project is a collaborative project between [Emory NLP](https://www.emorynlp.org) and [Real Life Science](https://rlsciences.com), which develops annotation guidelines and address automatic extraction of medical entities (e.g., ‘Patient Condition’ and ‘Procedure’) and their relations in disease-related social media data.
The paper will be published in [Proceedings of the EMNLP Workshop on Health Text Mining and Information Analysis](https://louhi2022.fbk.eu/)/2022.

## Model

The model employed in this project is adapted from [this paper](https://aclanthology.org/2022.naacl-main.395/) by @lxucs and @jdchoi77. Since our annotation does not include coreference, we skip the coreference evaluation part in the original model.

### Predict

To use the trained model, configure the experiment settings in [experiments_new.conf](RLS_model/extraction/gen-extraction/experiments_new.conf) file and use the following commands to predict:

`python run_med.py [config_name] [model_suffix] [gpu_id]`

The results will be saved under 'extraction/[config_name]' folder.

### Analysis

To view and analyze the model results, configure the settings in [postprocess.conf](RLS_model/postprocess/postprocess.conf). And run the [resultviewer.py](RLS_model/postprocess/resultviewer.py) file.
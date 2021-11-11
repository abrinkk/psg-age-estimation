<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Thanks again! Now go create something AMAZING! :D
***
***
***
*** To avoid retyping too much info. Do a search and replace for the following:
*** github_username, repo_name, twitter_handle, email, project_title, project_description
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->

<!-- PROJECT LOGO -->
<br />
<p align="center">

  <h3 align="center">Age Estimation from Sleep using Deep Learning Predicts Life Expectancy</h3>

  <p align="center">
    Deep learning framework for age estimation in nocturnal polysomnography recordings. 
    <br />
    <a href="https://github.com/abrinkk/psg-age-estimation"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/abrinkk/psg-age-estimation">View Demo</a>
    ·
    <a href="https://github.com/abrinkk/psg-age-estimation/issues">Report Bug</a>
    ·
    <a href="https://github.comabrinkk/psg-age-estimation/issues">Request Feature</a>
  </p>
</p>


<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
    <li><a href="#about-the-project">About The Project</a></li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#age-estimation-inference-in-new-pSGs">Age Estimation (Inference) in New PSGs</a></li>
        <li><a href="#training-age-estimation-models-from-scratch">Training Age Estimation Models from Scratch</a></li>
      </ul>
    </li>
    <li>
	  <a href="#usage">Usage</a>
	  <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
	  </ul>
	</li>
    <li><a href="#license">License</a></li>
    <li><a href="#academic-citation">Academic Citation</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

This project is an analysis tool for polysomnography (PSG) studies that estimates age based on electrophysiological waveforms. The age estimation error (estimated age - chronological age) is a health measure, which was found to be associated with higher risk for mortality. The software inputs PSG data in the European Data Format (.edf).

<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple steps.

### Prerequisites

Here is a list of steps to get prerequisites for the repository. 
* Get python v. 3.7.
* Get python packages from requirement.txt file. These can be installed with pip:
  ```sh
  pip install -r requirements.txt
  ```

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/abrinkk/psg-age-estimation.git
   ```

<!-- USAGE EXAMPLES -->
## Usage

### Age Estimation (Inference) in New PSGs

To estimate age in .edf files, a single script perform all steps: 1) preprocessing, 2) estimation with all four age estimation models, and 3) collects all results in a single .csv file:

   ```sh
   python age_estimation_all.py --input_folder FOLDER_WITH_EDF_FILES --output_folder FOLDER_WITH_PREPROCESSED_FILES
   ```

### Training Age Estimation Models from Scratch

The project includes further scripts in matlab and R for outcome analysis etc. Therefore, additional prerequisites are necessary:

* Matlab v. 2019b
* R v. 4.0.4

These are the steps required to follow the method outlined in our paper:

1. Split data in a training, validation, and test set with uniform age distributions for the training and validation set. 
   ```sh
   matlab data_age.m
   ```
2. Preprocess all data. This step select and references the correct channels, filters, and resamples the signals. This step is carried out for each cohort separately.
   ```sh
   python psg2h5.py --input_folder FOLDER_WITH_EDF_FILES --output_folder FOLDER_WITH_PREPROCESSED_FILES --cohort COHORT_NAME
   ```
3. Run the training script, which trains a model with a given configuration.
   ```sh
   python age_main.py --pre_hyperparam LR L2 DROPOUT_RATE DROPOUT_CHANNELS LOSS_FUNCTION BATCH_SIZE SEQUENCE_LENGTH SLEEP_DATA ONLY_EEG --bo 
   ```
   Here the pre_hyperparam flags indicate hyperparameters for phase (1) of the model. LR: learning rate, L2: L2 weight decay, DROPOUT_RATE: dropout rate for the last layer, DROPOUT_CHANNELS: dropout rate for each signal, LOSS_FUNCTION: choice of loss function, BATCH_SIZE: batch_size, SEQUENCE_LENGTH: length of epochs (set to 5 minutes), SLEEP_DATA: to ignore all data in wakefulness, ONLY_EEG: flag to choose channel configurations. The --bo flag indicates whether to use Bayesian optimization for the hyperparameters in phase (2) of the model.
4. (Optional) - Interpret model decisions. 
   ```sh
   python model_interpretation.py --m_run MODEL_NAME --atr_method RELEVANCE_ATTRIBUTION_METHOD
   ```
5. (Optional) - Evaluate mortality risk based on age estimation errors. 
   ```sh
   matlab sleep_age_statistics.m
   ```

NOTE: Matlab script need to be changed such that paths are correctly set.

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.

<!-- CITATION -->
## Academic Citation

Please use the following citation when referencing this software:

TBD.


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/abrinkk/repo.svg?style=for-the-badge
[contributors-url]: https://github.com/abrinkk/psg-age-estimation/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/abrinkk/repo.svg?style=for-the-badge
[forks-url]: https://github.com/abrinkk/psg-age-estimation/network/members
[stars-shield]: https://img.shields.io/github/stars/abrinkk/repo.svg?style=for-the-badge
[stars-url]: https://github.com/abrinkk/psg-age-estimation/stargazers
[issues-shield]: https://img.shields.io/github/issues/abrinkk/repo.svg?style=for-the-badge
[issues-url]: https://github.com/abrinkk/psg-age-estimation/issues
[license-shield]: https://img.shields.io/github/license/abrinkk/repo.svg?style=for-the-badge
[license-url]: https://github.com/abrinkk/psg-age-estimation/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/andreas-bk/
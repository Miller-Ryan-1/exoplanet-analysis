## IDENTIFYING STELLAR CANDIDATES WITH EARTHLIKE SATELLITES

<hr style="border-top: 10px groove blueviolet; margin-top: 1px; margin-bottom: 1px"></hr>

### Project Summary
<hr style="border-top: 10px groove blueviolet; margin-top: 1px; margin-bottom: 1px"></hr>

#### Project Objectives
> - Exoplanets are planets outside of our solar system and we're entering a new era of exoplanet discovery with the James Webb Space Telescope (JWST) coming online.  Expanding on the Kepler telescope and a number of other telescopes/missions geared towards exoplanet discovery, the JSWT will allow humanity to glimpse further into the unverse than ever before; but, it can also get amazing detail of things much closer.  In fact, under certain conditions, it can 'see' an extrasolar planet's atmosphere, possibly detecting telltale signs of life. 
> - With so many stars in our galaxy along (~100 Billion) and assets limited to one telescope, we seek to identify stellar candidates that have 'Earthlike' planets to focus our attention on.
> - Beyond identifying drivers, we will build a classification model to attempt to classify current and future stellar targets based on their potential to have an Earthlike planet.

#### Goals
> - Identify Key Stellar Characteristics of Stars with Earthlike planets orbiting them.
> - Construct a ML classification model that predicts whether a star has an Earthlike exoplanet based on key stellar features.  This will help focus future telescope observations on those stars with a high probability to have Earthlike planets, which will maximize our chances to detect signs of life.
> - Document my process well enough to be presented or read like a report.

#### Audience
> - Planetary Scientists
> - CodeUp Students!

#### Project Deliverables
> - A final report notebook (exoplanet_study_notebook.ipynb)
> - All necessary modules to make my project reproducible

#### Data Dictionary
- Note: Includes only those features selected for full EDA and Modeling:

|Target|Datatype|Definition|
|:-------|:--------|:----------|
| y | 2820 non-null: object | Earthlike or Not-Earthlike gravity measurement, based on planet's radius |
| y_encoded | 2820 non-null: int64 | *Alternate target encoded to 1 and 0* |

|Feature|Datatype|Definition|
|:-------|:--------|:----------|
| num_planets_in_sys       | 2820 non-null: int64 | number of planets in system |
| orbital_period        | 2820 non-null: float64 | how long it takes a planet to orbit it's star, in earth days |
| metallicity       | 2820 non-null: float64 | higher element (non-hydrogen or helium) content of star |
| star_age        | 2820 non-null: float64 | age of star, in billions of years |
| star_density        | 2820 non-null: float64 | average density of star, in g/cm^3 |
| multistar        | 2820 non-null: int64 | does the planet orbit more than one star? |
| discovery_order        | 2820 non-null: int64 | order the planet was discovered in relation to the other planets found at its star |
| lt_cluster        | 2820 non-null: int32 | cluster of the star on the main sequence diagram |


#### Initial Hypotheses

> - **Hypothesis 1 -**
> - There are stellar features that increase the likelihood of a star having an earthlike planet (or planets).

> - **Hypothesis 2 -** 
> - There is an optimal clustering feature that combines a number of these stellar characteristics to improve our ability to predict whether that star has an Earthlike planet(s) or not

<hr style="border-top: 10px groove blueviolet; margin-top: 1px; margin-bottom: 1px"></hr>

### Executive Summary - Conclusions & Next Steps
<hr style="border-top: 10px groove blueviolet; margin-top: 1px; margin-bottom: 1px"></hr>

> - Question: Exoplanets are planets outside of our solar system and we're entering a new era of exoplanet discovery with the James Webb Space Telescope (JWST) coming online.  Expanding on the Kepler telescope and a number of other telescopes/missions geared towards exoplanet discovery, the JSWT will allow humanity to glimpse further into the unverse than ever before; but, it can also get amazing detail of things much closer.  In fact, under certain conditions, it can 'see' an extrasolar planet's atmosphere, possibly detecting telltale signs of life.  As such, with so many stars in our galaxy along (~100 Billion) and assets limited to one telescope, can we identify stellar candidates that have 'Earthlike' planets to focus our attention on?
> - Actions: Using exoplanet telescope data consolidated by CalTech, I attempted to filter and identify key characteristics of stars with earthlike planets - with the goal of identifying planets which would most likely harbor detectable life.  As such, I did a detailed analysis of the best data I could put together on every planet so far discovered to see if there were any clear drivers of having earthlike planets.  Modeling was attempted to see if an ML Classification Model could be used to classify new and existing stellar candidates.
> - Conclusions:  Even with extensive analysis, there was not much success in identifying clear characteristics of stars which harbor Earthlike planets.  Modeling also failed to improve on baseline.  Overall this is not surprising as this is a very new field stretching cutting edge instruments to their furtherst capabilities.
> - Recommendations: Continue to cluster stellar characteristics in an attempt to find one statistically signifigant as an indicator of Earthlike planets.  Also, an imputing algorithm can be used to capture a number of planets that had nulls that could not be filled simply by combing and finding the mean of all availabel data for that planet.  Lastly, focus early JWST studies on building a dataset which allows for modeling the likelihood of a star having an earthlike planet, as this will make it a better tool in the long run by continually improving its targeting.

<hr style="border-top: 10px groove blueviolet; margin-top: 1px; margin-bottom: 1px"></hr>

### Pipeline Stages Breakdown

<hr style="border-top: 10px groove blueviolet; margin-top: 1px; margin-bottom: 1px"></hr>

##### Plan
- [x] Create README.md with data dictionary, project objectives and goals, come up with initial hypotheses.
- [x] Acquire data from the Cal Tech Exoplanet Database and create a function to automate this process. Save the function in an acquire.py file to import into the final Study Notebook. Also, cache locally, since that is always good practice.
- [x] Clean and prepare data for the first iteration through the pipeline: MVP preparation. Create a function to automate the process, store the function in a wrangle.py module, and prepare data in final Study Notebook by importing and using the function.
- [x] Investigate data, formulate hypotheses, visualize analsyis and run statistical tests when necessary (ensuring signifigance and hypotheses are created and usability assumptions met).  Document findings and takeaways.
- [x] Use clustering and cluster analysis to better understand the data and/or create new features.
- [x] Establish a baseline modeling accuracy.
- [x] Train multiple different classicitcation models, to include hyperparameter tuning.
- [x] Evaluate models on train and validate datasets.
- [x] Choose the model with that performs the best and evaluate that single model on the test dataset.
- [x] Document conclusions, takeaways, and next steps in the final Study Notebook.

___

##### Plan -> Acquire
> - Store functions that are needed to acquire data from the database server; make sure the acquire.py module contains the necessary imports for anyone with database access to run the code.
> - The final function will return a pandas DataFrame.
> - Import the acquire function from the acquire.py module and use it to acquire the data in the final Study Notebook.
> - Complete some initial data summarization (`.info()`, `.describe()`, `.value_counts()`, etc.).
> - Plot distributions of individual variables.
___

##### Plan -> Acquire -> Prepare/Wrange
> - Store functions needed to wrangle the data; make sure the module contains the necessary imports to run the code. The final functions (wrangle.py) should do the following:
    - Split the data into train/validate/test.
    - Handle any missing values.
    - Handle erroneous data and/or outliers that need addressing.
    - Encode variables as needed.
    - Create any new features, if made for this project.
> - Import the prepare functions from the wrangle.py module and use it to prepare the data in the final Study Notebook.
___

##### Plan -> Acquire -> Prepare -> Explore
> - Answer key questions, my hypotheses, and figure out the features that can be used in a classification model to best predict the target variable, has Earthlike planet (or not). 
> - Run at least 2 statistical tests in data exploration. Document my hypotheses, set an alpha before running the tests, and document the findings well.
> - Create visualizations and run statistical tests that work toward discovering variable relationships (independent with independent and independent with dependent). The goal is to identify features that are related to Earthlike planets (the target), identify any data integrity issues, and understand 'how the data works'. If there appears to be some sort of interaction or correlation, assume there is no causal relationship and brainstorm (and document) ideas on reasons there could be correlation.
> - Summarize my conclusions, provide clear answers to my specific questions, and summarize any takeaways/action plan from the work above.
___

##### Plan -> Acquire -> Prepare -> Explore -> Model
> - Feature Selection and Encoding: Are there any variables that seem to provide limited to no additional information? If so, remove them.  Also encode any non-numerical features of signifigance.
> - Establish a baseline accuracy to determine if having a model is better than no model and train and compare at least 4 different models.
> - Train (fit, transform, evaluate) multiple models, varying the algorithm and/or hyperparameters you use.
> - Compare evaluation metrics across all the models you train and select the ones you want to evaluate using your validate dataframe.  In this case we used Precision (Positive Predictive Value).
> - Based on the evaluation of the models using the train and validate datasets, choose the best model to try with the test data, once.
> - Test the final model on the out-of-sample data (the testing dataset), summarize the performance, interpret and document the results.
___

##### Plan -> Acquire -> Prepare -> Explore -> Model -> Deliver
> - Summarize my findings at the beginning like I would for an Executive Summary.
> - Walk the management team through the analysis I did to answer my questions and that lead to my findings. (Visualize relationships and Document takeaways.) 
> - Clearly call out the questions and answers I am analyzing as well as offer insights and recommendations based on my findings.

<hr style="border-top: 10px groove blueviolet; margin-top: 1px; margin-bottom: 1px"></hr>

### Reproduce My Project

<hr style="border-top: 10px groove blueviolet; margin-top: 1px; margin-bottom: 1px"></hr>

You will need all the necessary files listed below to run my final project notebook. 
- [x] Read this README.md
- [ ] Download the aquire.py, wrangle.py, explore.py and model_comparator.py files into your working directory.
- [ ] For more details on the analysis, in particular clustering charts, download the subsection workbooks: exoplanet_acquire_scrapbook.ipynb, exoplanet_wrangle_scrapbook.ipynb, exoplanet_clusering_and_EDA_scrapbook.ipynb, and exoplanet_modeling_scrapbook.ipynb.
- [ ] Run the exoplanet_study_notebook.ipynb notebook

##### Credit to Faith Kane (https://github.com/faithkane3) for the format of this README.md file.
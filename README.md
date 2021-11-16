# Splenomegaly-Detection

This is an ensemble model for training on data comprised of electronic radiologist notes to predict the presence of metastatic cancers. Model written in Python 3.

**NOTE: the model presented here was created, trained, tested, and validated on data that is not available for public use. The code will not run as is. This code has been cleaned of any PHI and can be viewed as a means of inspecting the architecture of the model. Running the model on the actual data is not possible.**

Training data is accepted in csv form. The csv is required to have a column of notes for a specified location (i.e. "spleen") for the x data and an associated column of metastases (i.e. "spleen_metastases") comprised of either Yes/No or Yes/Indeterminate/No values for the y data (0/1/2 is also accepted). An "impression" column is also required. These columns can be have any name, so long as those names are identified in the "Setup" portion of the code. So long as the data has these three columns the model should work. Any other columns will not disrupt prediction or be used at all.

Libraries you'll need:

- pandas
- keras
- numpy
- matplotlib
- sklearn
- seaborn

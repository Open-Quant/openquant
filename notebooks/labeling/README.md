# Research Notebooks
The following research notebooks explore the topic of labeling observations and implement various models for labeling to improve
model performance.  Meta-labeling is introduced as a method for applying a secondary model (Random Forest) on top of a primary model
to make decisions in terms of the size of the bet.

## Trend Following Notebook
Fit a primary model based on trend following and then adds meta-labeling to improve the model and strategy performance metrics. 
The out-of-sample results are compared against the stand-alone primary model.

## Bollinger-Band Notebook
Fits a primary model based on mean reversion and then adds meta-labeling to improve the model and strategy performance metrics.
The out-of-sample results are compared against the stand-alone primary model.
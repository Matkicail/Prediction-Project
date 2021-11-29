# Prediction-Project
Research Project Code

Code base for research project into stock market algorithms and their performance on various exchanges.

The CORN-based models implemented based on their research papers. 
CORN-K provides the crucial inspiration as the algorithm uses pattern-matching to produce strong returns.
CORN-K can experience some issues if the correlation coefficients are extremely high (very close to one) hence we introduce a new model AMA-K to assist if this is the case.
We understand that it may be preferable to use a uniform portfolio in cases where the market is very unique as it may represent a rapid change that has not occured in the past.

The AMA-K model is CORN-based and CORN inspired. AMA-K tries to integrate alternative methods for similarity through the use of Online-KMeans clustering.
Furthermore, it uses multiple agents over different time spans.

AMA-K needs more back testing and analysis to further validate its results. The method provided here was part of a research project.

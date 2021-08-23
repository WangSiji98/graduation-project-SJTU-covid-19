# covid19

##	fit_SEIR.py

code for fitting SEIR model

output:

* fitted epidemic curve
* fitted model parameters
* fitted stage division

##	SEIR_analyze.py

code for correlation analysis after fitting

input:
* model parameters and stage division gaven by fit_SEIR.py 

paramater:
* focus_type，Relative, Policy attention / total attention of epidemic related news; GlobalRelative, Policy attention / total attention of all news
* fixed_t0：-
* residual：True, the correlation between the difference of beta before and after the policy takes effect and the degree of attention is analyzed; False, analyze the correlation between beta value and attention after taking effect

output
* Excel file of correlation and M relation
* PDF of relevance and entry relationship

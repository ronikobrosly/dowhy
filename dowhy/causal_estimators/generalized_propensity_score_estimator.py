import itertools
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from scipy.stats import gamma, norm
import statsmodels.api as sm
from statsmodels.gam.generalized_additive_model import GLMGam
from statsmodels.gam.smooth_basis import BSplines

from dowhy.causal_estimator import CausalEstimator
from dowhy.causal_estimators.regression_estimator import RegressionEstimator
from dowhy.causal_identifier import IdentifiedEstimand


class GeneralizedPropensityScoreEstimator(RegressionEstimator):
    """
    Estimate the effect of a continuous treatment variable through
    calculation of the generalized propensity score.
    """
    def __init__(
        self,
        identified_estimand: IdentifiedEstimand,
        test_significance: Union[bool, str] = False,
        evaluate_effect_strength: bool = False,
        confidence_intervals: bool = False,
        num_null_simulations: int = CausalEstimator.DEFAULT_NUMBER_OF_SIMULATIONS_STAT_TEST,
        num_simulations: int = CausalEstimator.DEFAULT_NUMBER_OF_SIMULATIONS_CI,
        sample_size_fraction: int = CausalEstimator.DEFAULT_SAMPLE_SIZE_FRACTION,
        confidence_level: float = CausalEstimator.DEFAULT_CONFIDENCE_LEVEL,
        need_conditional_estimates: Union[bool, str] = "auto",
        num_quantiles_to_discretize_cont_cols: int = CausalEstimator.NUM_QUANTILES_TO_DISCRETIZE_CONT_COLS,
        **kwargs,
    ):
        """
        :param identified_estimand: probability expression
            representing the target identified estimand to estimate.
        :param test_significance: Binary flag or a string indicating whether to test significance and by which method. All estimators support test_significance="bootstrap" that estimates a p-value for the obtained estimate using the bootstrap method. Individual estimators can override this to support custom testing methods. The bootstrap method supports an optional parameter, num_null_simulations. If False, no testing is done. If True, significance of the estimate is tested using the custom method if available, otherwise by bootstrap.
        :param evaluate_effect_strength: (Experimental) whether to evaluate the strength of effect
        :param confidence_intervals: Binary flag or a string indicating whether the confidence intervals should be computed and which method should be used. All methods support estimation of confidence intervals using the bootstrap method by using the parameter confidence_intervals="bootstrap". The bootstrap method takes in two arguments (num_simulations and sample_size_fraction) that can be optionally specified in the params dictionary. Estimators may also override this to implement their own confidence interval method. If this parameter is False, no confidence intervals are computed. If True, confidence intervals are computed by the estimator's specific method if available, otherwise through bootstrap
        :param num_null_simulations: The number of simulations for testing the
            statistical significance of the estimator
        :param num_simulations: The number of simulations for finding the
            confidence interval (and/or standard error) for a estimate
        :param sample_size_fraction: The size of the sample for the bootstrap
            estimator
        :param confidence_level: The confidence level of the confidence
            interval estimate
        :param need_conditional_estimates: Boolean flag indicating whether
            conditional estimates should be computed. Defaults to True if
            there are effect modifiers in the graph
        :param num_quantiles_to_discretize_cont_cols: The number of quantiles
            into which a numeric effect modifier is split, to enable
            estimation of conditional treatment effect over it.
        :param kwargs: (optional) Additional estimator-specific parameters
        """
        super().__init__(
            identified_estimand=identified_estimand,
            test_significance=test_significance,
            evaluate_effect_strength=evaluate_effect_strength,
            confidence_intervals=confidence_intervals,
            num_null_simulations=num_null_simulations,
            num_simulations=num_simulations,
            sample_size_fraction=sample_size_fraction,
            confidence_level=confidence_level,
            need_conditional_estimates=need_conditional_estimates,
            num_quantiles_to_discretize_cont_cols=num_quantiles_to_discretize_cont_cols,
            **kwargs,
        )
        self.logger.info("INFO: Using Generalized Propensity Score Estimator")

    def fit(
        self,
        data: pd.DataFrame,
        effect_modifier_names: Optional[List[str]] = None,
    ):
        """
        Fits the estimator with data for effect estimation
        :param data: data frame containing the data
        :param treatment: name of the treatment variable
        :param outcome: name of the outcome variable
        :param effect_modifiers: Variables on which to compute separate
                    effects, or return a heterogeneous effect function. Not all
                    methods support this currently.
        """
        return super().fit(data, effect_modifier_names=effect_modifier_names)

    def construct_symbolic_estimator(self, estimand):
        expr = "b: " + ",".join(estimand.outcome_variable) + "~"
        var_list = estimand.treatment_variable + estimand.get_backdoor_variables()
        expr += "+".join(var_list)
        if self._effect_modifier_names:
            interaction_terms = [
                "{0}*{1}".format(x[0], x[1])
                for x in itertools.product(estimand.treatment_variable, self._effect_modifier_names)
            ]
            expr += "+" + "+".join(interaction_terms)
        return expr

    def _build_model(self, data: pd.DataFrame):
        """
        Fits a GAM that predicts the outcome (expected to be continuous) 
        from the continuous treatment and GPS
        """
        features = self._build_features(data)
        treatment = data[self._target_estimand.treatment_variable[0]]
        covariates = data.drop(
            [
                self._target_estimand.treatment_variable[0],
                self._target_estimand.outcome_variable[0]
            ],
            axis = 1
        )

        gps_function = self._determine_gps_function(treatment, covariates)
        gps_values = gps_function(treatment)
        X = np.column_stack((treatment.values, gps_values))

        bs = BSplines(
            X, 
            df=[5, 5], 
            degree=[3, 3],  
            include_intercept=True
        )

        model = GLMGam(
            data[self._target_estimand.outcome_variable[0]],
            exog=None, 
            smoother=bs
        ).fit()

        return (features, model)

    def _determine_gps_function(self, treatment, covariates):
        """
        Based on the distribution of continuous treatment values, 
        and model deviances, this function determines which GPS function 
        family should be used.
        """
        # If any negative values in treatment, you must use the normal GLM family.
        if any(treatment < 0):
            return self._create_normal_gps_function(treatment, covariates)

        # If all treatment values are positive, figure out best-fitting family
        return self._find_best_gps_model(treatment, covariates)

    def _create_normal_gps_function(self, treatment, covariates):
        """
        Models the GPS using a GLM of the Gaussian family
        """
        normal_gps_model = sm.GLM(
            treatment, covariates, family=sm.families.Gaussian()
        ).fit()

        pred_treat = normal_gps_model.fittedvalues
        sigma = np.std(normal_gps_model.resid_response)

        def gps_function(treatment_val, pred_treat=pred_treat, sigma=sigma):
            return norm.pdf(treatment_val, pred_treat, sigma)

        return gps_function

    def _create_lognormal_gps_function(self, treatment, covariates):
        """
        Models the GPS using a GLM of the Gaussian family (assumes treatment is lognormal)
        """
        lognormal_gps_model = sm.GLM(
            np.log(treatment), covariates, family=sm.families.Gaussian()
        ).fit()

        pred_log_treat = lognormal_gps_model.fittedvalues
        sigma = np.std(lognormal_gps_model.resid_response)

        def gps_function(treatment_val, pred_log_treat=pred_log_treat, sigma=sigma):
            return norm.pdf(np.log(treatment_val), pred_log_treat, sigma)

        return gps_function

    def _create_gamma_gps_function(self, treatment, covariates):
        """Models the GPS using a GLM of the Gamma family"""
        gamma_gps_model = sm.GLM(
            treatment, covariates, family=sm.families.Gamma(Inverse_Power())
        ).fit()

        mu = gamma_gps_model.mu
        scale = gamma_gps_model.scale
        shape = mu / gamma_gps_model.scale

        def gps_function(treatment_val):
            return gamma.pdf(treatment_val, a=shape, loc=0, scale=scale)

        return gps_function

    def _find_best_gps_model(self, treatment, covariates):
        """
        This function compares a few different gps models and picks 
        the one with the lowest deviance
        """
        models_to_try = {
            "normal_gps_model": self._create_normal_gps_function(treatment, covariates),
            "lognormal_gps_model": self._create_lognormal_gps_function(treatment, covariates),
            "gamma_gps_model": self._create_gamma_gps_function(treatment, covariates),
        }

        model_comparison_dict = {}

        for key, value in models_to_try.items():
            model_comparison_dict[key] = value[1]

        # Return model with lowest deviance
        best_model = min(model_comparison_dict, key=model_comparison_dict.get)

        return (
            best_model,
            models_to_try[best_model][0],
            models_to_try[best_model][1],
        )

    def predict_fn(self, data, model, features):
        return model.predict(features)

    def _estimate_confidence_intervals(self, confidence_level, method=None):
        if self._effect_modifier_names:
            # The average treatment effect is a combination of different
            # regression coefficients. Complicated to compute the confidence
            # interval analytically. For example, if y=a + b1.t + b2.tx, then
            # the average treatment effect is b1+b2.mean(x).
            # Refer Gelman, Hill. ARM Book. Chapter 9
            # http://www.stat.columbia.edu/~gelman/arm/chap9.pdf
            # TODO: Looking for contributions
            raise NotImplementedError
        else:
            conf_ints = self.model.conf_int(alpha=1 - confidence_level)
            # For a linear regression model, the causal effect of a variable is equal to the coefficient corresponding to the
            # variable. Hence, the model by default outputs the confidence interval corresponding to treatment=1 and control=0.
            # So for custom treatment and control values, we must multiply the confidence interval by the difference of the two.
            return (self._treatment_value - self._control_value) * conf_ints.to_numpy()[
                1 : (len(self._target_estimand.treatment_variable) + 1), :
            ]

    def _estimate_std_error(self, method=None):
        if self._effect_modifier_names:
            raise NotImplementedError
        else:
            std_error = self.model.bse[1 : (len(self._target_estimand.treatment_variable) + 1)]

            # For a linear regression model, the causal effect of a variable is equal to the coefficient corresponding to the
            # variable. Hence, the model by default outputs the standard error corresponding to treatment=1 and control=0.
            # So for custom treatment and control values, we must multiply the standard error by the difference of the two.
            return (self._treatment_value - self._control_value) * std_error.to_numpy()

    def _test_significance(self, estimate_value, method=None):
        pvalue = self.model.pvalues[1 : (len(self._target_estimand.treatment_variable) + 1)]
        return {"p_value": pvalue.to_numpy()}

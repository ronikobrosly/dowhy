"""
A temporary file for experimenting with the new GPS functionality.
TODO: Remove before eventually merging  
"""


from dowhy import CausalModel
import dowhy.datasets

# Generate some sample data
data = dowhy.datasets.linear_dataset(
    beta                          = 10,
    num_common_causes             = 5,
    num_samples                   = 1000,
    num_instruments               = 1,
    num_effect_modifiers          = 0,
    num_treatments                = 1,
    treatment_is_binary           = False,
    treatment_is_category         = False
)


# Step 1: Create a causal model from the data and given graph
model = CausalModel(
    data=data["df"],
    treatment=data["treatment_name"],
    outcome=data["outcome_name"],
    graph=data["gml_graph"]
)

# Step 2: Identify causal effect and return target estimands
identified_estimand = model.identify_effect()

# Step 3: Estimate the target estimand using a statistical method.
estimate = model.estimate_effect(
    identified_estimand,
    method_name="backdoor.generalized_propensity_score"
)
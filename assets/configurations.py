approximator_settings = {
    "lstm1_hidden_units": 512,
    "lstm2_hidden_units": 256,
    "lstm3_hidden_units": 128,
    "trainer": {
        "max_to_keep": 1,
        "default_lr": 5e-4,
        "memory": False,
    },
    "local_amortizer_settings": {
        "num_coupling_layers": 8,
        "coupling_design": 'interleaved'
    },
    "global_amortizer_settings": {
        "num_coupling_layers": 6,
        "coupling_design": 'interleaved'
    },
}

default_prior_settings = {
    # v0, b_tau, beta
    "gamma_shape_1": (0.0, 0.0, 25),
    "gamma_shape_2": (1.0, 0.2, 25),
    # sigma_b_v, sigma_a, sigma_tau_0
    "eta_shape_1": (0.0, 0.0, 0.0),
    "eta_shape_2": (0.1, 0.1, 0.05),
}

default_lower_bounds = (0.0, 0.0, 0.0)
default_upper_bounds = (6.0, 6.0, 2.0)
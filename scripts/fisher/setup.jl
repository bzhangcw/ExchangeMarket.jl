
method_kwargs = [
    # Keep log-barrier method at first position
    [:LogBar,
        HessianBar,
        Dict(
            :tol => 1e-10, :maxiter => 20,
            :optimizer => CESAnalytic,
            :option_mu => :pred_corr,
            :option_step => :logbar,
            :linsys => :DRq,
        )
    ],
    [:PathFol,
        HessianBar,
        Dict(
            :tol => 1e-12, :optimizer => CESAnalytic,
            :maxiter => 200, :μ => 1e0,
            :option_mu => :nothing,
            :option_step => :homotopy,
            :linsys => :DRq,
        )
    ],
    [:LogBarAfscKrylov,
        HessianBar,
        Dict(
            :tol => 1e-12, :maxiter => 20,
            :optimizer => CESAnalytic,
            :option_mu => :normal,
            :option_step => :affinesc,
            # :linsys => :DRq,
            :linsys => :krylov,
            # :linsys => :direct,
        )
    ],
    [:DampedNS,
        HessianBar,
        Dict(
            :tol => 1e-10, :optimizer => CESAnalytic,
            :maxiter => 200, :μ => 0.0,
            :option_mu => :nothing,
            :option_step => :damped_ns,
            :linsys => :DRq,
            # :linsys => :direct,
        )
    ],
    [:Tât,
        MirrorDec, Dict(
            :tol => 1e-7, :α => 500.0,
            :optimizer => CESAnalytic,
            :option_step => :eg,
            :option_stepsize => :cc13
        )
    ],
    [:PropRes,
        MirrorDec,
        Dict(
            :tol => 1e-7, :α => 500.0,
            :optimizer => PR,
            :option_step => :shmyrev
        )
    ],
]

# -----------------------------------------------------------------------
# setups for more complex best response mappings, 
#   may include linear constraints
# -----------------------------------------------------------------------
method_kwargs_br_more_complex = [
    [:HessianBarPD,
        HessianBar,
        Dict(
            :tol => 1e-10, :maxiter => 20,
            :optimizer => CESConic,
            :option_mu => :pred_corr,
            :linsys => :DRq,
            :option_step => :logbar
        )
    ],
]
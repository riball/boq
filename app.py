import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.core.problem import Problem

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(page_title="BOQ Sustainability Optimizer", layout="wide")
st.title("🌱 Sustainable BOQ Optimization Tool")
st.markdown("Minimize Emissions under Lifecycle Cost Constraint")

# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader("Upload Cleaned_BOQ.csv", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully")

    # ============================================================
    # EMISSION + COST DATA
    # ============================================================

    emission_factors = {
        'M5': 150, 'M10': 175, 'M15': 200, 'M20': 230,
        'M30': 290, 'M40': 350, 'M45': 380, 'M50': 410
    }

    RCC_STEEL_FACTOR = 115

    grade_cost_per_m3 = {
        'M5': 3500, 'M10': 3800, 'M15': 4200, 'M20': 4800,
        'M30': 5800, 'M40': 7000, 'M45': 7800, 'M50': 8500
    }

    lifecycle_penalty = {
        'M5': 3.5, 'M10': 2.8, 'M15': 2.2, 'M20': 1.8,
        'M30': 1.3, 'M40': 1.1, 'M45': 1.05, 'M50': 1.0
    }

    # Filter only concrete volume rows
    df_opt = df[df['Unit'] == 'M3'].copy().reset_index(drop=True)

    if len(df_opt) == 0:
        st.error("No M3 concrete rows found.")
        st.stop()

    # ============================================================
    # STRUCTURAL GROUPING
    # ============================================================

    def get_struct_category(elem):
        elem = str(elem).lower()
        if 'column' in elem: return 'column'
        elif 'beam' in elem: return 'beam'
        elif 'slab' in elem or 'floor' in elem: return 'slab'
        elif 'footing' in elem or 'raft' in elem or 'foundation' in elem: return 'foundation'
        elif 'wall' in elem: return 'wall'
        elif 'stair' in elem: return 'stair'
        elif 'basement' in elem: return 'basement'
        else: return 'general'

    df_opt['struct_category'] = df_opt['Structural Element'].apply(get_struct_category)

    groups = df_opt.groupby(['Material', 'struct_category'])
    group_keys = list(groups.groups.keys())
    n_groups = len(group_keys)

    # ============================================================
    # GRADE BOUNDS
    # ============================================================

    GRADE_LIST = ['M5', 'M10', 'M15', 'M20', 'M30', 'M40', 'M45', 'M50']
    IDX_TO_GRADE = {i: g for i, g in enumerate(GRADE_LIST)}

    min_grade_idx = {'column': 4, 'beam': 4, 'slab': 3, 'foundation': 2,
                     'wall': 2, 'stair': 3, 'basement': 3, 'general': 1}
    max_grade_idx = {'column': 7, 'beam': 7, 'slab': 6, 'foundation': 5,
                     'wall': 5, 'stair': 5, 'basement': 6, 'general': 5}

    xl = np.array([min_grade_idx.get(k[1], 1) for k in group_keys])
    xu = np.array([max_grade_idx.get(k[1], 7) for k in group_keys])

    # ============================================================
    # BASELINE CALCULATION
    # ============================================================

    original_emissions_absolute = df_opt.apply(
        lambda r: r['Quantity'] * (
            emission_factors[r['Grade']] +
            (RCC_STEEL_FACTOR if r['Material'] == 'RCC' else 0)
        ),
        axis=1
    ).sum()

    original_cost_absolute = df_opt.apply(
        lambda r: r['Quantity'] *
                  grade_cost_per_m3[r['Grade']] *
                  lifecycle_penalty[r['Grade']],
        axis=1
    ).sum()

    # ============================================================
    # OPTIMIZATION PROBLEM
    # ============================================================

    class BOQConstrainedOptimization(Problem):

        def __init__(self):
            super().__init__(
                n_var=n_groups,
                n_obj=1,
                n_constr=1,
                xl=xl,
                xu=xu,
                vtype=int
            )

        def _evaluate(self, X, out, *args, **kwargs):

            total_emissions = []
            constraint_cost = []

            for solution in X:

                emissions = 0
                lifecycle_cost = 0

                for group_idx, grade_idx in enumerate(solution):

                    grade = IDX_TO_GRADE[int(round(grade_idx))]
                    material, category = group_keys[group_idx]
                    group_rows = groups.get_group((material, category))

                    for _, row in group_rows.iterrows():

                        qty = row['Quantity']

                        ef = emission_factors[grade] + \
                             (RCC_STEEL_FACTOR if material == 'RCC' else 0)

                        emissions += qty * ef
                        lifecycle_cost += qty * grade_cost_per_m3[grade] * lifecycle_penalty[grade]

                total_emissions.append(emissions)
                constraint_cost.append(lifecycle_cost - original_cost_absolute)

            out["F"] = np.array(total_emissions).reshape(-1, 1)
            out["G"] = np.array(constraint_cost).reshape(-1, 1)

    # ============================================================
    # RUN OPTIMIZATION
    # ============================================================

    if st.button("🚀 Run Optimization"):

        with st.spinner("Running optimization..."):

            problem = BOQConstrainedOptimization()

            algorithm = NSGA2(
                pop_size=80,
                sampling=IntegerRandomSampling(),
                crossover=SBX(prob=0.9, eta=15),
                mutation=PM(prob=1.0, eta=20),
                eliminate_duplicates=True
            )

            result = minimize(
                problem,
                algorithm,
                get_termination("n_gen", 80),
                seed=42,
                verbose=False
            )

        st.success("✅ Optimization Complete")

        feasible_mask = result.G.flatten() <= 0
        feasible_emissions = result.F[feasible_mask]

        col1, col2, col3 = st.columns(3)

        col1.metric("Original Emissions (M kg CO₂e)",
                    f"{original_emissions_absolute/1e6:.2f}")

        if len(feasible_emissions) > 0:

            best_emission = feasible_emissions.min()
            reduction_pct = (original_emissions_absolute - best_emission) / original_emissions_absolute * 100

            col2.metric("Optimized Emissions (M kg CO₂e)",
                        f"{best_emission/1e6:.2f}")

            col3.metric("Emission Reduction (%)",
                        f"{reduction_pct:.2f}%")

        else:
            st.error("No feasible solution found under cost constraint.")

        # Download results
        results_df = pd.DataFrame({
            "Emissions_Million_kgCO2e": result.F.flatten() / 1e6
        })

        st.download_button(
            "📥 Download Optimization Results",
            data=results_df.to_csv(index=False),
            file_name="optimized_solutions.csv",
            mime="text/csv"
        )
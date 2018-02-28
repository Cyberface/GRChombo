/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

// General includes common to most GR problems
#include "ScalarFieldLevel.hpp"
#include "BoxLoops.hpp"
#include "NanCheck.hpp"
#include "PositiveChiAndAlpha.hpp"
#include "TraceARemoval.hpp"

// For RHS update
#include "MatterCCZ4.hpp"

// For constraints calculation
#include "MatterConstraints.hpp"

// For tag cells
#include "PhiAndChiTaggingCriterion.hpp"

// Problem specific includes
#include "BinaryBH.hpp"
#include "ChiRelaxationWithBH.hpp"
#include "ComputePack.hpp"
#include "Potential.hpp"
#include "ScalarField.hpp"
#include "SetValue.hpp"
#include "Weyl4Scalar.hpp"

// Things to do at each advance step, after the RK4 is calculated
void ScalarFieldLevel::specificAdvance()
{
    // Enforce trace free A_ij and positive chi and alpha
    BoxLoops::loop(make_compute_pack(TraceARemoval(), PositiveChiAndAlpha()),
                   m_state_new, m_state_new, FILL_GHOST_CELLS);

    // Check for nan's
    if (m_p.nan_check)
        BoxLoops::loop(NanCheck(), m_state_new, m_state_new, SKIP_GHOST_CELLS,
                       disable_simd());
}

// Initial data for field and metric variables
void ScalarFieldLevel::initialData()
{
    CH_TIME("ScalarFieldLevel::initialData");
    if (m_verbosity)
        pout() << "ScalarFieldLevel::initialData " << m_level << endl;

    // First set everything to zero ... we don't want undefined values in
    // constraints etc, then  initial conditions for scalar field - here a
    // constant in space which will oscillate in the potential, then set up BHs

    // Set up the compute class for the BinaryBH initial data
    BinaryBH binary(m_p.bh1_params, m_p.bh2_params, m_dx);
    // Set up the field value
    SetValue set_phi(m_p.field_amplitude, Interval(c_phi, c_phi));

    // Now loop over the grid
    BoxLoops::loop(make_compute_pack(SetValue(0.0), binary, set_phi),
                   m_state_new, m_state_new, FILL_GHOST_CELLS);
}

// Things to do before outputting a checkpoint file
void ScalarFieldLevel::preCheckpointLevel()
{
    fillAllGhosts();
    Potential potential(m_p.potential_params);
    ScalarFieldWithPotential scalar_field(potential);
    BoxLoops::loop(MatterConstraints<ScalarFieldWithPotential>(
                       scalar_field, m_dx, m_p.G_Newton),
                   m_state_new, m_state_new, SKIP_GHOST_CELLS);
}

// Things to do in RHS update, at each RK4 step
void ScalarFieldLevel::specificEvalRHS(GRLevelData &a_soln, GRLevelData &a_rhs,
                                       const double a_time)
{

    // Relaxation function for chi - this will eventually be done separately
    // with hdf5 as input
    if (m_time < m_p.relaxtime)
    {
        // Calculate chi relaxation right hand side
        // Note this assumes conformal chi and Mom constraint trivially
        // satisfied  No evolution in other variables, which are assumed to
        // satisfy constraints per initial conditions
        Potential potential(m_p.potential_params);
        ScalarFieldWithPotential scalar_field(potential);
        ChiRelaxationWithBH<ScalarFieldWithPotential> relaxation(
            scalar_field, m_dx, m_p.relaxspeed, m_p.G_Newton);
        SetValue set_other_values_zero(0.0, Interval(c_h11, c_Mom3));
        auto compute_pack1 =
            make_compute_pack(relaxation, set_other_values_zero);
        BoxLoops::loop(compute_pack1, a_soln, a_rhs, SKIP_GHOST_CELLS);
    }
    else
    {

        // Enforce trace free A_ij and positive chi and alpha
        BoxLoops::loop(
            make_compute_pack(TraceARemoval(), PositiveChiAndAlpha()), a_soln,
            a_soln, FILL_GHOST_CELLS);

        // Calculate MatterCCZ4 right hand side with matter_t = ScalarField
        // We don't want undefined values floating around in the constraints so
        // zero these
        Potential potential(m_p.potential_params);
        ScalarFieldWithPotential scalar_field(potential);
        MatterCCZ4<ScalarFieldWithPotential> my_ccz4_matter(
            scalar_field, m_p.ccz4_params, m_dx, m_p.sigma, m_p.formulation,
            m_p.G_Newton);
        SetValue set_constraints_zero(0.0, Interval(c_Weyl4_Re, c_Mom3));
        auto compute_pack2 =
            make_compute_pack(my_ccz4_matter, set_constraints_zero);
        BoxLoops::loop(compute_pack2, a_soln, a_rhs, SKIP_GHOST_CELLS);
    }
}

// Things to do at ODE update, after soln + rhs
void ScalarFieldLevel::specificUpdateODE(GRLevelData &a_soln,
                                         const GRLevelData &a_rhs, Real a_dt)
{
    // Enforce trace free A_ij
    BoxLoops::loop(TraceARemoval(), a_soln, a_soln, FILL_GHOST_CELLS);

    // Calculate the Weyl4 Scalar
    fillAllGhosts();
    BoxLoops::loop(Weyl4Scalar(m_p.center, m_dx), a_soln, a_soln,
                   SKIP_GHOST_CELLS);
}

// Specify if you want any plot files to be written, with which vars
void ScalarFieldLevel::specificWritePlotHeader(
    std::vector<int> &plot_states) const
{
    plot_states = {c_phi, c_chi, c_Weyl4_Re, c_Weyl4_Im};
}

void ScalarFieldLevel::computeTaggingCriterion(FArrayBox &tagging_criterion,
                                               const FArrayBox &current_state)
{
    // regrid based on gradients and tagging region
    BoxLoops::loop(PhiAndChiTaggingCriterion(m_dx, m_p.regrid_threshold_phi,
                                             m_p.regrid_threshold_chi, m_p.L, 
                                             m_p.extraction_radius, m_level, 
                                             m_p.extraction_level),
                   current_state, tagging_criterion);
}

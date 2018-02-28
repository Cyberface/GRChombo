/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#ifndef CHIRELAXATIONWITHBH_HPP_
#define CHIRELAXATIONWITHBH_HPP_

#include "ChiRelaxation.hpp"
#include "FourthOrderDerivatives.hpp"
#include "MatterCCZ4.hpp"

//!  Calculates RHS for relaxation of the conformal factor, for initial
//!  conditions
/*!
     The class calculates the RHS evolution for the relaxation of the conformal
     factor, as in ChiRelaxation, but assuming that the lapse is pre collapsed
     and thus that the rhs is frozen at the puncture
     TODO: This should inherit from ChiRelaxation but it doesn't want to yet...
     \sa m_relax_speed() \sa ChiRelaxation
*/

template <class matter_t> class ChiRelaxationWithBH
{
  protected:
    ChiRelaxation<matter_t> my_relaxation;
    FourthOrderDerivatives m_deriv;

  public:
    // Use the variable definitions in MatterCCZ4
    template <class data_t>
    using Vars = typename MatterCCZ4<matter_t>::template Vars<data_t>;

    template <class data_t>
    using Diff2Vars = typename MatterCCZ4<matter_t>::template Diff2Vars<data_t>;

    //! Inherit constructor of class ChiRelaxation
    ChiRelaxationWithBH(matter_t a_matter, double a_dx, double relax_speed,
                        double G_Newton = 1.0)
        : my_relaxation(a_matter, a_dx, relax_speed, G_Newton), m_deriv(a_dx)
    {
    }

    //! The compute member which calculates the RHS at each point
    template <class data_t> void compute(Cell<data_t> current_cell) const
    {
        // copy data from chombo gridpoint into local variable and calculate
        // derivs
        const auto vars = current_cell.template load_vars<Vars>();
        const auto d1 = m_deriv.template diff1<Vars>(current_cell);
        const auto d2 = m_deriv.template diff2<Diff2Vars>(current_cell);
        const auto advec =
            m_deriv.template advection<Vars>(current_cell, vars.shift);

        // work out RHS including advection
        // All components that are not explicitly set in rhs_equation are 0
        Vars<data_t> rhs;
        VarsTools::assign(rhs, 0.);
        my_relaxation.rhs_equation(rhs, vars, d1, d2,
                                   advec); // using function in relax

        // freeze evolution at the puncture
        rhs.chi = rhs.chi * pow(vars.lapse, 4.0);
        FOR2(i, j) { rhs.A[i][j] = rhs.A[i][j] * pow(vars.lapse, 4.0); }

        // Write the rhs into the output FArrayBox
        current_cell.store_vars(rhs);
    }
};

#endif /* CHIRELAXATIONWITHBH_HPP_ */

/* Basilisk 1D spherically symmetric compressible collapse (local vs ambient Mach) */
#include "spherisym.h"
#include "bubble.h"

#ifndef PG0
# define PG0 100.0
#endif
#ifndef GAMMA_G
# define GAMMA_G 1.4
#endif

#ifndef PINF_D
# define PINF_D 500.0
#endif

#ifndef SIGMA_L
# define SIGMA_L 0.0
#endif
#ifndef MU_L
# define MU_L 0.0
#endif
#ifndef MAXLEV
# define MAXLEV 13
#endif

int main()
{
  pg0  = PG0;
  pinf = PINF_D;
  tend = 2.0*0.915*sqrt (1./(pinf - pg0));

  rhoL = 1.;

  rhoG = 1.0e-5*pg0;

  gamma1 = 7.14;

  gamma2 = GAMMA_G;
  PI1 = 300*100.0;

  mu1 = MU_L;
  mu2 = MU_L*1.0e-4;
  f.sigma = SIGMA_L;

  CFLac = 10.;
  L0 = 50;
  N = 1 << MAXLEV;
  MAXLEVEL = MAXLEV;
  run();
}

event mach (i++)
{
  static double v_prev = -1., t_prev = 0.;
  double volume = 0.;
  foreach (reduction(+:volume))
    volume += dv()*(1. - f[]);

  double R = pow (3.*volume, 1./3.);
  if (v_prev >= 0. && t > t_prev) {

    double Rp = pow (3.*v_prev, 1./3.);
    double Rdot = (R - Rp)/(t - t_prev);
    double c = sqrt (gamma1*(pinf + PI1)/rhoL);
    fprintf (stderr, "%12.8f %12.8f %14.6f %12.6f\n",
	     t, R, Rdot, fabs(Rdot)/c);
  }
  v_prev = volume; t_prev = t;
}

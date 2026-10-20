/* Basilisk axisymmetric VOF collapse with seeded l=2 Legendre perturbation */
#include "axi.h"
#include "bubble.h"

#ifndef EPS
# define EPS 0.01
#endif
#ifndef NFRAMES
# define NFRAMES 60
#endif

#ifndef MAXLEV
# define MAXLEV 11
#endif

#ifndef PG0
# define PG0 100.0
#endif

int main()
{
  pg0  = PG0;
  pinf = 500.;
  tend = 2.50001*0.915*sqrt (1./(pinf - pg0));

  MAXLEVEL = MAXLEV;
#if INVISCID

  mu1 = 0.;
  mu2 = 0.;
#else
  double Reynolds = 10.;
  mu1 = sqrt(pinf - pg0)/Reynolds;
  mu2 = mu1*0.0001;
#endif

  CFLac = 10.;
  gamma2 = 1.4;
  gamma1 = 7.14;
  PI1 = 300*pg0;

  L0 = 50;
  N = 1 << MINLEVEL;
  run();
}

event seed_mode (i = 0)
{
  fraction (f, sq(x) + sq(y)
	    - sq(R0*(1. + EPS*(3.*sq(x)/(sq(x) + sq(y) + 1e-30) - 1.)/2.)));

  foreach() {
    frho1[] = f[]*rhoL;
    frho2[] = (1. - f[])*rhoG;
    double r = sqrt(sq(x) + sq(y));
    double pL = pinf*(1. - R0/r) + (pg0 - 2*f.sigma/R0)*R0/r;
    fE1[] = f[]*(pL + PI1*gamma1)/(gamma1 - 1.);
    fE2[] = (1. - f[])*pg0/(gamma2 - 1.);
    p[] = average_pressure (point);
  }
}

int frame = 0;

event shape (t += 2.50001*0.915*sqrt(1./(500. - PG0))/NFRAMES;
	     t <= tend)
{
  char name[80];
  sprintf (name, "facets-%04d.dat", frame);
  FILE * fp = fopen (name, "w");
  output_facets (f, fp);
  fclose (fp);

  static FILE * idx = NULL;
  if (!idx) idx = fopen ("frames.dat", "w");
  fprintf (idx, "%d %.8f\n", frame, t);
  fflush (idx);
  frame++;
}

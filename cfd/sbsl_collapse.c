/* Basilisk axisymmetric VOF collapse at the R0 = 7 um SBSL operating point */
#include "axi.h"
#include "bubble.h"

#ifndef AMPL
# define AMPL 1.375
#endif
#ifndef RMAX
# define RMAX 9.348
#endif
#ifndef EPS
# define EPS 0.001
#endif
#ifndef MAXLEV
# define MAXLEV 12
#endif
#ifndef NFRAMES
# define NFRAMES 80
#endif

int main()
{
  R0   = RMAX;
  pinf = 1.0 + AMPL;

  pg0 = (1.0 + 2.0*0.10256) * pow (1.0/RMAX, 3.0*5.0/3.0);

  rhoL = 1.0;
  rhoG = 1.0e-3 * pg0;

  gamma1 = 7.15;
  gamma2 = 5.0/3.0;
  PI1    = 3026.0;

  mu1 = 0.014225;
  mu2 = mu1*1.0e-4;
  f.sigma = 0.10256;

  tend = 1.2*0.91468*RMAX*sqrt (1.0/(pinf - pg0));

  CFLac = 10.;
  L0 = 50;
  N = 1 << MINLEVEL;
  MAXLEVEL = MAXLEV;
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

event shape (t += 1.2*0.91468*RMAX*sqrt(1.0/(1.0 + AMPL - 1.6883e-05))/NFRAMES; t <= tend)
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

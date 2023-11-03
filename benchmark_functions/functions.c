/// Sphere function
double Sphere(double *vars, unsigned int nvars) {
	unsigned int i;
	double fitness = 0.0;

	for (i = 0; i < nvars; ++i) {
		fitness += (double) (vars[i] * vars[i]);
	}

	return fitness;
}

/// Quadric function
double Quadric(double *vars, unsigned int nvars) {
	unsigned int i, j;
	double fitaux, fitness = 0.0;

	for (i = 1; i <= nvars; ++i) {
		fitaux = 0;

		for (j = 0; j < i; ++j) {
			fitaux += (double) vars[j];
		}

		fitness += fitaux * fitaux;
	}
	return fitness;
}
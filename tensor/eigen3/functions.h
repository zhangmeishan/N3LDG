//useful functions
inline dtype fequal(const dtype& x) {
	return x;
}

inline dtype ftanh(const dtype& x) {
	return tanh(x);
}

inline dtype fsigmoid(const dtype& x) {
	return 1.0 / (1.0 + exp(-x));
}

inline dtype frelu(const dtype& x) {
	if (x <= 0) return 0;
	return x;
}

inline dtype fleaky_relu(const dtype& x) {
	if (x < 0) return (0.1*x);
	return x;
}
inline dtype fexp(const dtype& x) {
	return exp(x);
}

//derive function
inline dtype dequal(const dtype& x, const dtype& y) {
	return 1;
}

inline dtype dtanh(const dtype& x, const dtype& y) {
	return (1 + y) * (1 - y);
}

inline dtype dleaky_relu(const dtype& x, const dtype& y) {
	if (x < 0) return 0.1;
	return 1;
}

inline dtype dsigmoid(const dtype& x, const dtype& y) {
	return (1 - y) * y;
}

inline dtype drelu(const dtype& x, const dtype& y) {
	if (x <= 0) return 0;
	return 1;
}

inline dtype dexp(const dtype& x, const dtype& y) {
	return y;
}

void randGenerationUniform(
		std::uniform_real_distribution<dtype>& distribution, 
		std::default_random_engine& generator, dtype* val, int size) {
	for(int idx = 0; idx < size; idx++) {
		val[idx] = distribution(generator);
	}
}

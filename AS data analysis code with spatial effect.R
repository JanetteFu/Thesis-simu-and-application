AS_spatial <- "data {
  int<lower=1> N; // Number of data rows
  //int<lower=1> N_val; // Number of data rows
  int<lower=1> M; // Number of features
  matrix[N, M] X; //parameters 
  int<lower =0, upper = 1> y[N];
  //matrix[N_val, M] X_val; 
  int<lower=1> N_area;
  int<lower=0> N_edges;                   // number of edges
  int<lower=1, upper=N> node1[N_edges];   // node1[i] adjacent to node2[i]
  int<lower=1, upper=N> node2[N_edges];   // and node1[i] < node2[i]
  int S[N];   //area subject i belongs to
}

transformed data {
  real m0 = 10;           
  real slab_scale = 4;    
  real slab_scale2 = square(slab_scale);
  real slab_df = 20;     
  real half_slab_df = 0.5 * slab_df;
}

parameters {
  vector[M] beta_tilde;
  vector<lower=0>[M] lambda;
  real<lower=0> c2_tilde;
  real tau_tilde;
  real alpha;
  real phi;
  real<lower=0> sigma;      
  vector[N_area] zeta;      // random (spatial) effects
}

transformed parameters {
  vector[M] beta;
  real tau0 = (m0 / (M - m0)) * (2 / sqrt(1.0 * N));
  real tau = tau0 * tau_tilde;
  
  real c2 = slab_scale2 * c2_tilde;
  
  vector[M] lambda_tilde;
  lambda_tilde = sqrt( c2 * square(lambda) ./ (c2 + square(tau) *
                                                 square(lambda)) );
  
  beta = tau * lambda_tilde .* beta_tilde;
}

model {
  beta_tilde ~ normal(0, 1);
  lambda ~ cauchy(0, 1);
  tau_tilde ~ cauchy(0, 1);
  c2_tilde ~ inv_gamma(half_slab_df, half_slab_df);
  alpha ~ normal(0, 2);
  sigma ~ gamma(1,1);
  
  target += -0.5 * dot_self(zeta[node1] - zeta[node2]);
  sum(zeta) ~ normal(0, 0.001 * N);  
  
  for (n in 1:N) {
    y[n] ~ bernoulli_logit(X[n,] * beta + alpha + sigma*zeta[S[n]]);
  } 
}"
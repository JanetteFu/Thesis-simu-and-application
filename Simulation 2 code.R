### The hierarchical model that simultaneously impute and fit
hierarchical <- "data {
  int<lower=1> N; // Number of data rows
  int<lower = 1> N_obs; // Number of observed
  int<lower = 1> N_mis; // Number of missing
  int<lower=1> M; // Number of features
  int<lower=1> R; // Number of sites
  matrix[N_obs, M] X_obs; // matrix of features for observed outcomes
  matrix[N_mis, M] X_mis; // matrix of features for missing outcomes
  int S_obs[N_obs]; //vector of site for observed outcomes
  int S_mis[N_mis]; //vector of site for missing outcomes
  int S[N];
  matrix[N,M] X_val;
  
  int<lower =0, upper = 1> y_obs[N_obs];
  int<lower =0, upper = 1> y_obs2[N_obs];
  int<lower =0, upper = 1> y_mis2[N_mis];
}

transformed data {
  real m0 = 10;           
  real slab_scale = 3;    
  real slab_scale2 = square(slab_scale);
  real slab_df = 20;      
  real half_slab_df = 0.5 * slab_df;
}

parameters {
  matrix[M, R] beta_tilde;
  
  matrix<lower=0>[M, R] lambda;
  real<lower=0> c2_tilde;
  vector<lower=0>[R] tau_tilde;
  real alpha;
  matrix[M, R] beta_tilde2;
  
  matrix<lower=0>[M, R] lambda2;
  real<lower=0> c2_tilde2;
  vector<lower=0>[R] tau_tilde2;
  real alpha2;
  real phi;
}

transformed parameters {
  matrix[M, R] beta;
  matrix[M, R] beta2;
  {
    real tau0 = (m0 / (M - m0)) * (2 / sqrt(1.0 * N));
    vector[R] tau = tau0 * tau_tilde; // tau ~ cauchy(0, tau0)
    vector[R] tau2 = tau0 * tau_tilde2;
    real c2 = slab_scale2 * c2_tilde;
    
    matrix[M, R] lambda_tilde;
    matrix[M, R] lambda_tilde2;
    for (r in 1:R)
      lambda_tilde[,r] = sqrt( c2 * square(lambda[,r]) ./ (c2 +
                                                             square(tau[r]) * square(lambda[,r])) );
    for (r in 1:R)
      lambda_tilde2[,r] = sqrt( c2 * square(lambda2[,r]) ./ (c2 +
                                                               square(tau2[r]) * square(lambda2[,r])) );
    for (r in 1:R) beta[,r] = tau[r] * lambda_tilde[,r] .*
      beta_tilde[,r];
    for (r in 1:R) beta2[,r] = tau2[r] * lambda_tilde2[,r] .*
      beta_tilde2[,r];
  }
}
model {
  for (r in 1:R) beta_tilde[,r] ~ normal(0, 1);
  for (r in 1:R) lambda[,r] ~ cauchy(0, 1);
  tau_tilde ~ cauchy(0, 1);
  c2_tilde ~ inv_gamma(half_slab_df, half_slab_df);
  alpha ~ normal(0, 2);
  
  for (r in 1:R) beta_tilde2[,r] ~ normal(0, 1);
  for (r in 1:R) lambda2[,r] ~ cauchy(0, 1);
  tau_tilde2 ~ cauchy(0, 1);
  c2_tilde2 ~ inv_gamma(half_slab_df, half_slab_df);
  alpha2 ~ normal(0, 2);
  
  for (n in 1:N_obs) {
    y_obs[n] ~ bernoulli_logit(X_obs[n] * beta[,S_obs[n]] + alpha);
    y_obs2[n] ~ bernoulli_logit(X_obs[n] * beta2[,S_obs[n]] + alpha2 +
                                  phi*inv_logit(X_obs[n] * beta[, S_obs[n]] + alpha));
  }
  
  for (n in 1:N_mis) {
    real mu_mis = X_mis[n] * beta[,S_mis[n]] + alpha;
    target += log_mix(inv_logit(mu_mis), bernoulli_logit_lpmf(1 | mu_mis), 
                      bernoulli_logit_lpmf(0 | mu_mis));
    y_mis2[n] ~ bernoulli_logit(X_mis[n] * beta2[,S_mis[n]] + alpha2 +
                                  phi*inv_logit(X_mis[n] * beta[, S_mis[n]] + alpha));                                 
  }
}

generated quantities {
  vector[N] yhat2_val;
  vector[N] yhat_val;
  for (n in 1:N) {
    yhat_val[n] = inv_logit(X_val[n] * beta[, S[n]] + alpha);
    yhat2_val[n] = inv_logit(X_val[n] * beta2[, S[n]] + alpha2 +
                               phi*inv_logit(X_val[n] * beta[, S[n]] + alpha));
  }
}
"
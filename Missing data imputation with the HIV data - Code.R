hiv_hierach <- "data {
  int<lower=1> N; // Number of data rows
  int<lower = 1> N_obs; // Number of observed
  int<lower = 1> N_mis; // Number of incomplete
  int<lower=1> M; // Number of features
  int<lower=1> R; // Number of townships
  matrix[N_obs, M] X_obs; // matrix of features for observed outcomes
  matrix[N_mis, M] X_mis; // matrix of features for incomplete outcomes
  int S_obs[N_obs]; //vector of township for observed outcomes
  int S_mis[N_mis]; //vector of township for incomplete outcomes
  
  int<lower =0, upper = 1> partner_obs[N_obs];
  int<lower =0, upper = 1> active_obs[N_obs];
  int<lower =0, upper = 1> hiv_complete[N_obs];
  int<lower =0, upper = 1> hiv_mis[N_mis];
}

transformed data {
  real tau0 = 0.001;
  real slab_scale = 4;   
  real slab_scale2 = square(slab_scale);
  real slab_df = 20;    
  real half_slab_df = 0.5 * slab_df;
}

parameters {
  
  real<lower=0> c2_tilde;
  
  matrix<lower=0>[M, R] lambda_p;
  vector<lower=0>[R] tau_p_tilde;
  real alpha_p;
  matrix[M, R] beta_p_tilde;
  
  matrix<lower=0>[M, R] lambda_a;
  vector<lower=0>[R] tau_a_tilde;
  real alpha_a;
  matrix[M, R] beta_a_tilde;
  
  matrix<lower=0>[M, R] lambda2;
  vector<lower=0>[R] tau_tilde2;
  real alpha2;
  matrix[M, R] beta_tilde2;
  
  vector<lower=0>[R] lambda_mis;
  vector<lower=0>[R] tau_tilde_mis;
  vector[R] beta_tilde_mis;
  
  vector<lower=0>[R] lambda_mis2;
  vector<lower=0>[R] tau_tilde_mis2;
  vector[R] beta_tilde_mis2;
}

transformed parameters {
  matrix[M, R] beta_p;
  matrix[M, R] beta_a;
  matrix[M, R] beta2;
  vector[R] beta_mis;
  vector[R] beta_mis2;
  {
    vector[R] tau_p = tau0 * tau_p_tilde;
    vector[R] tau_a = tau0 * tau_a_tilde;
    vector[R] tau2 = tau0 * tau_tilde2;
    vector[R] tau_mis = tau0 * tau_tilde_mis;
    vector[R] tau_mis2 = tau0 * tau_tilde_mis2;
    
    real c2 = slab_scale2 * c2_tilde;
    
    matrix[M, R] lambda_tilde_p;
    matrix[M, R] lambda_tilde_a;
    matrix[M, R] lambda_tilde2;
    vector[R] lambda_tilde_mis;
    vector[R] lambda_tilde_mis2;
    
    for (r in 1:R)
      lambda_tilde_p[,r] = sqrt( c2 * square(lambda_p[,r]) ./ (c2 
                                                               + square(tau_p[r]) * square(lambda_p[,r])) );
    for (r in 1:R)
      lambda_tilde_a[,r] = sqrt( c2 * square(lambda_a[,r]) ./ (c2 
                                                               + square(tau_a[r]) * square(lambda_a[,r])) );
    for (r in 1:R)
      lambda_tilde2[,r] = sqrt( c2 * square(lambda2[,r]) ./ (c2 
                                                             + square(tau2[r]) * square(lambda2[,r])) );
    for (r in 1:R)
      lambda_tilde_mis[r] = sqrt( c2 * square(lambda_mis[r]) ./ (c2 
                                                                 + square(tau_mis[r]) * square(lambda_mis[r])) ); 
    for (r in 1:R)
      lambda_tilde_mis2[r] = sqrt( c2 * square(lambda_mis2[r]) ./ (c2 
                                                                   + square(tau_mis2[r]) * square(lambda_mis2[r])) ); 
    for (r in 1:R) beta_p[,r] = tau_p[r] * lambda_tilde_p[,r] .*
        beta_p_tilde[,r];
    for (r in 1:R) beta_a[,r] = tau_a[r] * lambda_tilde_a[,r] .*
        beta_a_tilde[,r];
    for (r in 1:R) beta2[,r] = tau2[r] * lambda_tilde2[,r] .*
        beta_tilde2[,r];
    for (r in 1:R) beta_mis[r] = tau_mis[r] * lambda_tilde_mis[r] .*
        beta_tilde_mis[r];
    for (r in 1:R) beta_mis2[r] = tau_mis2[r] * lambda_tilde_mis2[r] .* 
        beta_tilde_mis2[r];
  }
}
model {
  c2_tilde ~ inv_gamma(half_slab_df, half_slab_df);
  sex_last ~ bernoulli(994/1534);
  
  for (r in 1:R) beta_p_tilde[,r] ~ normal(0, 1);
  for (r in 1:R) lambda_p[,r] ~ cauchy(0, 1);
  tau_p_tilde ~ cauchy(0, 1);
  alpha_p ~ normal(0, 2);
  for (r in 1:R) beta_a_tilde[,r] ~ normal(0, 1);
  for (r in 1:R) lambda_a[,r] ~ cauchy(0, 1);
  tau_a_tilde ~ cauchy(0, 1);
  alpha_a ~ normal(0, 2);
  
  for (r in 1:R) beta_tilde2[,r] ~ normal(0, 1);
  for (r in 1:R) lambda2[,r] ~ cauchy(0, 1);
  tau_tilde2 ~ cauchy(0, 1);
  alpha2 ~ normal(0, 2);
  
  for (r in 1:R) beta_tilde_mis[r] ~ normal(0, 1);
  for (r in 1:R) lambda_mis[r] ~ cauchy(0, 1);
  tau_tilde_mis ~ cauchy(0, 1);
  
  for (r in 1:R) beta_tilde_mis2[r] ~ normal(0, 1);
  for (r in 1:R) lambda_mis2[r] ~ cauchy(0, 1);
  tau_tilde_mis2 ~ cauchy(0, 1);
  
  for (n in 1:(N_obs-1)) {
    partner_obs[n] ~ bernoulli_logit(X_obs[n,] * beta_p[,S_obs[n]] 
                                     + alpha_p);
    active_obs[n] ~ bernoulli_logit(X_obs[n,] * beta_a[,S_obs[n]] 
                                    + alpha_a);
    hiv_complete[n] ~ bernoulli_logit(X_obs[n,] * beta2[,S_obs[n]] 
                                      + alpha2 + partner_obs[n] * beta_mis[S_obs[n]] 
                                      + active_obs[n] * beta_mis2[S_obs[n]]);
    
  }
  partner_obs[N_obs] ~ bernoulli_logit(X_obs[N_obs,1:(M-1)] * beta_p[1:(M-1),S_obs[N_obs]] 
                                       + sex_last * beta_p[M,S_obs[N_obs]] + alpha_p);
  active_obs[N_obs] ~ bernoulli_logit(X_obs[N_obs,1:(M-1)] * beta_a[1:(M-1),S_obs[N_obs]] 
                                      + sex_last * beta_a[M,S_obs[N_obs]] + alpha_a);
  hiv_complete[N_obs] ~ bernoulli_logit(X_obs[N_obs,1:(M-1)] * beta2[1:(M-1),S_obs[N_obs]] 
                                        + sex_last * beta2[M,S_obs[N_obs]] + alpha2 
                                        + partner_obs[N_obs] * beta_mis[S_obs[N_obs]] 
                                        + active_obs[N_obs] * beta_mis2[S_obs[N_obs]]);
  
  for (n in 1:N_mis) {
    real partner_mis = X_mis[n,] * beta_p[,S_mis[n]] + alpha_p;
    real active_mis = X_mis[n,] * beta_a[,S_mis[n]] + alpha_a;
    target += log_mix(inv_logit(partner_mis), binomial_lpmf(1 | partner_mis), binomial_lpmf(0 | partner_mis));
    target += log_mix(inv_logit(active_mis), binomial_lpmf(1 | active_mis), binomial_lpmf(0 | active_mis));
    hiv_mis[n] ~ bernoulli_logit(inv_logit(partner_mis) * beta_mis[S_mis[n]] + inv_logit(active_mis) * 
         beta_mis2[S_mis[n]] + X_mis[n,] * beta2[,S_mis[n]] + alpha2);                                 
  }
}
"
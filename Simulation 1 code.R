### with laplace prior 

laplace <- "data { 
  int<lower=1> N; // Number of data rows
  int<lower=1> N_val; // Number of data rows
  int<lower=1> M; // Number of features
  matrix[N, M] X; //parameters 
  matrix[N, M] X_val; //parameters 
  int<lower =0, upper = 1> y[N];
  int<lower =0, upper = 1> y_val[N];
}

parameters {
  vector[M] beta;
  real alpha;
}

model {
  beta ~ double_exponential(0, 1);
  alpha ~ normal(0, 2);
  
  for (n in 1:N) {
    y[n] ~ bernoulli_logit(X[n,] * beta + alpha);
  }
}
generated quantities {
  vector[N] yhat_val;
  for (n in 1:N) {
    yhat_val[n] = inv_logit(X_val[n] * beta + alpha);
  }
} 
"

### Horseshoe prior 
hoseshoe <- "
data { 
  int<lower=1> N; // Number of data rows
  int<lower=1> N_val; // Number of data rows
  int<lower=1> M; // Number of features
  matrix[N, M] X; //parameters 
  matrix[N, M] X_val; //parameters 
  int<lower =0, upper = 1> y[N];
  int<lower =0, upper = 1> y_val[N];
}

parameters {
  vector[M] beta_tilde;
  vector<lower=0>[M] lambda;
  real<lower=0> tau_tilde;
  real alpha;
}

transformed parameters {
  vector[M] beta;
  
  beta = beta_tilde .* lambda * .25 * tau_tilde;
}
model {
  beta_tilde ~ normal(0, 1);
  lambda ~ cauchy(0, 1);
  tau_tilde ~ cauchy(0, 1);
  alpha ~ normal(0, 2);
  
  for (n in 1:N) {
    y[n] ~ bernoulli_logit(X[n,] * beta + alpha);
  }
}
generated quantities {
  vector[N] yhat_val;
  for (n in 1:N) {
    yhat_val[n] = inv_logit(X_val[n] * beta + alpha);
  }
}
"

regularized <- "
data { 
  int<lower=1> N; // Number of data rows
  int<lower=1> N_val; // Number of data rows
  int<lower=1> M; // Number of features
  matrix[N, M] X; //parameters 
  matrix[N, M] X_val; //parameters 
  int<lower =0, upper = 1> y[N];
  int<lower =0, upper = 1> y_val[N];
}

transformed data {
  real m0 = 10;           // Expected number of large slopes
  real slab_scale = 3;    // Scale for large slopes
  real slab_scale2 = square(slab_scale);
  real slab_df = 20;      // Effective degrees of freedom for large slopes
  real half_slab_df = 0.5 * slab_df;
}

parameters {
  vector[M] beta_tilde;
  vector<lower=0>[M] lambda;
  real<lower=0> c2_tilde;
  real tau_tilde;
  real alpha;
  real phi;
}

transformed parameters {
  vector[M] beta;
  {
    real tau0 = (m0 / (M - m0)) * (2 / sqrt(1.0 * N));
    real tau = tau0 * tau_tilde;
    
    real c2 = slab_scale2 * c2_tilde;
    
    vector[M] lambda_tilde;
    lambda_tilde = sqrt( c2 * square(lambda) ./ (c2 + square(tau) *
                                                   square(lambda)) );
    
    beta = tau * lambda_tilde .* beta_tilde;
  }
}
model {
  beta_tilde ~ normal(0, 1);
  lambda ~ cauchy(0, 1);
  tau_tilde ~ cauchy(0, 1);
  c2_tilde ~ inv_gamma(half_slab_df, half_slab_df);
  alpha ~ normal(0, 2);
  
  for (n in 1:N) {
    y[n] ~ bernoulli_logit(X[n,] * beta + alpha);
  }
}
generated quantities {
  vector[N] yhat_val;
  for (n in 1:N) {
    yhat_val[n] = inv_logit(X_val[n] * beta + alpha);
  }
}
"
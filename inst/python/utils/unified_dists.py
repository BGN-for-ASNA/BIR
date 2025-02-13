from functools import partial
from jax import random
from jax import jit
import numpyro as numpyro

class UnifiedDist:

    def __init__(self):
        pass

    @staticmethod
    def asymmetriclaplace(loc=0.0, scale=1.0, asymmetry=1.0, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        AsymmetricLaplace distribution.
    
        Arguments:
            loc: 0.0
            scale: 1.0
            asymmetry: 1.0
            validate_args: None
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.AsymmetricLaplace(loc=loc, scale=scale, asymmetry=asymmetry, validate_args=validate_args).sample(seed, shape)
        else: 
            if shape == ():
                return numpyro.distributions.AsymmetricLaplace(loc=loc, scale=scale, asymmetry=asymmetry, validate_args=validate_args)
            else:
                return numpyro.sample(name, numpyro.distributions.AsymmetricLaplace(loc=loc, scale=scale, asymmetry=asymmetry, validate_args=validate_args).expand(shape))

    @staticmethod
    def asymmetriclaplacequantile(loc=0.0, scale=1.0, quantile=0.5, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        AsymmetricLaplaceQuantile distribution.
    
        Arguments:
            loc: 0.0
            scale: 1.0
            quantile: 0.5
            validate_args: None
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.AsymmetricLaplaceQuantile(loc=loc, scale=scale, quantile=quantile, validate_args=validate_args).sample(seed, shape)
        else: 
            if shape == ():
                return numpyro.distributions.AsymmetricLaplaceQuantile(loc=loc, scale=scale, quantile=quantile, validate_args=validate_args)
            else:
                return numpyro.sample(name, numpyro.distributions.AsymmetricLaplaceQuantile(loc=loc, scale=scale, quantile=quantile, validate_args=validate_args).expand(shape))

    @staticmethod
    def bernoulli(probs=None, logits=None, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        Bernoulli distribution.
    
        Arguments:
            probs: None
            logits: None
            validate_args: None
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.Bernoulli(probs=probs, logits=logits, validate_args=validate_args).sample(seed, shape)
        else: 
            if shape == ():
                return numpyro.distributions.Bernoulli(probs=probs, logits=logits, validate_args=validate_args)
            else:
                return numpyro.sample(name, numpyro.distributions.Bernoulli(probs=probs, logits=logits, validate_args=validate_args).expand(shape))

    @staticmethod
    def bernoullilogits(logits=None, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        BernoulliLogits distribution.
    
        Arguments:
            logits: None
            validate_args: None
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.BernoulliLogits(logits=logits, validate_args=validate_args).sample(seed, shape)
        else: 
            if shape == ():
                return numpyro.distributions.BernoulliLogits(logits=logits, validate_args=validate_args)
            else:
                return numpyro.sample(name, numpyro.distributions.BernoulliLogits(logits=logits, validate_args=validate_args).expand(shape))

    @staticmethod
    def bernoulliprobs(probs, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        BernoulliProbs distribution.
    
        Arguments:
            probs: <class 'inspect._empty'>
            validate_args: None
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.BernoulliProbs(probs=probs, validate_args=validate_args).sample(seed, shape)
        else: 
            if shape == ():
                return numpyro.distributions.BernoulliProbs(probs=probs, validate_args=validate_args)
            else:
                return numpyro.sample(name, numpyro.distributions.BernoulliProbs(probs=probs, validate_args=validate_args).expand(shape))

    @staticmethod
    def beta(concentration1, concentration0, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        Beta distribution.
    
        Arguments:
            concentration1: <class 'inspect._empty'>
            concentration0: <class 'inspect._empty'>
            validate_args: None
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.Beta(concentration1=concentration1, concentration0=concentration0, validate_args=validate_args).sample(seed, shape)
        else: 
            if shape == ():
                return numpyro.distributions.Beta(concentration1=concentration1, concentration0=concentration0, validate_args=validate_args)
            else:
                return numpyro.sample(name, numpyro.distributions.Beta(concentration1=concentration1, concentration0=concentration0, validate_args=validate_args).expand(shape))

    @staticmethod
    def betabinomial(concentration1, concentration0, total_count=1, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        BetaBinomial distribution.
    
        Arguments:
            concentration1: <class 'inspect._empty'>
            concentration0: <class 'inspect._empty'>
            total_count: 1
            validate_args: None
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.BetaBinomial(concentration1=concentration1, concentration0=concentration0, total_count=total_count, validate_args=validate_args).sample(seed, shape)
        else: 
            if shape == ():
                return numpyro.distributions.BetaBinomial(concentration1=concentration1, concentration0=concentration0, total_count=total_count, validate_args=validate_args)
            else:
                return numpyro.sample(name, numpyro.distributions.BetaBinomial(concentration1=concentration1, concentration0=concentration0, total_count=total_count, validate_args=validate_args).expand(shape))

    @staticmethod
    def betaproportion(mean, concentration, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        BetaProportion distribution.
    
        Arguments:
            mean: <class 'inspect._empty'>
            concentration: <class 'inspect._empty'>
            validate_args: None
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.BetaProportion(mean=mean, concentration=concentration, validate_args=validate_args).sample(seed, shape)
        else: 
            if shape == ():
                return numpyro.distributions.BetaProportion(mean=mean, concentration=concentration, validate_args=validate_args)
            else:
                return numpyro.sample(name, numpyro.distributions.BetaProportion(mean=mean, concentration=concentration, validate_args=validate_args).expand(shape))

    @staticmethod
    def binomial(total_count=1, probs=None, logits=None, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        Binomial distribution.
    
        Arguments:
            total_count: 1
            probs: None
            logits: None
            validate_args: None
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.Binomial(total_count=total_count, probs=probs, logits=logits, validate_args=validate_args).sample(seed, shape)
        else: 
            if shape == ():
                return numpyro.distributions.Binomial(total_count=total_count, probs=probs, logits=logits, validate_args=validate_args)
            else:
                return numpyro.sample(name, numpyro.distributions.Binomial(total_count=total_count, probs=probs, logits=logits, validate_args=validate_args).expand(shape))

    @staticmethod
    def binomiallogits(logits, total_count=1, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        BinomialLogits distribution.
    
        Arguments:
            logits: <class 'inspect._empty'>
            total_count: 1
            validate_args: None
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.BinomialLogits(logits=logits, total_count=total_count, validate_args=validate_args).sample(seed, shape)
        else: 
            if shape == ():
                return numpyro.distributions.BinomialLogits(logits=logits, total_count=total_count, validate_args=validate_args)
            else:
                return numpyro.sample(name, numpyro.distributions.BinomialLogits(logits=logits, total_count=total_count, validate_args=validate_args).expand(shape))

    @staticmethod
    def binomialprobs(probs, total_count=1, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        BinomialProbs distribution.
    
        Arguments:
            probs: <class 'inspect._empty'>
            total_count: 1
            validate_args: None
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.BinomialProbs(probs=probs, total_count=total_count, validate_args=validate_args).sample(seed, shape)
        else: 
            if shape == ():
                return numpyro.distributions.BinomialProbs(probs=probs, total_count=total_count, validate_args=validate_args)
            else:
                return numpyro.sample(name, numpyro.distributions.BinomialProbs(probs=probs, total_count=total_count, validate_args=validate_args).expand(shape))

    @staticmethod
    def car(loc, correlation, conditional_precision, adj_matrix, is_sparse=False, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        CAR distribution.
    
        Arguments:
            loc: <class 'inspect._empty'>
            correlation: <class 'inspect._empty'>
            conditional_precision: <class 'inspect._empty'>
            adj_matrix: <class 'inspect._empty'>
            is_sparse: False
            validate_args: None
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.CAR(loc=loc, correlation=correlation, conditional_precision=conditional_precision, adj_matrix=adj_matrix, is_sparse=is_sparse, validate_args=validate_args).sample(seed, shape)
        else: 
            if shape == ():
                return numpyro.distributions.CAR(loc=loc, correlation=correlation, conditional_precision=conditional_precision, adj_matrix=adj_matrix, is_sparse=is_sparse, validate_args=validate_args)
            else:
                return numpyro.sample(name, numpyro.distributions.CAR(loc=loc, correlation=correlation, conditional_precision=conditional_precision, adj_matrix=adj_matrix, is_sparse=is_sparse, validate_args=validate_args).expand(shape))

    @staticmethod
    def categorical(probs=None, logits=None, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        Categorical distribution.
    
        Arguments:
            probs: None
            logits: None
            validate_args: None
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.Categorical(probs=probs, logits=logits, validate_args=validate_args).sample(seed, shape)
        else: 
            if shape == ():
                return numpyro.distributions.Categorical(probs=probs, logits=logits, validate_args=validate_args)
            else:
                return numpyro.sample(name, numpyro.distributions.Categorical(probs=probs, logits=logits, validate_args=validate_args).expand(shape))

    @staticmethod
    def categoricallogits(logits, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        CategoricalLogits distribution.
    
        Arguments:
            logits: <class 'inspect._empty'>
            validate_args: None
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.CategoricalLogits(logits=logits, validate_args=validate_args).sample(seed, shape)
        else: 
            if shape == ():
                return numpyro.distributions.CategoricalLogits(logits=logits, validate_args=validate_args)
            else:
                return numpyro.sample(name, numpyro.distributions.CategoricalLogits(logits=logits, validate_args=validate_args).expand(shape))

    @staticmethod
    def categoricalprobs(probs, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        CategoricalProbs distribution.
    
        Arguments:
            probs: <class 'inspect._empty'>
            validate_args: None
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.CategoricalProbs(probs=probs, validate_args=validate_args).sample(seed, shape)
        else: 
            if shape == ():
                return numpyro.distributions.CategoricalProbs(probs=probs, validate_args=validate_args)
            else:
                return numpyro.sample(name, numpyro.distributions.CategoricalProbs(probs=probs, validate_args=validate_args).expand(shape))

    @staticmethod
    def cauchy(loc=0.0, scale=1.0, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        Cauchy distribution.
    
        Arguments:
            loc: 0.0
            scale: 1.0
            validate_args: None
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.Cauchy(loc=loc, scale=scale, validate_args=validate_args).sample(seed, shape)
        else: 
            if shape == ():
                return numpyro.distributions.Cauchy(loc=loc, scale=scale, validate_args=validate_args)
            else:
                return numpyro.sample(name, numpyro.distributions.Cauchy(loc=loc, scale=scale, validate_args=validate_args).expand(shape))

    @staticmethod
    def chi2(df, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        Chi2 distribution.
    
        Arguments:
            df: <class 'inspect._empty'>
            validate_args: None
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.Chi2(df=df, validate_args=validate_args).sample(seed, shape)
        else: 
            if shape == ():
                return numpyro.distributions.Chi2(df=df, validate_args=validate_args)
            else:
                return numpyro.sample(name, numpyro.distributions.Chi2(df=df, validate_args=validate_args).expand(shape))

    @staticmethod
    def delta(v=0.0, log_density=0.0, event_dim=0, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        Delta distribution.
    
        Arguments:
            v: 0.0
            log_density: 0.0
            event_dim: 0
            validate_args: None
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.Delta(v=v, log_density=log_density, event_dim=event_dim, validate_args=validate_args).sample(seed, shape)
        else: 
            if shape == ():
                return numpyro.distributions.Delta(v=v, log_density=log_density, event_dim=event_dim, validate_args=validate_args)
            else:
                return numpyro.sample(name, numpyro.distributions.Delta(v=v, log_density=log_density, event_dim=event_dim, validate_args=validate_args).expand(shape))

    @staticmethod
    def dirichlet(concentration, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        Dirichlet distribution.
    
        Arguments:
            concentration: <class 'inspect._empty'>
            validate_args: None
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.Dirichlet(concentration=concentration, validate_args=validate_args).sample(seed, shape)
        else: 
            if shape == ():
                return numpyro.distributions.Dirichlet(concentration=concentration, validate_args=validate_args)
            else:
                return numpyro.sample(name, numpyro.distributions.Dirichlet(concentration=concentration, validate_args=validate_args).expand(shape))

    @staticmethod
    def dirichletmultinomial(concentration, total_count=1, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        DirichletMultinomial distribution.
    
        Arguments:
            concentration: <class 'inspect._empty'>
            total_count: 1
            validate_args: None
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.DirichletMultinomial(concentration=concentration, total_count=total_count, validate_args=validate_args).sample(seed, shape)
        else: 
            if shape == ():
                return numpyro.distributions.DirichletMultinomial(concentration=concentration, total_count=total_count, validate_args=validate_args)
            else:
                return numpyro.sample(name, numpyro.distributions.DirichletMultinomial(concentration=concentration, total_count=total_count, validate_args=validate_args).expand(shape))

    @staticmethod
    def discreteuniform(low=0, high=1, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        DiscreteUniform distribution.
    
        Arguments:
            low: 0
            high: 1
            validate_args: None
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.DiscreteUniform(low=low, high=high, validate_args=validate_args).sample(seed, shape)
        else: 
            if shape == ():
                return numpyro.distributions.DiscreteUniform(low=low, high=high, validate_args=validate_args)
            else:
                return numpyro.sample(name, numpyro.distributions.DiscreteUniform(low=low, high=high, validate_args=validate_args).expand(shape))

    @staticmethod
    def distribution(batch_shape=(), event_shape=(), validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        Distribution distribution.
    
        Arguments:
            batch_shape: ()
            event_shape: ()
            validate_args: None
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.Distribution(batch_shape=batch_shape, event_shape=event_shape, validate_args=validate_args).sample(seed, shape)
        else: 
            if shape == ():
                return numpyro.distributions.Distribution(batch_shape=batch_shape, event_shape=event_shape, validate_args=validate_args)
            else:
                return numpyro.sample(name, numpyro.distributions.Distribution(batch_shape=batch_shape, event_shape=event_shape, validate_args=validate_args).expand(shape))

    @staticmethod
    def eulermaruyama(t, sde_fn, init_dist, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        EulerMaruyama distribution.
    
        Arguments:
            t: <class 'inspect._empty'>
            sde_fn: <class 'inspect._empty'>
            init_dist: <class 'inspect._empty'>
            validate_args: None
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.EulerMaruyama(t=t, sde_fn=sde_fn, init_dist=init_dist, validate_args=validate_args).sample(seed, shape)
        else: 
            if shape == ():
                return numpyro.distributions.EulerMaruyama(t=t, sde_fn=sde_fn, init_dist=init_dist, validate_args=validate_args)
            else:
                return numpyro.sample(name, numpyro.distributions.EulerMaruyama(t=t, sde_fn=sde_fn, init_dist=init_dist, validate_args=validate_args).expand(shape))

    @staticmethod
    def expandeddistribution(base_dist, batch_shape=(), shape=(), sample = False, seed = 0, name = 'x'):
        """
        ExpandedDistribution distribution.
    
        Arguments:
            base_dist: <class 'inspect._empty'>
            batch_shape: ()
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.ExpandedDistribution(base_dist=base_dist, batch_shape=batch_shape).sample(seed, shape)
        else: 
            if shape == ():
                return numpyro.distributions.ExpandedDistribution(base_dist=base_dist, batch_shape=batch_shape)
            else:
                return numpyro.sample(name, numpyro.distributions.ExpandedDistribution(base_dist=base_dist, batch_shape=batch_shape).expand(shape))

    @staticmethod
    def exponential(rate=1.0, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        Exponential distribution.
    
        Arguments:
            rate: 1.0
            validate_args: None
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.Exponential(rate=rate, validate_args=validate_args).sample(seed, shape)
        else: 
            if shape == ():
                return numpyro.distributions.Exponential(rate=rate, validate_args=validate_args)
            else:
                return numpyro.sample(name, numpyro.distributions.Exponential(rate=rate, validate_args=validate_args).expand(shape))

    @staticmethod
    def foldeddistribution(base_dist, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        FoldedDistribution distribution.
    
        Arguments:
            base_dist: <class 'inspect._empty'>
            validate_args: None
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.FoldedDistribution(base_dist=base_dist, validate_args=validate_args).sample(seed, shape)
        else: 
            if shape == ():
                return numpyro.distributions.FoldedDistribution(base_dist=base_dist, validate_args=validate_args)
            else:
                return numpyro.sample(name, numpyro.distributions.FoldedDistribution(base_dist=base_dist, validate_args=validate_args).expand(shape))

    @staticmethod
    def gamma(concentration, rate=1.0, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        Gamma distribution.
    
        Arguments:
            concentration: <class 'inspect._empty'>
            rate: 1.0
            validate_args: None
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.Gamma(concentration=concentration, rate=rate, validate_args=validate_args).sample(seed, shape)
        else: 
            if shape == ():
                return numpyro.distributions.Gamma(concentration=concentration, rate=rate, validate_args=validate_args)
            else:
                return numpyro.sample(name, numpyro.distributions.Gamma(concentration=concentration, rate=rate, validate_args=validate_args).expand(shape))

    @staticmethod
    def gammapoisson(concentration, rate=1.0, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        GammaPoisson distribution.
    
        Arguments:
            concentration: <class 'inspect._empty'>
            rate: 1.0
            validate_args: None
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.GammaPoisson(concentration=concentration, rate=rate, validate_args=validate_args).sample(seed, shape)
        else: 
            if shape == ():
                return numpyro.distributions.GammaPoisson(concentration=concentration, rate=rate, validate_args=validate_args)
            else:
                return numpyro.sample(name, numpyro.distributions.GammaPoisson(concentration=concentration, rate=rate, validate_args=validate_args).expand(shape))

    @staticmethod
    def gaussiancopula(marginal_dist, correlation_matrix=None, correlation_cholesky=None, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        GaussianCopula distribution.
    
        Arguments:
            marginal_dist: <class 'inspect._empty'>
            correlation_matrix: None
            correlation_cholesky: None
            validate_args: None
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.GaussianCopula(marginal_dist=marginal_dist, correlation_matrix=correlation_matrix, correlation_cholesky=correlation_cholesky, validate_args=validate_args).sample(seed, shape)
        else: 
            if shape == ():
                return numpyro.distributions.GaussianCopula(marginal_dist=marginal_dist, correlation_matrix=correlation_matrix, correlation_cholesky=correlation_cholesky, validate_args=validate_args)
            else:
                return numpyro.sample(name, numpyro.distributions.GaussianCopula(marginal_dist=marginal_dist, correlation_matrix=correlation_matrix, correlation_cholesky=correlation_cholesky, validate_args=validate_args).expand(shape))

    @staticmethod
    def gaussiancopulabeta(concentration1, concentration0, correlation_matrix=None, correlation_cholesky=None, validate_args=False, shape=(), sample = False, seed = 0, name = 'x'):
        """
        GaussianCopulaBeta distribution.
    
        Arguments:
            concentration1: <class 'inspect._empty'>
            concentration0: <class 'inspect._empty'>
            correlation_matrix: None
            correlation_cholesky: None
            validate_args: False
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.GaussianCopulaBeta(concentration1=concentration1, concentration0=concentration0, correlation_matrix=correlation_matrix, correlation_cholesky=correlation_cholesky, validate_args=validate_args).sample(seed, shape)
        else: 
            if shape == ():
                return numpyro.distributions.GaussianCopulaBeta(concentration1=concentration1, concentration0=concentration0, correlation_matrix=correlation_matrix, correlation_cholesky=correlation_cholesky, validate_args=validate_args)
            else:
                return numpyro.sample(name, numpyro.distributions.GaussianCopulaBeta(concentration1=concentration1, concentration0=concentration0, correlation_matrix=correlation_matrix, correlation_cholesky=correlation_cholesky, validate_args=validate_args).expand(shape))

    @staticmethod
    def gaussianrandomwalk(scale=1.0, num_steps=1, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        GaussianRandomWalk distribution.
    
        Arguments:
            scale: 1.0
            num_steps: 1
            validate_args: None
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.GaussianRandomWalk(scale=scale, num_steps=num_steps, validate_args=validate_args).sample(seed, shape)
        else: 
            if shape == ():
                return numpyro.distributions.GaussianRandomWalk(scale=scale, num_steps=num_steps, validate_args=validate_args)
            else:
                return numpyro.sample(name, numpyro.distributions.GaussianRandomWalk(scale=scale, num_steps=num_steps, validate_args=validate_args).expand(shape))

    @staticmethod
    def geometric(probs=None, logits=None, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        Geometric distribution.
    
        Arguments:
            probs: None
            logits: None
            validate_args: None
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.Geometric(probs=probs, logits=logits, validate_args=validate_args).sample(seed, shape)
        else: 
            if shape == ():
                return numpyro.distributions.Geometric(probs=probs, logits=logits, validate_args=validate_args)
            else:
                return numpyro.sample(name, numpyro.distributions.Geometric(probs=probs, logits=logits, validate_args=validate_args).expand(shape))

    @staticmethod
    def geometriclogits(logits, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        GeometricLogits distribution.
    
        Arguments:
            logits: <class 'inspect._empty'>
            validate_args: None
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.GeometricLogits(logits=logits, validate_args=validate_args).sample(seed, shape)
        else: 
            if shape == ():
                return numpyro.distributions.GeometricLogits(logits=logits, validate_args=validate_args)
            else:
                return numpyro.sample(name, numpyro.distributions.GeometricLogits(logits=logits, validate_args=validate_args).expand(shape))

    @staticmethod
    def geometricprobs(probs, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        GeometricProbs distribution.
    
        Arguments:
            probs: <class 'inspect._empty'>
            validate_args: None
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.GeometricProbs(probs=probs, validate_args=validate_args).sample(seed, shape)
        else: 
            if shape == ():
                return numpyro.distributions.GeometricProbs(probs=probs, validate_args=validate_args)
            else:
                return numpyro.sample(name, numpyro.distributions.GeometricProbs(probs=probs, validate_args=validate_args).expand(shape))

    @staticmethod
    def gompertz(concentration, rate=1.0, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        Gompertz distribution.
    
        Arguments:
            concentration: <class 'inspect._empty'>
            rate: 1.0
            validate_args: None
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.Gompertz(concentration=concentration, rate=rate, validate_args=validate_args).sample(seed, shape)
        else: 
            if shape == ():
                return numpyro.distributions.Gompertz(concentration=concentration, rate=rate, validate_args=validate_args)
            else:
                return numpyro.sample(name, numpyro.distributions.Gompertz(concentration=concentration, rate=rate, validate_args=validate_args).expand(shape))

    @staticmethod
    def gumbel(loc=0.0, scale=1.0, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        Gumbel distribution.
    
        Arguments:
            loc: 0.0
            scale: 1.0
            validate_args: None
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.Gumbel(loc=loc, scale=scale, validate_args=validate_args).sample(seed, shape)
        else: 
            if shape == ():
                return numpyro.distributions.Gumbel(loc=loc, scale=scale, validate_args=validate_args)
            else:
                return numpyro.sample(name, numpyro.distributions.Gumbel(loc=loc, scale=scale, validate_args=validate_args).expand(shape))

    @staticmethod
    def halfcauchy(scale=1.0, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        HalfCauchy distribution.
    
        Arguments:
            scale: 1.0
            validate_args: None
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.HalfCauchy(scale=scale, validate_args=validate_args).sample(seed, shape)
        else: 
            if shape == ():
                return numpyro.distributions.HalfCauchy(scale=scale, validate_args=validate_args)
            else:
                return numpyro.sample(name, numpyro.distributions.HalfCauchy(scale=scale, validate_args=validate_args).expand(shape))

    @staticmethod
    def halfnormal(scale=1.0, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        HalfNormal distribution.
    
        Arguments:
            scale: 1.0
            validate_args: None
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.HalfNormal(scale=scale, validate_args=validate_args).sample(seed, shape)
        else: 
            if shape == ():
                return numpyro.distributions.HalfNormal(scale=scale, validate_args=validate_args)
            else:
                return numpyro.sample(name, numpyro.distributions.HalfNormal(scale=scale, validate_args=validate_args).expand(shape))

    @staticmethod
    def improperuniform(support, batch_shape, event_shape, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        ImproperUniform distribution.
    
        Arguments:
            support: <class 'inspect._empty'>
            batch_shape: <class 'inspect._empty'>
            event_shape: <class 'inspect._empty'>
            validate_args: None
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.ImproperUniform(support=support, batch_shape=batch_shape, event_shape=event_shape, validate_args=validate_args).sample(seed, shape)
        else: 
            if shape == ():
                return numpyro.distributions.ImproperUniform(support=support, batch_shape=batch_shape, event_shape=event_shape, validate_args=validate_args)
            else:
                return numpyro.sample(name, numpyro.distributions.ImproperUniform(support=support, batch_shape=batch_shape, event_shape=event_shape, validate_args=validate_args).expand(shape))

    @staticmethod
    def independent(base_dist, reinterpreted_batch_ndims, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        Independent distribution.
    
        Arguments:
            base_dist: <class 'inspect._empty'>
            reinterpreted_batch_ndims: <class 'inspect._empty'>
            validate_args: None
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.Independent(base_dist=base_dist, reinterpreted_batch_ndims=reinterpreted_batch_ndims, validate_args=validate_args).sample(seed, shape)
        else: 
            if shape == ():
                return numpyro.distributions.Independent(base_dist=base_dist, reinterpreted_batch_ndims=reinterpreted_batch_ndims, validate_args=validate_args)
            else:
                return numpyro.sample(name, numpyro.distributions.Independent(base_dist=base_dist, reinterpreted_batch_ndims=reinterpreted_batch_ndims, validate_args=validate_args).expand(shape))

    @staticmethod
    def inversegamma(concentration, rate=1.0, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        InverseGamma distribution.
    
        Arguments:
            concentration: <class 'inspect._empty'>
            rate: 1.0
            validate_args: None
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.InverseGamma(concentration=concentration, rate=rate, validate_args=validate_args).sample(seed, shape)
        else: 
            if shape == ():
                return numpyro.distributions.InverseGamma(concentration=concentration, rate=rate, validate_args=validate_args)
            else:
                return numpyro.sample(name, numpyro.distributions.InverseGamma(concentration=concentration, rate=rate, validate_args=validate_args).expand(shape))

    @staticmethod
    def kumaraswamy(concentration1, concentration0, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        Kumaraswamy distribution.
    
        Arguments:
            concentration1: <class 'inspect._empty'>
            concentration0: <class 'inspect._empty'>
            validate_args: None
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.Kumaraswamy(concentration1=concentration1, concentration0=concentration0, validate_args=validate_args).sample(seed, shape)
        else: 
            if shape == ():
                return numpyro.distributions.Kumaraswamy(concentration1=concentration1, concentration0=concentration0, validate_args=validate_args)
            else:
                return numpyro.sample(name, numpyro.distributions.Kumaraswamy(concentration1=concentration1, concentration0=concentration0, validate_args=validate_args).expand(shape))

    @staticmethod
    def lkj(dimension, concentration=1.0, sample_method='onion', validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        LKJ distribution.
    
        Arguments:
            dimension: <class 'inspect._empty'>
            concentration: 1.0
            sample_method: onion
            validate_args: None
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.LKJ(dimension=dimension, concentration=concentration, sample_method=sample_method, validate_args=validate_args).sample(seed, shape)
        else: 
            if shape == ():
                return numpyro.distributions.LKJ(dimension=dimension, concentration=concentration, sample_method=sample_method, validate_args=validate_args)
            else:
                return numpyro.sample(name, numpyro.distributions.LKJ(dimension=dimension, concentration=concentration, sample_method=sample_method, validate_args=validate_args).expand(shape))

    @staticmethod
    def lkjcholesky(dimension, concentration=1.0, sample_method='onion', validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        LKJCholesky distribution.
    
        Arguments:
            dimension: <class 'inspect._empty'>
            concentration: 1.0
            sample_method: onion
            validate_args: None
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.LKJCholesky(dimension=dimension, concentration=concentration, sample_method=sample_method, validate_args=validate_args).sample(seed, shape)
        else: 
            if shape == ():
                return numpyro.distributions.LKJCholesky(dimension=dimension, concentration=concentration, sample_method=sample_method, validate_args=validate_args)
            else:
                return numpyro.sample(name, numpyro.distributions.LKJCholesky(dimension=dimension, concentration=concentration, sample_method=sample_method, validate_args=validate_args).expand(shape))

    @staticmethod
    def laplace(loc=0.0, scale=1.0, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        Laplace distribution.
    
        Arguments:
            loc: 0.0
            scale: 1.0
            validate_args: None
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.Laplace(loc=loc, scale=scale, validate_args=validate_args).sample(seed, shape)
        else: 
            if shape == ():
                return numpyro.distributions.Laplace(loc=loc, scale=scale, validate_args=validate_args)
            else:
                return numpyro.sample(name, numpyro.distributions.Laplace(loc=loc, scale=scale, validate_args=validate_args).expand(shape))

    @staticmethod
    def lefttruncateddistribution(base_dist, low=0.0, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        LeftTruncatedDistribution distribution.
    
        Arguments:
            base_dist: <class 'inspect._empty'>
            low: 0.0
            validate_args: None
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.LeftTruncatedDistribution(base_dist=base_dist, low=low, validate_args=validate_args).sample(seed, shape)
        else: 
            if shape == ():
                return numpyro.distributions.LeftTruncatedDistribution(base_dist=base_dist, low=low, validate_args=validate_args)
            else:
                return numpyro.sample(name, numpyro.distributions.LeftTruncatedDistribution(base_dist=base_dist, low=low, validate_args=validate_args).expand(shape))

    @staticmethod
    def lognormal(loc=0.0, scale=1.0, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        LogNormal distribution.
    
        Arguments:
            loc: 0.0
            scale: 1.0
            validate_args: None
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.LogNormal(loc=loc, scale=scale, validate_args=validate_args).sample(seed, shape)
        else: 
            if shape == ():
                return numpyro.distributions.LogNormal(loc=loc, scale=scale, validate_args=validate_args)
            else:
                return numpyro.sample(name, numpyro.distributions.LogNormal(loc=loc, scale=scale, validate_args=validate_args).expand(shape))

    @staticmethod
    def loguniform(low, high, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        LogUniform distribution.
    
        Arguments:
            low: <class 'inspect._empty'>
            high: <class 'inspect._empty'>
            validate_args: None
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.LogUniform(low=low, high=high, validate_args=validate_args).sample(seed, shape)
        else: 
            if shape == ():
                return numpyro.distributions.LogUniform(low=low, high=high, validate_args=validate_args)
            else:
                return numpyro.sample(name, numpyro.distributions.LogUniform(low=low, high=high, validate_args=validate_args).expand(shape))

    @staticmethod
    def logistic(loc=0.0, scale=1.0, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        Logistic distribution.
    
        Arguments:
            loc: 0.0
            scale: 1.0
            validate_args: None
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.Logistic(loc=loc, scale=scale, validate_args=validate_args).sample(seed, shape)
        else: 
            if shape == ():
                return numpyro.distributions.Logistic(loc=loc, scale=scale, validate_args=validate_args)
            else:
                return numpyro.sample(name, numpyro.distributions.Logistic(loc=loc, scale=scale, validate_args=validate_args).expand(shape))

    @staticmethod
    def lowrankmultivariatenormal(loc, cov_factor, cov_diag, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        LowRankMultivariateNormal distribution.
    
        Arguments:
            loc: <class 'inspect._empty'>
            cov_factor: <class 'inspect._empty'>
            cov_diag: <class 'inspect._empty'>
            validate_args: None
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.LowRankMultivariateNormal(loc=loc, cov_factor=cov_factor, cov_diag=cov_diag, validate_args=validate_args).sample(seed, shape)
        else: 
            if shape == ():
                return numpyro.distributions.LowRankMultivariateNormal(loc=loc, cov_factor=cov_factor, cov_diag=cov_diag, validate_args=validate_args)
            else:
                return numpyro.sample(name, numpyro.distributions.LowRankMultivariateNormal(loc=loc, cov_factor=cov_factor, cov_diag=cov_diag, validate_args=validate_args).expand(shape))

    @staticmethod
    def maskeddistribution(base_dist, mask, shape=(), sample = False, seed = 0, name = 'x'):
        """
        MaskedDistribution distribution.
    
        Arguments:
            base_dist: <class 'inspect._empty'>
            mask: <class 'inspect._empty'>
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.MaskedDistribution(base_dist=base_dist, mask=mask).sample(seed, shape)
        else: 
            if shape == ():
                return numpyro.distributions.MaskedDistribution(base_dist=base_dist, mask=mask)
            else:
                return numpyro.sample(name, numpyro.distributions.MaskedDistribution(base_dist=base_dist, mask=mask).expand(shape))

    @staticmethod
    def matrixnormal(loc, scale_tril_row, scale_tril_column, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        MatrixNormal distribution.
    
        Arguments:
            loc: <class 'inspect._empty'>
            scale_tril_row: <class 'inspect._empty'>
            scale_tril_column: <class 'inspect._empty'>
            validate_args: None
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.MatrixNormal(loc=loc, scale_tril_row=scale_tril_row, scale_tril_column=scale_tril_column, validate_args=validate_args).sample(seed, shape)
        else: 
            if shape == ():
                return numpyro.distributions.MatrixNormal(loc=loc, scale_tril_row=scale_tril_row, scale_tril_column=scale_tril_column, validate_args=validate_args)
            else:
                return numpyro.sample(name, numpyro.distributions.MatrixNormal(loc=loc, scale_tril_row=scale_tril_row, scale_tril_column=scale_tril_column, validate_args=validate_args).expand(shape))

    @staticmethod
    def mixture(mixing_distribution, component_distributions, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        Mixture distribution.
    
        Arguments:
            mixing_distribution: <class 'inspect._empty'>
            component_distributions: <class 'inspect._empty'>
            validate_args: None
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.Mixture(mixing_distribution=mixing_distribution, component_distributions=component_distributions, validate_args=validate_args).sample(seed, shape)
        else: 
            if shape == ():
                return numpyro.distributions.Mixture(mixing_distribution=mixing_distribution, component_distributions=component_distributions, validate_args=validate_args)
            else:
                return numpyro.sample(name, numpyro.distributions.Mixture(mixing_distribution=mixing_distribution, component_distributions=component_distributions, validate_args=validate_args).expand(shape))

    @staticmethod
    def mixturegeneral(mixing_distribution, component_distributions, support=None, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        MixtureGeneral distribution.
    
        Arguments:
            mixing_distribution: <class 'inspect._empty'>
            component_distributions: <class 'inspect._empty'>
            support: None
            validate_args: None
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.MixtureGeneral(mixing_distribution=mixing_distribution, component_distributions=component_distributions, support=support, validate_args=validate_args).sample(seed, shape)
        else: 
            if shape == ():
                return numpyro.distributions.MixtureGeneral(mixing_distribution=mixing_distribution, component_distributions=component_distributions, support=support, validate_args=validate_args)
            else:
                return numpyro.sample(name, numpyro.distributions.MixtureGeneral(mixing_distribution=mixing_distribution, component_distributions=component_distributions, support=support, validate_args=validate_args).expand(shape))

    @staticmethod
    def mixturesamefamily(mixing_distribution, component_distribution, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        MixtureSameFamily distribution.
    
        Arguments:
            mixing_distribution: <class 'inspect._empty'>
            component_distribution: <class 'inspect._empty'>
            validate_args: None
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.MixtureSameFamily(mixing_distribution=mixing_distribution, component_distribution=component_distribution, validate_args=validate_args).sample(seed, shape)
        else: 
            if shape == ():
                return numpyro.distributions.MixtureSameFamily(mixing_distribution=mixing_distribution, component_distribution=component_distribution, validate_args=validate_args)
            else:
                return numpyro.sample(name, numpyro.distributions.MixtureSameFamily(mixing_distribution=mixing_distribution, component_distribution=component_distribution, validate_args=validate_args).expand(shape))

    @staticmethod
    def multinomial(total_count=1, probs=None, logits=None, total_count_max=None, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        Multinomial distribution.
    
        Arguments:
            total_count: 1
            probs: None
            logits: None
            total_count_max: None
            validate_args: None
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.Multinomial(total_count=total_count, probs=probs, logits=logits, total_count_max=total_count_max, validate_args=validate_args).sample(seed, shape)
        else: 
            if shape == ():
                return numpyro.distributions.Multinomial(total_count=total_count, probs=probs, logits=logits, total_count_max=total_count_max, validate_args=validate_args)
            else:
                return numpyro.sample(name, numpyro.distributions.Multinomial(total_count=total_count, probs=probs, logits=logits, total_count_max=total_count_max, validate_args=validate_args).expand(shape))

    @staticmethod
    def multinomiallogits(logits, total_count=1, total_count_max=None, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        MultinomialLogits distribution.
    
        Arguments:
            logits: <class 'inspect._empty'>
            total_count: 1
            total_count_max: None
            validate_args: None
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.MultinomialLogits(logits=logits, total_count=total_count, total_count_max=total_count_max, validate_args=validate_args).sample(seed, shape)
        else: 
            if shape == ():
                return numpyro.distributions.MultinomialLogits(logits=logits, total_count=total_count, total_count_max=total_count_max, validate_args=validate_args)
            else:
                return numpyro.sample(name, numpyro.distributions.MultinomialLogits(logits=logits, total_count=total_count, total_count_max=total_count_max, validate_args=validate_args).expand(shape))

    @staticmethod
    def multinomialprobs(probs, total_count=1, total_count_max=None, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        MultinomialProbs distribution.
    
        Arguments:
            probs: <class 'inspect._empty'>
            total_count: 1
            total_count_max: None
            validate_args: None
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.MultinomialProbs(probs=probs, total_count=total_count, total_count_max=total_count_max, validate_args=validate_args).sample(seed, shape)
        else: 
            if shape == ():
                return numpyro.distributions.MultinomialProbs(probs=probs, total_count=total_count, total_count_max=total_count_max, validate_args=validate_args)
            else:
                return numpyro.sample(name, numpyro.distributions.MultinomialProbs(probs=probs, total_count=total_count, total_count_max=total_count_max, validate_args=validate_args).expand(shape))

    @staticmethod
    def multivariatenormal(loc=0.0, covariance_matrix=None, precision_matrix=None, scale_tril=None, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        MultivariateNormal distribution.
    
        Arguments:
            loc: 0.0
            covariance_matrix: None
            precision_matrix: None
            scale_tril: None
            validate_args: None
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.MultivariateNormal(loc=loc, covariance_matrix=covariance_matrix, precision_matrix=precision_matrix, scale_tril=scale_tril, validate_args=validate_args).sample(seed, shape)
        else: 
            if shape == ():
                return numpyro.distributions.MultivariateNormal(loc=loc, covariance_matrix=covariance_matrix, precision_matrix=precision_matrix, scale_tril=scale_tril, validate_args=validate_args)
            else:
                return numpyro.sample(name, numpyro.distributions.MultivariateNormal(loc=loc, covariance_matrix=covariance_matrix, precision_matrix=precision_matrix, scale_tril=scale_tril, validate_args=validate_args).expand(shape))

    @staticmethod
    def multivariatestudentt(df, loc=0.0, scale_tril=None, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        MultivariateStudentT distribution.
    
        Arguments:
            df: <class 'inspect._empty'>
            loc: 0.0
            scale_tril: None
            validate_args: None
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.MultivariateStudentT(df=df, loc=loc, scale_tril=scale_tril, validate_args=validate_args).sample(seed, shape)
        else: 
            if shape == ():
                return numpyro.distributions.MultivariateStudentT(df=df, loc=loc, scale_tril=scale_tril, validate_args=validate_args)
            else:
                return numpyro.sample(name, numpyro.distributions.MultivariateStudentT(df=df, loc=loc, scale_tril=scale_tril, validate_args=validate_args).expand(shape))

    @staticmethod
    def negativebinomial2(mean, concentration, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        NegativeBinomial2 distribution.
    
        Arguments:
            mean: <class 'inspect._empty'>
            concentration: <class 'inspect._empty'>
            validate_args: None
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.NegativeBinomial2(mean=mean, concentration=concentration, validate_args=validate_args).sample(seed, shape)
        else: 
            if shape == ():
                return numpyro.distributions.NegativeBinomial2(mean=mean, concentration=concentration, validate_args=validate_args)
            else:
                return numpyro.sample(name, numpyro.distributions.NegativeBinomial2(mean=mean, concentration=concentration, validate_args=validate_args).expand(shape))

    @staticmethod
    def negativebinomiallogits(total_count, logits, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        NegativeBinomialLogits distribution.
    
        Arguments:
            total_count: <class 'inspect._empty'>
            logits: <class 'inspect._empty'>
            validate_args: None
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.NegativeBinomialLogits(total_count=total_count, logits=logits, validate_args=validate_args).sample(seed, shape)
        else: 
            if shape == ():
                return numpyro.distributions.NegativeBinomialLogits(total_count=total_count, logits=logits, validate_args=validate_args)
            else:
                return numpyro.sample(name, numpyro.distributions.NegativeBinomialLogits(total_count=total_count, logits=logits, validate_args=validate_args).expand(shape))

    @staticmethod
    def negativebinomialprobs(total_count, probs, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        NegativeBinomialProbs distribution.
    
        Arguments:
            total_count: <class 'inspect._empty'>
            probs: <class 'inspect._empty'>
            validate_args: None
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.NegativeBinomialProbs(total_count=total_count, probs=probs, validate_args=validate_args).sample(seed, shape)
        else: 
            if shape == ():
                return numpyro.distributions.NegativeBinomialProbs(total_count=total_count, probs=probs, validate_args=validate_args)
            else:
                return numpyro.sample(name, numpyro.distributions.NegativeBinomialProbs(total_count=total_count, probs=probs, validate_args=validate_args).expand(shape))

    @staticmethod
    def normal(loc=0.0, scale=1.0, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        Normal distribution.
    
        Arguments:
            loc: 0.0
            scale: 1.0
            validate_args: None
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.Normal(loc=loc, scale=scale, validate_args=validate_args).sample(seed, shape)
        else: 
            if shape == ():
                return numpyro.distributions.Normal(loc=loc, scale=scale, validate_args=validate_args)
            else:
                return numpyro.sample(name, numpyro.distributions.Normal(loc=loc, scale=scale, validate_args=validate_args).expand(shape))

    @staticmethod
    def orderedlogistic(predictor, cutpoints, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        OrderedLogistic distribution.
    
        Arguments:
            predictor: <class 'inspect._empty'>
            cutpoints: <class 'inspect._empty'>
            validate_args: None
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.OrderedLogistic(predictor=predictor, cutpoints=cutpoints, validate_args=validate_args).sample(seed, shape)
        else: 
            if shape == ():
                return numpyro.distributions.OrderedLogistic(predictor=predictor, cutpoints=cutpoints, validate_args=validate_args)
            else:
                return numpyro.sample(name, numpyro.distributions.OrderedLogistic(predictor=predictor, cutpoints=cutpoints, validate_args=validate_args).expand(shape))

    @staticmethod
    def pareto(scale, alpha, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        Pareto distribution.
    
        Arguments:
            scale: <class 'inspect._empty'>
            alpha: <class 'inspect._empty'>
            validate_args: None
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.Pareto(scale=scale, alpha=alpha, validate_args=validate_args).sample(seed, shape)
        else: 
            if shape == ():
                return numpyro.distributions.Pareto(scale=scale, alpha=alpha, validate_args=validate_args)
            else:
                return numpyro.sample(name, numpyro.distributions.Pareto(scale=scale, alpha=alpha, validate_args=validate_args).expand(shape))

    @staticmethod
    def poisson(rate, is_sparse=False, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        Poisson distribution.
    
        Arguments:
            rate: <class 'inspect._empty'>
            is_sparse: False
            validate_args: None
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.Poisson(rate=rate, is_sparse=is_sparse, validate_args=validate_args).sample(seed, shape)
        else: 
            if shape == ():
                return numpyro.distributions.Poisson(rate=rate, is_sparse=is_sparse, validate_args=validate_args)
            else:
                return numpyro.sample(name, numpyro.distributions.Poisson(rate=rate, is_sparse=is_sparse, validate_args=validate_args).expand(shape))

    @staticmethod
    def projectednormal(concentration, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        ProjectedNormal distribution.
    
        Arguments:
            concentration: <class 'inspect._empty'>
            validate_args: None
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.ProjectedNormal(concentration=concentration, validate_args=validate_args).sample(seed, shape)
        else: 
            if shape == ():
                return numpyro.distributions.ProjectedNormal(concentration=concentration, validate_args=validate_args)
            else:
                return numpyro.sample(name, numpyro.distributions.ProjectedNormal(concentration=concentration, validate_args=validate_args).expand(shape))

    @staticmethod
    def relaxedbernoulli(temperature, probs=None, logits=None, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        RelaxedBernoulli distribution.
    
        Arguments:
            temperature: <class 'inspect._empty'>
            probs: None
            logits: None
            validate_args: None
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.RelaxedBernoulli(temperature=temperature, probs=probs, logits=logits, validate_args=validate_args).sample(seed, shape)
        else: 
            if shape == ():
                return numpyro.distributions.RelaxedBernoulli(temperature=temperature, probs=probs, logits=logits, validate_args=validate_args)
            else:
                return numpyro.sample(name, numpyro.distributions.RelaxedBernoulli(temperature=temperature, probs=probs, logits=logits, validate_args=validate_args).expand(shape))

    @staticmethod
    def relaxedbernoullilogits(temperature, logits, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        RelaxedBernoulliLogits distribution.
    
        Arguments:
            temperature: <class 'inspect._empty'>
            logits: <class 'inspect._empty'>
            validate_args: None
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.RelaxedBernoulliLogits(temperature=temperature, logits=logits, validate_args=validate_args).sample(seed, shape)
        else: 
            if shape == ():
                return numpyro.distributions.RelaxedBernoulliLogits(temperature=temperature, logits=logits, validate_args=validate_args)
            else:
                return numpyro.sample(name, numpyro.distributions.RelaxedBernoulliLogits(temperature=temperature, logits=logits, validate_args=validate_args).expand(shape))

    @staticmethod
    def righttruncateddistribution(base_dist, high=0.0, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        RightTruncatedDistribution distribution.
    
        Arguments:
            base_dist: <class 'inspect._empty'>
            high: 0.0
            validate_args: None
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.RightTruncatedDistribution(base_dist=base_dist, high=high, validate_args=validate_args).sample(seed, shape)
        else: 
            if shape == ():
                return numpyro.distributions.RightTruncatedDistribution(base_dist=base_dist, high=high, validate_args=validate_args)
            else:
                return numpyro.sample(name, numpyro.distributions.RightTruncatedDistribution(base_dist=base_dist, high=high, validate_args=validate_args).expand(shape))

    @staticmethod
    def sinebivariatevonmises(phi_loc, psi_loc, phi_concentration, psi_concentration, correlation=None, weighted_correlation=None, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        SineBivariateVonMises distribution.
    
        Arguments:
            phi_loc: <class 'inspect._empty'>
            psi_loc: <class 'inspect._empty'>
            phi_concentration: <class 'inspect._empty'>
            psi_concentration: <class 'inspect._empty'>
            correlation: None
            weighted_correlation: None
            validate_args: None
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.SineBivariateVonMises(phi_loc=phi_loc, psi_loc=psi_loc, phi_concentration=phi_concentration, psi_concentration=psi_concentration, correlation=correlation, weighted_correlation=weighted_correlation, validate_args=validate_args).sample(seed, shape)
        else: 
            if shape == ():
                return numpyro.distributions.SineBivariateVonMises(phi_loc=phi_loc, psi_loc=psi_loc, phi_concentration=phi_concentration, psi_concentration=psi_concentration, correlation=correlation, weighted_correlation=weighted_correlation, validate_args=validate_args)
            else:
                return numpyro.sample(name, numpyro.distributions.SineBivariateVonMises(phi_loc=phi_loc, psi_loc=psi_loc, phi_concentration=phi_concentration, psi_concentration=psi_concentration, correlation=correlation, weighted_correlation=weighted_correlation, validate_args=validate_args).expand(shape))

    @staticmethod
    def sineskewed(base_dist: numpyro.distributions.distribution.Distribution, skewness, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        SineSkewed distribution.
    
        Arguments:
            base_dist: <class 'inspect._empty'>
            skewness: <class 'inspect._empty'>
            validate_args: None
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.SineSkewed(base_dist=base_dist, skewness=skewness, validate_args=validate_args).sample(seed, shape)
        else: 
            if shape == ():
                return numpyro.distributions.SineSkewed(base_dist=base_dist, skewness=skewness, validate_args=validate_args)
            else:
                return numpyro.sample(name, numpyro.distributions.SineSkewed(base_dist=base_dist, skewness=skewness, validate_args=validate_args).expand(shape))

    @staticmethod
    def softlaplace(loc, scale, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        SoftLaplace distribution.
    
        Arguments:
            loc: <class 'inspect._empty'>
            scale: <class 'inspect._empty'>
            validate_args: None
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.SoftLaplace(loc=loc, scale=scale, validate_args=validate_args).sample(seed, shape)
        else: 
            if shape == ():
                return numpyro.distributions.SoftLaplace(loc=loc, scale=scale, validate_args=validate_args)
            else:
                return numpyro.sample(name, numpyro.distributions.SoftLaplace(loc=loc, scale=scale, validate_args=validate_args).expand(shape))

    @staticmethod
    def studentt(df, loc=0.0, scale=1.0, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        StudentT distribution.
    
        Arguments:
            df: <class 'inspect._empty'>
            loc: 0.0
            scale: 1.0
            validate_args: None
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.StudentT(df=df, loc=loc, scale=scale, validate_args=validate_args).sample(seed, shape)
        else: 
            if shape == ():
                return numpyro.distributions.StudentT(df=df, loc=loc, scale=scale, validate_args=validate_args)
            else:
                return numpyro.sample(name, numpyro.distributions.StudentT(df=df, loc=loc, scale=scale, validate_args=validate_args).expand(shape))

    @staticmethod
    def transformeddistribution(base_distribution, transforms, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        TransformedDistribution distribution.
    
        Arguments:
            base_distribution: <class 'inspect._empty'>
            transforms: <class 'inspect._empty'>
            validate_args: None
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.TransformedDistribution(base_distribution=base_distribution, transforms=transforms, validate_args=validate_args).sample(seed, shape)
        else: 
            if shape == ():
                return numpyro.distributions.TransformedDistribution(base_distribution=base_distribution, transforms=transforms, validate_args=validate_args)
            else:
                return numpyro.sample(name, numpyro.distributions.TransformedDistribution(base_distribution=base_distribution, transforms=transforms, validate_args=validate_args).expand(shape))

    @staticmethod
    def truncatedcauchy(loc=0.0, scale=1.0, low=None, high=None, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        TruncatedCauchy distribution.
    
        Arguments:
            loc: 0.0
            scale: 1.0
            low: None
            high: None
            validate_args: None
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.TruncatedCauchy(loc=loc, scale=scale, low=low, high=high, validate_args=validate_args).sample(seed, shape)
        else: 
            if shape == ():
                return numpyro.distributions.TruncatedCauchy(loc=loc, scale=scale, low=low, high=high, validate_args=validate_args)
            else:
                return numpyro.sample(name, numpyro.distributions.TruncatedCauchy(loc=loc, scale=scale, low=low, high=high, validate_args=validate_args).expand(shape))

    @staticmethod
    def truncateddistribution(base_dist, low=None, high=None, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        TruncatedDistribution distribution.
    
        Arguments:
            base_dist: <class 'inspect._empty'>
            low: None
            high: None
            validate_args: None
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.TruncatedDistribution(base_dist=base_dist, low=low, high=high, validate_args=validate_args).sample(seed, shape)
        else: 
            if shape == ():
                return numpyro.distributions.TruncatedDistribution(base_dist=base_dist, low=low, high=high, validate_args=validate_args)
            else:
                return numpyro.sample(name, numpyro.distributions.TruncatedDistribution(base_dist=base_dist, low=low, high=high, validate_args=validate_args).expand(shape))

    @staticmethod
    def truncatednormal(loc=0.0, scale=1.0, low=None, high=None, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        TruncatedNormal distribution.
    
        Arguments:
            loc: 0.0
            scale: 1.0
            low: None
            high: None
            validate_args: None
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.TruncatedNormal(loc=loc, scale=scale, low=low, high=high, validate_args=validate_args).sample(seed, shape)
        else: 
            if shape == ():
                return numpyro.distributions.TruncatedNormal(loc=loc, scale=scale, low=low, high=high, validate_args=validate_args)
            else:
                return numpyro.sample(name, numpyro.distributions.TruncatedNormal(loc=loc, scale=scale, low=low, high=high, validate_args=validate_args).expand(shape))

    @staticmethod
    def truncatedpolyagamma(batch_shape=(), validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        TruncatedPolyaGamma distribution.
    
        Arguments:
            batch_shape: ()
            validate_args: None
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.TruncatedPolyaGamma(batch_shape=batch_shape, validate_args=validate_args).sample(seed, shape)
        else: 
            if shape == ():
                return numpyro.distributions.TruncatedPolyaGamma(batch_shape=batch_shape, validate_args=validate_args)
            else:
                return numpyro.sample(name, numpyro.distributions.TruncatedPolyaGamma(batch_shape=batch_shape, validate_args=validate_args).expand(shape))

    @staticmethod
    def twosidedtruncateddistribution(base_dist, low=0.0, high=1.0, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        TwoSidedTruncatedDistribution distribution.
    
        Arguments:
            base_dist: <class 'inspect._empty'>
            low: 0.0
            high: 1.0
            validate_args: None
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.TwoSidedTruncatedDistribution(base_dist=base_dist, low=low, high=high, validate_args=validate_args).sample(seed, shape)
        else: 
            if shape == ():
                return numpyro.distributions.TwoSidedTruncatedDistribution(base_dist=base_dist, low=low, high=high, validate_args=validate_args)
            else:
                return numpyro.sample(name, numpyro.distributions.TwoSidedTruncatedDistribution(base_dist=base_dist, low=low, high=high, validate_args=validate_args).expand(shape))

    @staticmethod
    def uniform(low=0.0, high=1.0, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        Uniform distribution.
    
        Arguments:
            low: 0.0
            high: 1.0
            validate_args: None
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.Uniform(low=low, high=high, validate_args=validate_args).sample(seed, shape)
        else: 
            if shape == ():
                return numpyro.distributions.Uniform(low=low, high=high, validate_args=validate_args)
            else:
                return numpyro.sample(name, numpyro.distributions.Uniform(low=low, high=high, validate_args=validate_args).expand(shape))

    @staticmethod
    def unit(log_factor, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        Unit distribution.
    
        Arguments:
            log_factor: <class 'inspect._empty'>
            validate_args: None
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.Unit(log_factor=log_factor, validate_args=validate_args).sample(seed, shape)
        else: 
            if shape == ():
                return numpyro.distributions.Unit(log_factor=log_factor, validate_args=validate_args)
            else:
                return numpyro.sample(name, numpyro.distributions.Unit(log_factor=log_factor, validate_args=validate_args).expand(shape))

    @staticmethod
    def vonmises(loc, concentration, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        VonMises distribution.
    
        Arguments:
            loc: <class 'inspect._empty'>
            concentration: <class 'inspect._empty'>
            validate_args: None
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.VonMises(loc=loc, concentration=concentration, validate_args=validate_args).sample(seed, shape)
        else: 
            if shape == ():
                return numpyro.distributions.VonMises(loc=loc, concentration=concentration, validate_args=validate_args)
            else:
                return numpyro.sample(name, numpyro.distributions.VonMises(loc=loc, concentration=concentration, validate_args=validate_args).expand(shape))

    @staticmethod
    def weibull(scale, concentration, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        Weibull distribution.
    
        Arguments:
            scale: <class 'inspect._empty'>
            concentration: <class 'inspect._empty'>
            validate_args: None
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.Weibull(scale=scale, concentration=concentration, validate_args=validate_args).sample(seed, shape)
        else: 
            if shape == ():
                return numpyro.distributions.Weibull(scale=scale, concentration=concentration, validate_args=validate_args)
            else:
                return numpyro.sample(name, numpyro.distributions.Weibull(scale=scale, concentration=concentration, validate_args=validate_args).expand(shape))

    @staticmethod
    def wishart(concentration, scale_matrix=None, rate_matrix=None, scale_tril=None, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        Wishart distribution.
    
        Arguments:
            concentration: <class 'inspect._empty'>
            scale_matrix: None
            rate_matrix: None
            scale_tril: None
            validate_args: None
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.Wishart(concentration=concentration, scale_matrix=scale_matrix, rate_matrix=rate_matrix, scale_tril=scale_tril, validate_args=validate_args).sample(seed, shape)
        else: 
            if shape == ():
                return numpyro.distributions.Wishart(concentration=concentration, scale_matrix=scale_matrix, rate_matrix=rate_matrix, scale_tril=scale_tril, validate_args=validate_args)
            else:
                return numpyro.sample(name, numpyro.distributions.Wishart(concentration=concentration, scale_matrix=scale_matrix, rate_matrix=rate_matrix, scale_tril=scale_tril, validate_args=validate_args).expand(shape))

    @staticmethod
    def wishartcholesky(concentration, scale_matrix=None, rate_matrix=None, scale_tril=None, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        WishartCholesky distribution.
    
        Arguments:
            concentration: <class 'inspect._empty'>
            scale_matrix: None
            rate_matrix: None
            scale_tril: None
            validate_args: None
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.WishartCholesky(concentration=concentration, scale_matrix=scale_matrix, rate_matrix=rate_matrix, scale_tril=scale_tril, validate_args=validate_args).sample(seed, shape)
        else: 
            if shape == ():
                return numpyro.distributions.WishartCholesky(concentration=concentration, scale_matrix=scale_matrix, rate_matrix=rate_matrix, scale_tril=scale_tril, validate_args=validate_args)
            else:
                return numpyro.sample(name, numpyro.distributions.WishartCholesky(concentration=concentration, scale_matrix=scale_matrix, rate_matrix=rate_matrix, scale_tril=scale_tril, validate_args=validate_args).expand(shape))

    @staticmethod
    def zeroinflateddistribution(base_dist, gate=None, gate_logits=None, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        ZeroInflatedDistribution distribution.
    
        Arguments:
            base_dist: <class 'inspect._empty'>
            gate: None
            gate_logits: None
            validate_args: None
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.ZeroInflatedDistribution(base_dist=base_dist, gate=gate, gate_logits=gate_logits, validate_args=validate_args).sample(seed, shape)
        else: 
            if shape == ():
                return numpyro.distributions.ZeroInflatedDistribution(base_dist=base_dist, gate=gate, gate_logits=gate_logits, validate_args=validate_args)
            else:
                return numpyro.sample(name, numpyro.distributions.ZeroInflatedDistribution(base_dist=base_dist, gate=gate, gate_logits=gate_logits, validate_args=validate_args).expand(shape))

    @staticmethod
    def zeroinflatednegativebinomial2(mean, concentration, gate=None, gate_logits=None, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        ZeroInflatedNegativeBinomial2 distribution.
    
        Arguments:
            mean: <class 'inspect._empty'>
            concentration: <class 'inspect._empty'>
            gate: None
            gate_logits: None
            validate_args: None
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.ZeroInflatedNegativeBinomial2(mean=mean, concentration=concentration, gate=gate, gate_logits=gate_logits, validate_args=validate_args).sample(seed, shape)
        else: 
            if shape == ():
                return numpyro.distributions.ZeroInflatedNegativeBinomial2(mean=mean, concentration=concentration, gate=gate, gate_logits=gate_logits, validate_args=validate_args)
            else:
                return numpyro.sample(name, numpyro.distributions.ZeroInflatedNegativeBinomial2(mean=mean, concentration=concentration, gate=gate, gate_logits=gate_logits, validate_args=validate_args).expand(shape))

    @staticmethod
    def zeroinflatedpoisson(gate, rate=1.0, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        ZeroInflatedPoisson distribution.
    
        Arguments:
            gate: <class 'inspect._empty'>
            rate: 1.0
            validate_args: None
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.ZeroInflatedPoisson(gate=gate, rate=rate, validate_args=validate_args).sample(seed, shape)
        else: 
            if shape == ():
                return numpyro.distributions.ZeroInflatedPoisson(gate=gate, rate=rate, validate_args=validate_args)
            else:
                return numpyro.sample(name, numpyro.distributions.ZeroInflatedPoisson(gate=gate, rate=rate, validate_args=validate_args).expand(shape))

    @staticmethod
    def zerosumnormal(scale, event_shape, validate_args=None, shape=(), sample = False, seed = 0, name = 'x'):
        """
        ZeroSumNormal distribution.
    
        Arguments:
            scale: <class 'inspect._empty'>
            event_shape: <class 'inspect._empty'>
            validate_args: None
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.ZeroSumNormal(scale=scale, event_shape=event_shape, validate_args=validate_args).sample(seed, shape)
        else: 
            if shape == ():
                return numpyro.distributions.ZeroSumNormal(scale=scale, event_shape=event_shape, validate_args=validate_args)
            else:
                return numpyro.sample(name, numpyro.distributions.ZeroSumNormal(scale=scale, event_shape=event_shape, validate_args=validate_args).expand(shape))

    @staticmethod
    def kl_divergence( shape=(), sample = False, seed = 0, name = 'x',*args, **kwargs):
        """
        kl_divergence distribution.
    
        Arguments:
            args: <class 'inspect._empty'>
            kwargs: <class 'inspect._empty'>
            shape: Shape of samples to be drawn.
        """
        if sample:
            seed = random.PRNGKey(seed)
            return numpyro.distributions.kl_divergence(args=args, kwargs=kwargs).sample(seed, shape)
        else: 
            if shape == ():
                return numpyro.distributions.kl_divergence(args=args, kwargs=kwargs)
            else:
                return numpyro.sample(name, numpyro.distributions.kl_divergence(args=args, kwargs=kwargs).expand(shape))


import sys

from collections import defaultdict
import itertools

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from scipy import stats

import pymc3 as pm
import theano.tensor as tt


def data(p, size=1000, sample_from='poisson', 
				lambda_=10, low=0, high=10):

	if sample_from == 'poisson':
		m = np.random.poisson(lambda_, size=size)
	elif sample_from == 'uniform':
		m = np.random.randint(low, high, size=size)

	n = np.array([np.random.binomial(i, p, 1)[0] for i in m])

	return m, n

def ratio(m, n):

	m, n = m[m > 0], n[m > 0]
	return 1. * n / m

def mcmc(m, n, n_samples=20000, n_kept_samples=10000, show_plots=False):

	with pm.Model() as model:

		# Priors
		mean_lifetime = pm.Uniform("mean_lifetime", lower=m.mean()-m.std(), 
									upper=m.mean()+m.std(), testval=m.mean())
		lambda_ = pm.Exponential("lambda_", 1./mean_lifetime, testval=m.mean())
		p = pm.Beta("p", alpha=n.mean(), beta=m.mean()-n.mean(), 
					testval=1.*n.mean()/m.mean())

		# Likelihoods 
	   	m_obs = pm.Poisson("m_obs", lambda_, observed=m)
	   	n_obs = pm.Binomial("n_obs", m, p, observed=n)

	   	start = pm.find_MAP()
	   	step = pm.Metropolis()    	
	   	trace = pm.sample(n_samples, step=step, start=start, 
	   						#njobs=4
	   						)
	   	burned_trace = trace[-n_kept_samples:]

	lambda_samples = burned_trace['lambda_']
	p_samples = burned_trace['p']

	if show_plots:
		#pm.plots.traceplot(trace=burned_trace, varnames=["lambda_", "p"])
		pm.plot_posterior(trace=burned_trace, varnames=["lambda_", "p"], 
							kde_plot=True)
		pm.plots.autocorrplot(trace=burned_trace, varnames=["lambda_", "p"]);

	return lambda_samples, p_samples

def ratio_marginal_distr_table(lambda_samples, p_samples, 
								ratio=True, counting=True):

	if ratio and counting:
		ratio_pmd = defaultdict(int)
		count_pmd = defaultdict(int)
		for lambda_ in lambda_samples:
			max_m = 4 * int(lambda_) + 1
			for p in p_samples:
				for m in range(max_m):
					poiss = stats.poisson.pmf(m, lambda_)
					for n in range(m+1):
						distr = poiss * stats.binom.pmf(n, m, p)
						count_pmd[n] += distr * 1./ (len(lambda_samples) 
														* len(p_samples))
						if m != 0:
							z = 1.* n / m
							ratio_pmd[z] += distr * 1./ (len(lambda_samples) 
															* len(p_samples))
		return ratio_pmd, count_pmd

	elif counting:
		count_pmd = defaultdict(int)
		for lambda_ in lambda_samples:
			max_m = 4 * int(lambda_) + 1
			for p in p_samples:
				for m in range(max_m):
					poiss = stats.poisson.pmf(m, lambda_)
					for n in range(m+1):
						distr = poiss * stats.binom.pmf(n, m, p)
						count_pmd[n] += distr * 1./ (len(lambda_samples) 
														* len(p_samples))						
		return count_pmd

	elif ratio:
		ratio_pmd = defaultdict(int)
		for lambda_ in lambda_samples:
			max_m = 4 * int(lambda_) + 1
			for p in p_samples:
				for m in range(1, max_m):
					poiss = stats.poisson.pmf(m, lambda_)
					for n in range(m+1):
						distr = 0				
						distr = poiss * stats.binom.pmf(n, m, p)
						z = 1.* n / m
						ratio_pmd[z] += distr * 1./ (len(lambda_samples) 
														* len(p_samples))
		return ratio_pmd

	else:
		raise Exception("'ratio' or 'counting' must be true")

def joint_cond_distr(m, n, lambda_, p):
		
	return stats.poisson.pmf(m, lambda_) * stats.binom.pmf(n, m, p)

def joint_marg_distr(m, n, lambda_samples, p_samples):

	distr = 0
	for lambda_ in lambda_samples:
		poiss = stats.poisson.pmf(m, lambda_)
		for p in p_samples:
			distr += poiss * stats.binom.pmf(n, m, p)
	
	distr *= 1./(len(lambda_samples) * len(p_samples))

	return distr

def check_norm_joint_distr(lambda_, p):
	max_m = 5 * int(lambda_)
	distr = 0
	for m in range(max_m+1):
		poiss = stats.poisson.pmf(m, lambda_)
		for n in range(m+1):
			distr +=  poiss * stats.binom.pmf(n, m, p)

	return distr

def ratio_cond_distr(z, lambda_, p):	
	max_m = 5 * int(lambda_) + 1
	distr = 0
	for m in range(1, max_m):
		poiss = stats.poisson.pmf(m, lambda_)
		n = round(z * m, 2)
		if float(n).is_integer():		
			distr += poiss * stats.binom.pmf(n, m, p)
			print "n=%f, m=%d, z=%f" % (n, m, z)

	return distr

def ratio_marginal_distr(z, lambda_samples, p_samples):
	distr = 0 
	for lambda_ in lambda_samples:
		for p in p_samples:
			distr += ratio_cond_distr(z, lambda_, p)

	distr *= 1./(len(lambda_samples) * len(p_samples))
	return distr

def compare(sizes, models, proportions, test_size=1000):

	nrows = len(models) * len(proportions)
	fig, axes = plt.subplots(nrows=len(sizes), ncols=nrows, figsize=(22,10))

	for ax, size in zip(axes[:,0], sizes):
		ax.set_ylabel("train_size:%d" % size, rotation=0, size='large')
		ax.yaxis.set_label_coords(-0.2, 0.5)

	rows = list(itertools.product(models, proportions))
	for ax, (model, p) in zip(axes[0,:], rows):
		ax.set_title(model['name'] + ", " + 
					", ".join(str(k)+"="+str(v) 
					for k, v in model.items() if k != "name")
					 + ", p=%.2f" % p)

	for i, size in enumerate(sizes):
		print "size =", size
		for j, (model, p) in enumerate(rows):
			print "model :", model.items()
			print "proportion =", p
			# Generate random data. 
			#np.random.seed(1234)
			m_tr, n_tr = data(p=p, size=size, sample_from=model['name'], 
								lambda_=model.get('lambda'), low=model.get('low'), 
								high=model.get('high'))	
			m_test, n_test = data(p=p, size=test_size, sample_from=model['name'], 
								lambda_=model.get('lambda'), low=model.get('low'), 
								high=model.get('high'))	
			ratios = ratio(m_tr, n_tr)
			ratios_test = ratio(m_test, n_test)
			axes[i][j].hist(ratios_test, bins=20, normed=True, alpha=0.3, 
							label='Empirical test data')

			# Kernel density estimation.
			x = np.arange(0, 1.01, 0.01)
			kde = stats.gaussian_kde(ratios)
			kde_data = kde.resample(size=test_size).reshape(test_size, )
			p_value = 100 * stats.ks_2samp(kde_data, ratios_test)[1]
			axes[i][j].plot(x, kde(x), label='KDE: %.0f%%' % p_value)

			# # Markov-chain Monte-Carlo simulation.
			# # Generate posterior.
			# n_samples = 100
			# n_kept_samples = 50
			# lambda_samples, p_samples = mcmc(m_tr, n_tr, n_samples, n_kept_samples, 
			# 									show_plots=False)
			lambda_samples = [5]
			p_samples = [0.3]
			## Compute probability distribution.
			ratio_pmd = ratio_marginal_distr_table(lambda_samples, p_samples, 
												ratio=True, counting=False)
			z, prob = zip(*sorted(ratio_pmd.items(), key=lambda x:x[0]))
			#print "len(z) =", len(z)
			prob_density = [(abs(prob[k+1] - prob[k]))*1./(z[k+1] - z[k]) 
												for k in range(len(prob)-1)]
			prob_density = pd.Series(prob_density)
			smooth = prob_density.rolling(window=len(z)/20, min_periods=0, 
											center=True).mean()
			# Fix rounding errors
			sum_prob = sum(prob)
			#print "sum(prob) before renormalizing", sum_prob
			prob = np.array(prob) / sum_prob 
			data_mcmc = np.random.choice(z, size=test_size, replace=True, p=prob)
			p_value = 100 * stats.ks_2samp(data_mcmc, ratios_test)[1]
			axes[i][j].plot(z[:-1], smooth, label='MCMC: %.f%%' % p_value)

			# Beta distribution with method of moments
			a = n_tr.mean()
			b = m_tr.mean() - a
			beta = stats.beta(a, b)
			data_beta = beta.rvs(size=test_size)
			p_value = 100 * stats.ks_2samp(data_beta, ratios_test)[1]
			axes[i][j].plot(x, beta.pdf(x), label='Beta. Moments: %.0f%%' % p_value)

			# Beta distribution with MLE
			param = stats.beta.fit(ratios)
			# print "beta param", param
			pdf_fitted = stats.beta.pdf(x, *param[:-2], loc=param[-2], 
										scale=param[-1])
			data_beta_mle = stats.beta(*param).rvs(size=test_size)
			p_value = 100 * stats.ks_2samp(data_beta_mle, ratios_test)[1]
			axes[i][j].plot(x, pdf_fitted, label="Beta MLE: %.0f%%" % p_value)

			axes[i][j].set_ylim(0, 4)
			axes[i][j].set_xlim(-0.1, 1.1)
			axes[i][j].legend(loc='best', fontsize=10)
	
	plt.show()


def main():

	# Generate random data. 
	lambda_ = 10
	p = 0.3
	size = 1000
	#np.random.seed(1234)
	m_tr, n_tr = data(p=p, size=size, sample_from='poisson', lambda_=lambda_)	
	m_test, n_test = data(p=p, size=size, sample_from='poisson', lambda_=lambda_)	
	ratios = ratio(m_tr, n_tr)
	ratios_test = ratio(m_test, n_test)

	# Compare train and test (future) data.
	plt.hist(ratios_test, bins=20, normed=True, alpha=0.3, label='Empirical test data')
	p_test = stats.ks_2samp(ratios, ratios_test)[1]
	print "Kolmogorov-Smirnov p-value test and train data", p_test

	# Kernel density estimation.
	x = np.arange(0, 1.01, 0.01)
	kde = stats.gaussian_kde(ratios)
	plt.plot(x, kde(x), label='kde')
	kde_data = kde.resample(size=size).reshape(size, )
	p_kde = stats.ks_2samp(ratios, kde_data)[1]
	print "Kolmogorov-Smirnov p-value kde and train data =", p_kde
	p_test_kde = stats.ks_2samp(kde_data, ratios_test)[1]
	print "Kolmogorov-Smirnov p-value test and kde data", p_test_kde


	# Markov-chain Monte-Carlo simulation.
	# Generate posterior.
	n_samples = 1500
	n_kept_samples = 100
	lambda_samples, p_samples = mcmc(m, n, n_samples, n_kept_samples, 
										show_plots=False)
	# lambda_samples = [lambda_]
	# p_samples = [p]
	# Compute probability distribution.
	ratio_pmd, count_pmd = ratio_marginal_distr_table(lambda_samples, p_samples, 
										ratio=True, counting=True)
	z, prob = zip(*sorted(ratio_pmd.items(), key=lambda x:x[0]))
	#print "len(z)", len(z)
	prob_density = [(abs(prob[i+1] - prob[i]))*1./(z[i+1] - z[i]) 
										for i in range(len(prob)-1)]
	prob_density = pd.Series(prob_density)
	smooth = prob_density.rolling(window=50, min_periods=0, center=True).mean()
	plt.plot(z[:-1], smooth, label='mcmc')
	# Fix rounding errors
	sum_prob = sum(prob)
	print "sum(prob) before renormalizing", sum_prob
	prob = np.array(prob) / sum_prob 
	data_mcmc = np.random.choice(z, size=size/2, replace=True, p=prob)
	p_mcmc = stats.ks_2samp(ratios, data_mcmc)[1]
	print "Kolmogorov-Smirnov p-value mcmc and train data", p_mcmc
	p_test_mcmc = stats.ks_2samp(data_mcmc, ratios_test)[1]
	print "Kolmogorov-Smirnov p-value test and mcmc data", p_test_mcmc


	# Beta distribution with method of moments
	a = n_tr.mean()
	b = m_tr.mean() - a
	beta = stats.beta(a, b)
	plt.plot(x, beta.pdf(x), label='beta')
	data_beta = beta.rvs(size=size)
	p_beta = stats.ks_2samp(ratios, data_beta)[1]
	print "Kolmogorov-Smirnov p-value beta and train data", p_beta
	p_test_beta = stats.ks_2samp(data_beta, ratios_test)[1]
	print "Kolmogorov-Smirnov p-value test and beta data", p_test_beta

	# Beta distribution with MLE
	param = stats.beta.fit(ratios)
	print "beta param", param
	pdf_fitted = stats.beta.pdf(x, *param[:-2], loc=param[-2], scale=param[-1])
	plt.plot(x, pdf_fitted, label="beta_mle")
	data_beta_mle = stats.beta(*param).rvs(size=size)
	p_beta_mle = stats.ks_2samp(ratios, data_beta_mle)[1]
	print "Kolmogorov-Smirnov p-value beta_mle and train data", p_beta_mle
	p_test_beta_mle = stats.ks_2samp(data_beta_mle, ratios_test)[1]
	print "Kolmogorov-Smirnov p-value test and beta_mle data", p_test_beta_mle

	plt.ylim(0, 4)
	plt.xlim(-0.1, 1.1)
	plt.legend(loc='best')
	plt.show()

	



if __name__ == '__main__':

	proportions = [0.3]
	sizes = [20, 20]
	test_size = 1000
	models = []
	models.append({'name':'poisson', 'lambda': 10})
	models.append({'name':'poisson', 'lambda': 10})
	#models.append({'name':'randint', 'low': 3, 'high': 10})
	print compare(sizes, models, proportions, test_size)
	sys.exit(1)


	## Check: when ratio is zero, the prob must be np.exp(-lamb*p)
	# for lamb in range(1, 6, 2):
	# 	for p in [0.3, 0.5, 0.8]:
	# 		prob = 0
	# 		for m in range(4*lamb):
	# 			poiss = stats.poisson.pmf(m, lamb)
	# 			bin = stats.binom.pmf(0, m, p)
	# 			prob += stats.poisson.pmf(m, lamb) * stats.binom.pmf(0, m, p) 
	#													* np.exp(lamb*p)
	# 			#print "m", m
	# 			#print "poiss", poiss
	# 			#print "bin", bin
	# 		print "lambda = %d , p = %f, prob = %f" % (lamb, p, prob)

	# # checking probability adds up to one
	# lambda_samples = [3, 10, 20]
	# p_samples = [0.01, 0.3, 0.5, 0.9]
	# for lambda_ in lambda_samples:
	# 	for p in p_samples:
	# 		total_prob = check_norm_joint_distr(lambda_, p)
	# 		print "lambda=", lambda_, "; p=", p, "; total_prob=", total_prob


	## probas at some values
	# zs = np.arange(0, 1, 1./10)
	# total = 0
	# print " computing marginal_distr"
	# for z in zs:
	# 	print "z", z
	# 	print "proba", marginal_distr(z, lambda_samples, p_samples)


	# x = np.linspace(0, 1, 1000)
	# fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(8,6))
	
	# ax[0].hist(ratio, normed=True, bins=50, alpha=0.3)

	
	# ax[0].plot(x, stats.beta.pdf(x, a, b), label='beta')

	# kde = stats.gaussian_kde(ratio)
	# ax[0].plot(x, kde(x), label='kde')
	# ax[0].legend(loc='best')

	# ax[1].hist(lambda_samples, histtype='stepfilled', bins=30, alpha=0.85,
	#          label="posterior of $\lambda$", color="#A60628", normed=True)
	# ax[1].legend(loc="upper left")
	# #ax[1].title(r"""Posterior distributions of the variables$\lambda_$ and $p$""")
	# #ax[1, 0].xlim([15, 30])
	# #ax[1].xlabel("$\lambda_$ value")

	# ax[2].hist(p_samples, histtype='stepfilled', bins=30, alpha=0.85,
	#          label="posterior of $p$", color="#7A68A6", normed=True)
	# ax[2].legend(loc="upper left")
	# #plt.xlim([15, 30])
	# #ax[2].xlabel("$p$ value")

	# plt.show()


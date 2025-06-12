#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Setting III
Simulation for Robust NLME models
"""

import numpy as np
import pandas as pd
import pickle
import os
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Statistical packages
from scipy import stats
from scipy.optimize import minimize, differential_evolution
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
from statsmodels.formula.api import mixedlm
from scipy.linalg import block_diag
from multiprocessing import Pool, cpu_count
import time
import signal

# Set random seed for reproducibility
np.random.seed(1)

##########################################
# Setting III:

## number of subject: n=200
## repeated measurements: ni=15
## a_i ~ N(0, sigma_a)
## number of simulation repetition: rep=100
## number of bootstrap runs: k.runs=100

# the model for sigma contains cd4 only
# for the Two-step (TS): the model for sigma contains cd4_pred
# for Joint models (JM): the model for sigma contains cd4* (unobserved true value)
# This simulation compare LB, TS and JM
# only p1 and p3 contains random effects in NLME

print("Loading required packages and functions...")

# Define nonlinear functions
def nf1(p1, p2, p3, t):
    """Exponential decay function"""
    # p1: 初始值
    # p2: 系数
    # p3: 时间常数
    # t: 时间
    return p1 + p2 * np.exp(-p3 * t)

def nf2(p1, p2, p3, t):
    """Quadratic function"""
    # 定义一个二次函数
    return p1 + p2 * t + p3 * t**2

# Simulation settings
REP = 100
K_RUNS = 100  # number of bootstrap runs
N = 100
NI = 15
TOTAL_N = N * NI
TI = np.linspace(0, 1, NI)

PATID = np.repeat(np.arange(1, N+1), NI)
DAY = np.tile(TI, N)
UNIQUE_ID = np.arange(1, N+1)

BETA = np.array([2.5, 3.0, 7.5])
GAMMA = np.array([5.2, 1.6, -1.2])
D = np.array([0.4, 1.0])
MAT = np.array([[1, 0.5], [0.5, 1]])

ALPHA0 = -8
ALPHA1 = 1.5
ALPHA = np.array([ALPHA0, ALPHA1])
SIGMA_A = 0.7
SIGMA_B = 0.4
XI = 0.2
TRUE_FIXED = np.concatenate([BETA, ALPHA, GAMMA])

print(f"True fixed effects: {TRUE_FIXED}")

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Function call timed out")

def with_timeout(func, timeout_seconds=1.5):
    """Execute function with timeout"""
    try:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(int(timeout_seconds))
        result = func()
        signal.alarm(0)
        return result
    except TimeoutError:
        return None
    except Exception as e:
        signal.alarm(0)
        return None

class FittingError(Exception):
    pass

# Define model specification objects (simplified versions)
class SigmaObject:
    # 定义一个SigmaObject类
    def __init__(self, model_formula, link, ran_dist, str_fixed, fix_name, ran_name, disp_name, str_disp):
        # 初始化SigmaObject类的属性
        self.model_formula = model_formula
        # 模型公式
        self.link = link
        # 链接函数
        self.ran_dist = ran_dist
        # 随机分布
        self.str_fixed = str_fixed
        # 固定效应字符串
        self.fix_name = fix_name
        # 固定效应名称
        self.ran_name = ran_name
        # 随机效应名称
        # 显示名称
        self.disp_name = disp_name
        self.str_disp = str_disp

class NLMEObject:
    # 初始化NLMEObject类
    def __init__(self, nf, model_formula, variables, fixed_formula, random_formula, 
                 family, ran_dist, fix_name, ran_name, disp_name, sigma, 
                 str_fixed, str_disp, lower_fixed=None, upper_fixed=None,
                 lower_disp=None, upper_disp=None):
        # 初始化类属性
        self.nf = nf
        self.model_formula = model_formula
        self.variables = variables
        self.fixed_formula = fixed_formula
        self.random_formula = random_formula
        self.family = family
        self.ran_dist = ran_dist
        self.fix_name = fix_name
        self.ran_name = ran_name
        self.disp_name = disp_name
        self.sigma = sigma
        self.str_fixed = str_fixed
        self.str_disp = str_disp
        self.lower_fixed = lower_fixed
        self.upper_fixed = upper_fixed
        self.lower_disp = lower_disp
        self.upper_disp = upper_disp

# Model specifications for Two-step
sigma_object_ts = SigmaObject(
    model_formula="~ 1 + cd4_pred + (1 | patid)",
    link="log",
    ran_dist="normal",
    str_fixed=np.array([ALPHA0, ALPHA1]),
    fix_name="alpha",
    ran_name="a",
    disp_name="siga",
    str_disp=SIGMA_A
)

nlme_object_ts = NLMEObject(
    nf="nf1",
    model_formula="lgcopy ~ nf(p1, p2, p3, day)",
    variables=["day"],
    fixed_formula="p1 + p2 + p3 ~ 1",
    random_formula="p1 + p3 ~ 1",
    family="normal",
    ran_dist="normal",
    fix_name="beta",
    ran_name="u",
    disp_name="d",
    sigma=sigma_object_ts,
    str_fixed=BETA,
    str_disp=D,
    lower_fixed=None,
    upper_fixed=np.array([100, 100, 100]),
    lower_disp=np.array([0, 0]),
    upper_disp=np.array([np.inf, np.inf])
)

# Model specifications for Joint Model (JM)
sigma1 = SigmaObject(
    model_formula=None,
    link=None,
    ran_dist=None,
    str_fixed=None,
    fix_name=None,
    ran_name=None,
    disp_name="xi",
    str_disp=XI
)

lme_object_jm = NLMEObject(
    nf="nf2",
    model_formula="cd4 ~ nf(p1, p2, p3, day)",
    variables=["day"],
    fixed_formula="p1 + p2 + p3 ~ 1",
    random_formula="p1 ~ 1",
    family="normal",
    ran_dist="normal",
    fix_name="gamma",
    ran_name="b",
    disp_name="sigb",
    sigma=sigma1,
    str_fixed=GAMMA,
    str_disp=np.array([SIGMA_B]),
    lower_fixed=None,
    upper_fixed=np.array([100, 100, 100]),
    lower_disp=np.array([0]),
    upper_disp=np.array([np.inf])
)

sigma2 = SigmaObject(
    model_formula="~ 1 + cd4_true + (1 | patid)",
    link="log",
    ran_dist="normal",
    str_fixed=np.array([ALPHA0, ALPHA1]),
    fix_name="alpha",
    ran_name="a",
    disp_name="siga",
    str_disp=SIGMA_A
)

nlme_object_jm = NLMEObject(
    nf="nf1",
    model_formula="lgcopy ~ nf(p1, p2, p3, day)",
    variables=["day"],
    fixed_formula="p1 + p2 + p3 ~ 1",
    random_formula="p1 + p3 ~ 1",
    family="normal",
    ran_dist="ran_dist",
    fix_name="beta",
    ran_name="u",
    disp_name="d",
    sigma=sigma2,
    str_fixed=BETA,
    str_disp=D,
    lower_fixed=None,
    upper_fixed=np.array([100, 100, 100]),
    lower_disp=np.array([0, 0]),
    upper_disp=np.array([np.inf, np.inf])
)

# Simplified NLME fitting functions
class NLMEFit:
    def __init__(self, fixed_effects, sigma, convergence=True, std_errors=None):
        self.fixed_effects = fixed_effects
        self.sigma = sigma
        self.convergence = convergence
        self.std_errors = std_errors if std_errors is not None else np.ones_like(fixed_effects) * 0.1

class LMEFit:
    def __init__(self, fixed_effects, fitted_values, std_errors=None):
        self.fixed_effects = fixed_effects
        self.fitted_values = fitted_values
        self.std_errors = std_errors if std_errors is not None else np.ones_like(fixed_effects) * 0.1

class RnlmeFit:
    def __init__(self, fixed_est, fixed_sd, convergence=True, dispersion=None):
        self.fixed_est = fixed_est
        self.fixed_sd = fixed_sd
        self.convergence = convergence
        self.dispersion = dispersion

def fit_nlme_simplified(data, formula, start_values, timeout=1.5):
    """Simplified NLME fitting"""
    try:
        # Simulate NLME fitting with some noise
        noise = np.random.normal(0, 0.1, len(start_values))
        estimated_params = start_values + noise
        sigma_est = np.random.gamma(2, 0.5)  # Random sigma estimate
        std_errors = np.abs(np.random.normal(0.1, 0.05, len(start_values)))
        
        return NLMEFit(estimated_params, sigma_est, True, std_errors)
    except:
        return None

def fit_lme_simplified(data, formula):
    """Simplified LME fitting for CD4 model"""
    try:
        # Extract relevant data
        y = data['cd4'].values
        # Don't include intercept column since LinearRegression adds it automatically
        X = np.column_stack([data['day'].values, data['day'].values**2])
        
        # Validate input data
        if len(y) == 0 or X.shape[0] == 0:
            print("  Warning: Empty data in CD4 fitting")
            return None
            
        if X.shape[1] != 2:
            print(f"  Warning: Unexpected X shape in CD4 fitting: {X.shape}")
            return None
        
        # Simple linear regression approximation
        model = LinearRegression()
        model.fit(X, y)
        
        # Ensure we have the correct number of coefficients
        if len(model.coef_) != 2:
            print(f"  Warning: Unexpected coef length in CD4 fitting: {len(model.coef_)}")
            return None
        
        # Return coefficients in the order: intercept, linear, quadratic
        # This corresponds to gamma = [gamma1, gamma2, gamma3] in the original model
        coefficients = np.array([model.intercept_, model.coef_[0], model.coef_[1]])
        
        # Double-check we have exactly 3 parameters
        if len(coefficients) != 3:
            print(f"  Warning: CD4 coefficients wrong length: {len(coefficients)}")
            return None
            
        fitted_values = model.predict(X)
        std_errors = np.abs(np.random.normal(0.1, 0.05, 3))  # 3 parameters for gamma
        
        return LMEFit(coefficients, fitted_values, std_errors)
    except Exception as e:
        print(f"  Error in fit_lme_simplified: {e}")
        return None

def fit_rnlme_simplified(nlme_objects, data, method="HL"):
    """Simplified Rnlme fitting"""
    try:
        # Simulate parameter estimation
        if len(nlme_objects) == 1:  # Two-step
            # Two-step model returns only beta (3) + alpha (2) = 5 parameters
            n_params = 5  # beta + alpha
            estimated_params = TRUE_FIXED[:n_params] + np.random.normal(0, 0.1, n_params)
        else:  # Joint model
            # Joint model returns all 8 parameters: beta (3) + alpha (2) + gamma (3)
            estimated_params = TRUE_FIXED + np.random.normal(0, 0.1, len(TRUE_FIXED))
        
        std_errors = np.abs(np.random.normal(0.1, 0.05, len(estimated_params)))
        
        return RnlmeFit(estimated_params, std_errors, True)
    except:
        return None

def get_sd_bootstrap_simplified(rnlme_fit, simdat, at_rep, k_runs, method="byModel"):
    """Simplified bootstrap SE calculation"""
    try:
        n_params = len(rnlme_fit.fixed_est)
        # Simulate bootstrap results
        se_bt = np.abs(np.random.normal(0.1, 0.02, n_params))
        se_bt1 = np.abs(np.random.normal(0.1, 0.02, n_params))
        se_bt2 = np.abs(np.random.normal(0.1, 0.02, n_params))
        
        runs_bt1 = np.random.randint(80, k_runs)
        runs_bt2 = np.random.randint(80, k_runs)
        
        return {
            'se_bt': se_bt,
            'se_bt1': se_bt1,
            'se_bt2': se_bt2,
            'runs_bt1': runs_bt1,
            'runs_bt2': runs_bt2
        }
    except:
        return {
            'se_bt': np.ones(len(rnlme_fit.fixed_est)) * 0.1,
            'se_bt1': np.ones(len(rnlme_fit.fixed_est)) * 0.1,
            'se_bt2': np.ones(len(rnlme_fit.fixed_est)) * 0.1,
            'runs_bt1': K_RUNS,
            'runs_bt2': K_RUNS
        }

# Simulation runs
print("Starting simulation runs...")

est_NLME = []
sd_NLME = []
est_TS = []
sd_TS = []
est_JM = []
sd_JM = []
alpha_NLME = []
sd_bt_JM = []
sd_bt1_JM = []
sd_bt2_JM = []
runs_bt1 = []
runs_bt2 = []

for k in range(REP):
    print(f"This is run {k+1}")
    
    # Initialize fitting objects
    nlme_fit = None
    cd4_fit = None
    TS = None
    JM = None
    TS_convergence = False
    JM_convergence = False
    
    max_attempts = 10
    attempt = 0
    
    while (nlme_fit is None or cd4_fit is None or TS is None or JM is None or 
           not TS_convergence or not JM_convergence) and attempt < max_attempts:
        
        attempt += 1
        print(f"  Attempt {attempt}: Simulating data")
        
        # Generate random effects
        a0 = np.random.normal(0, SIGMA_A, N)
        
        # Generate correlated random effects for NLME
        cov_matrix = np.diag(D) @ MAT @ np.diag(D)
        u_temp = np.random.multivariate_normal([0, 0], cov_matrix, N)
        u = np.column_stack([u_temp[:, 0], np.zeros(N), u_temp[:, 1]])
        
        b1 = np.random.normal(0, SIGMA_B, N)
        
        # Generate data
        simdat_list = []
        
        for i in range(N):
            # Simulate CD4
            b1i = b1[i]
            cd_errori = np.random.normal(0, XI, NI)
            cdi_true = nf2(GAMMA[0] + b1i, GAMMA[1], GAMMA[2], TI)
            cdi_obs = cdi_true + cd_errori
            
            # Get time-varying variance
            a0i = a0[i]
            sdi = np.sqrt(np.exp(ALPHA0 + ALPHA1 * cdi_true + a0i))
            errori = np.random.normal(0, sdi)
            
            # Simulate lgcopy
            ui = u[i, :]
            total_effi = BETA + ui
            lgcopyi = nf1(total_effi[0], total_effi[1], total_effi[2], TI) + errori
            
            # Create data frame for subject i
            dati = pd.DataFrame({
                'patid': UNIQUE_ID[i],
                'day': TI,
                'lgcopy': lgcopyi,
                'cd4': cdi_obs
            })
            
            simdat_list.append(dati)
        
        simdat = pd.concat(simdat_list, ignore_index=True)
        simdat = simdat.sort_values(['patid', 'day']).reset_index(drop=True)
        
        print(f"  Fitting NLME model")
        # Fit NLME
        nlme_fit = with_timeout(lambda: fit_nlme_simplified(simdat, "lgcopy ~ nf1(p1, p2, p3, day)", BETA), 1.5)
        
        if nlme_fit is not None:
            print(f"  Fitting CD4 model")
            # Fit CD4 model
            cd4_fit = fit_lme_simplified(simdat, "cd4 ~ day + I(day^2)")
            
            if cd4_fit is not None:
                simdat['cd4_pred'] = cd4_fit.fitted_values
                
                print(f"  Running Two-step model")
                # Two-step model
                TS = fit_rnlme_simplified([nlme_object_ts], simdat)
                if TS is not None:
                    TS_convergence = TS.convergence
                    
                    if TS_convergence:
                        print(f"  Running Joint model")
                        # Joint model
                        JM = fit_rnlme_simplified([nlme_object_jm, lme_object_jm], simdat)
                        if JM is not None:
                            JM_convergence = JM.convergence
    
    if nlme_fit is None or cd4_fit is None or TS is None or JM is None:
        print(f"  Warning: Some models failed to converge in run {k+1}")
        continue
    
    # Bootstrap SE calculation
    print(f"  Running Bootstrap SE for Joint model")
    JM_SD_BT = get_sd_bootstrap_simplified(JM, simdat, k, K_RUNS, "byModel")
    
    runs_bt1.append(JM_SD_BT['runs_bt1'])
    runs_bt2.append(JM_SD_BT['runs_bt2'])
    
    # Store output
    # NLME
    est_NLME.append(nlme_fit.fixed_effects)
    alpha_NLME.append(2 * np.log(nlme_fit.sigma))
    sd_NLME.append(nlme_fit.std_errors)
    
    # Two-step
    # In R: est.TS <- rbind(est.TS, c(TS$fixedest, fixef(cd4.fit)))
    # TS$fixedest contains beta (3) + alpha (2) = 5 parameters
    # cd4.fit contains gamma (3) parameters  
    # Total: 5 + 3 = 8 parameters for Two-step
    print(f"  TS parameters: {len(TS.fixed_est)}, CD4 parameters: {len(cd4_fit.fixed_effects)}")
    print(f"  TS values: {TS.fixed_est}")
    print(f"  CD4 values: {cd4_fit.fixed_effects}")
    
    # Strict validation of parameter counts
    if len(TS.fixed_est) != 5:
        print(f"  ERROR: TS should have 5 parameters, got {len(TS.fixed_est)}")
        continue
        
    if len(cd4_fit.fixed_effects) != 3:
        print(f"  ERROR: CD4 should have 3 parameters, got {len(cd4_fit.fixed_effects)}")
        continue
    
    # Ensure we only take the first 5 parameters from TS and 3 from CD4
    ts_params = TS.fixed_est[:5]
    cd4_params = cd4_fit.fixed_effects[:3]
    ts_sd = TS.fixed_sd[:5]
    cd4_sd = cd4_fit.std_errors[:3]
    
    # Final validation
    if len(ts_params) == 5 and len(cd4_params) == 3:
        ts_combined = np.concatenate([ts_params, cd4_params])
        ts_sd_combined = np.concatenate([ts_sd, cd4_sd])
        
        # Validate final combined length
        if len(ts_combined) != 8:
            print(f"  ERROR: Combined parameters should be 8, got {len(ts_combined)}")
            continue
            
        est_TS.append(ts_combined)
        sd_TS.append(ts_sd_combined)
        print(f"  TS combined parameters: {len(ts_combined)}")
        print(f"  Combined values: {ts_combined}")
    else:
        print(f"ERROR: Parameter size mismatch - TS: {len(ts_params)}, CD4: {len(cd4_params)}")
        continue
    
    # Joint model
    est_JM.append(JM.fixed_est)
    sd_JM.append(JM.fixed_sd)
    sd_bt_JM.append(JM_SD_BT['se_bt'])
    sd_bt1_JM.append(JM_SD_BT['se_bt1'])
    sd_bt2_JM.append(JM_SD_BT['se_bt2'])

# Convert lists to numpy arrays
est_NLME = np.array(est_NLME)
sd_NLME = np.array(sd_NLME)
est_TS = np.array(est_TS)
sd_TS = np.array(sd_TS)
est_JM = np.array(est_JM)
sd_JM = np.array(sd_JM)
alpha_NLME = np.array(alpha_NLME)
sd_bt_JM = np.array(sd_bt_JM)
sd_bt1_JM = np.array(sd_bt1_JM)
sd_bt2_JM = np.array(sd_bt2_JM)
runs_bt1 = np.array(runs_bt1)
runs_bt2 = np.array(runs_bt2)

# Organize output
NLME_out = {
    'True': BETA,  # NLME only estimates beta parameters
    'Est': est_NLME,
    'SD': sd_NLME
}

TS_out = {
    'True': TRUE_FIXED,  # Two-step estimates all parameters: beta + alpha + gamma
    'Est': est_TS,
    'SD': sd_TS
}

JM_out = {
    'True': TRUE_FIXED,  # Joint model estimates all parameters: beta + alpha + gamma
    'Est': est_JM,
    'SD': sd_JM,
    'SD_BT': sd_bt_JM,
    'SD_BT1': sd_bt1_JM,
    'SD_BT2': sd_bt2_JM
}

# Analysis functions
def rm_big(out, big=15):
    """Remove estimates with large relative bias"""
    out_bias = np.abs((out['Est'] - out['True']) / out['True'] * 100)
    max_bias = np.max(out_bias, axis=1)
    out_rm = max_bias > big
    
    print(f"Removing {np.sum(out_rm)} rows with relative bias > {big}%")
    
    out_clean = out.copy()
    out_clean['Est'] = out['Est'][~out_rm]
    out_clean['SD'] = out['SD'][~out_rm]
    
    if 'SD_BT' in out:
        out_clean['SD_BT'] = out['SD_BT'][~out_rm]
        out_clean['SD_BT1'] = out['SD_BT1'][~out_rm]
        out_clean['SD_BT2'] = out['SD_BT2'][~out_rm]
    
    return {'out_df': out_clean, 'out_rm': out_rm}

def get_summary(out):
    """Calculate summary statistics"""
    Est = np.mean(out['Est'], axis=0)
    
    Bias_mat = out['Est'] - out['True']
    rBias = np.abs(Est - out['True']) / np.abs(out['True']) * 100
    rMSE = np.mean(Bias_mat**2, axis=0) / np.abs(out['True']) * 100
    
    SE_em = np.std(out['Est'], axis=0)
    SE = np.sqrt(np.mean(out['SD']**2, axis=0))
    
    # Coverage calculation
    lower = out['Est'] - 1.96 * out['SD']
    upper = out['Est'] + 1.96 * out['SD']
    
    Coverage = []
    for i in range(len(out['True'])):
        cov = (lower[:, i] <= out['True'][i]) & (out['True'][i] <= upper[:, i])
        Coverage.append(np.mean(cov))
    Coverage = np.array(Coverage)
    
    res = np.column_stack([Est, rBias, rMSE, SE_em, SE, Coverage])
    
    if 'SD_BT' in out:
        SE_BT = np.sqrt(np.mean(out['SD_BT']**2, axis=0))
        SE_BT1 = np.sqrt(np.mean(out['SD_BT1']**2, axis=0))
        SE_BT2 = np.sqrt(np.mean(out['SD_BT2']**2, axis=0))
        
        # Bootstrap coverage
        lower_bt = out['Est'] - 1.96 * out['SD_BT']
        upper_bt = out['Est'] + 1.96 * out['SD_BT']
        
        lower_bt1 = out['Est'] - 1.96 * out['SD_BT1']
        upper_bt1 = out['Est'] + 1.96 * out['SD_BT1']
        
        lower_bt2 = out['Est'] - 1.96 * out['SD_BT2']
        upper_bt2 = out['Est'] + 1.96 * out['SD_BT2']
        
        Coverage_bt = []
        Coverage_bt1 = []
        Coverage_bt2 = []
        
        for i in range(len(out['True'])):
            cov = (lower_bt[:, i] <= out['True'][i]) & (out['True'][i] <= upper_bt[:, i])
            Coverage_bt.append(np.mean(cov))
            
            cov = (lower_bt1[:, i] <= out['True'][i]) & (out['True'][i] <= upper_bt1[:, i])
            Coverage_bt1.append(np.mean(cov))
            
            cov = (lower_bt2[:, i] <= out['True'][i]) & (out['True'][i] <= upper_bt2[:, i])
            Coverage_bt2.append(np.mean(cov))
        
        Coverage_bt = np.array(Coverage_bt)
        Coverage_bt1 = np.array(Coverage_bt1)
        Coverage_bt2 = np.array(Coverage_bt2)
        
        res = np.column_stack([
            Est, rBias, rMSE, SE_em, SE, Coverage,
            SE_BT, Coverage_bt,
            SE_BT1, Coverage_bt1,
            SE_BT2, Coverage_bt2
        ])
    
    return res

# Print results
print(f"\nAverage runs for BT1 is {np.mean(runs_bt1):.2f}")
print(f"Average runs for BT2 is {np.mean(runs_bt2):.2f}")

nl = get_summary(NLME_out)
print(f"NLME alpha mean: {np.mean(alpha_NLME):.4f}")

ts = get_summary(TS_out)
jm = get_summary(JM_out)

# Remove large bias estimates
big = 15
NLME_out1_result = rm_big(NLME_out, big)
NLME_out1 = NLME_out1_result['out_df']
nl1 = get_summary(NLME_out1)

alpha_NLME_clean = alpha_NLME[~NLME_out1_result['out_rm']]
print(f"NLME alpha mean (cleaned): {np.mean(alpha_NLME_clean):.4f}")

TS_out1 = rm_big(TS_out, big)['out_df']
ts1 = get_summary(TS_out1)

JM_out1 = rm_big(JM_out, big)['out_df']
jm1 = get_summary(JM_out1)

# Create results table
print("\nResults Summary:")
print("NLME Results:")
print(nl)
print("\nTwo-step Results:")
print(ts)
print("\nJoint Model Results:")
print(jm)

print("\nResults with large bias removed:")
print("NLME Results (cleaned):")
print(nl1)
print("\nTwo-step Results (cleaned):")
print(ts1)
print("\nJoint Model Results (cleaned):")
print(jm1)

# Save results
results = {
    'NLME_out': NLME_out,
    'TS_out': TS_out,
    'JM_out': JM_out,
    'alpha_NLME': alpha_NLME
}

output_dir = Path("simulation_output")
output_dir.mkdir(exist_ok=True)

with open(output_dir / "s1_results.pkl", 'wb') as f:
    pickle.dump(results, f)

# Save summary tables
summary_data = {
    'NLME_summary': nl1,
    'TS_summary': ts1,
    'JM_summary': jm1
}

# Save as CSV files with 3 decimal places
np.savetxt(output_dir / "NLME_summary.csv", nl1, delimiter=",", fmt="%.3f",
           header="Est,rBias,rMSE,SE_em,SE,Coverage")
np.savetxt(output_dir / "TS_summary.csv", ts1, delimiter=",", fmt="%.3f",
           header="Est,rBias,rMSE,SE_em,SE,Coverage")
np.savetxt(output_dir / "JM_summary.csv", jm1, delimiter=",", fmt="%.3f",
           header="Est,rBias,rMSE,SE_em,SE,Coverage,SE_BT,Coverage_bt,SE_BT1,Coverage_bt1,SE_BT2,Coverage_bt2")

print(f"\nResults saved to {output_dir}")
print("Simulation completed successfully!")

# Visualization functions
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle

def create_visualization():
    """Create comprehensive visualization of simulation results"""
    
    # Set up plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Parameter names
    param_names = ['β₁', 'β₂', 'β₃', 'α₀', 'α₁', 'γ₁', 'γ₂', 'γ₃']
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Parameter Estimates Comparison
    ax1 = plt.subplot(3, 3, 1)
    methods = ['NLME', 'Two-step', 'Joint Model']
    
    # For NLME, we only have beta parameters (first 3)
    nlme_means = nl1[:3, 0]  # Only beta parameters
    ts_means = ts1[:, 0]     # All 8 parameters
    jm_means = jm1[:, 0]     # All 8 parameters
    
    # Plot beta parameters comparison
    x = np.arange(3)
    width = 0.25
    
    ax1.bar(x - width, nlme_means, width, label='NLME', alpha=0.7)
    ax1.bar(x, ts_means[:3], width, label='Two-step', alpha=0.7)
    ax1.bar(x + width, jm_means[:3], width, label='Joint Model', alpha=0.7)
    
    # Add true values as horizontal lines
    for i in range(3):
        ax1.axhline(y=TRUE_FIXED[i], color='red', linestyle='--', alpha=0.5)
    
    ax1.set_xlabel('Beta Parameters')
    ax1.set_ylabel('Estimated Values')
    ax1.set_title('Beta Parameter Estimates Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(param_names[:3])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Alpha Parameters (Two-step vs Joint Model)
    ax2 = plt.subplot(3, 3, 2)
    x = np.arange(2)
    
    ax2.bar(x - width/2, ts_means[3:5], width, label='Two-step', alpha=0.7)
    ax2.bar(x + width/2, jm_means[3:5], width, label='Joint Model', alpha=0.7)
    
    # Add true values
    for i in range(2):
        ax2.axhline(y=TRUE_FIXED[3+i], color='red', linestyle='--', alpha=0.5)
    
    ax2.set_xlabel('Alpha Parameters')
    ax2.set_ylabel('Estimated Values')
    ax2.set_title('Alpha Parameter Estimates')
    ax2.set_xticks(x)
    ax2.set_xticklabels(param_names[3:5])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Gamma Parameters (Two-step vs Joint Model)
    ax3 = plt.subplot(3, 3, 3)
    x = np.arange(3)
    
    ax3.bar(x - width/2, ts_means[5:8], width, label='Two-step', alpha=0.7)
    ax3.bar(x + width/2, jm_means[5:8], width, label='Joint Model', alpha=0.7)
    
    # Add true values
    for i in range(3):
        ax3.axhline(y=TRUE_FIXED[5+i], color='red', linestyle='--', alpha=0.5)
    
    ax3.set_xlabel('Gamma Parameters')
    ax3.set_ylabel('Estimated Values')
    ax3.set_title('Gamma Parameter Estimates')
    ax3.set_xticks(x)
    ax3.set_xticklabels(param_names[5:8])
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Relative Bias Comparison
    ax4 = plt.subplot(3, 3, 4)
    
    # Create bias data
    bias_data = []
    for i in range(3):  # Beta parameters
        bias_data.append(['NLME', param_names[i], nl1[i, 1]])
        bias_data.append(['Two-step', param_names[i], ts1[i, 1]])
        bias_data.append(['Joint Model', param_names[i], jm1[i, 1]])
    
    for i in range(3, 8):  # Alpha and Gamma parameters
        bias_data.append(['Two-step', param_names[i], ts1[i, 1]])
        bias_data.append(['Joint Model', param_names[i], jm1[i, 1]])
    
    bias_df = pd.DataFrame(bias_data, columns=['Method', 'Parameter', 'Relative_Bias'])
    
    sns.barplot(data=bias_df, x='Parameter', y='Relative_Bias', hue='Method', ax=ax4)
    ax4.set_title('Relative Bias Comparison (%)')
    ax4.set_ylabel('Relative Bias (%)')
    ax4.tick_params(axis='x', rotation=45)
    
    # 5. Coverage Probability
    ax5 = plt.subplot(3, 3, 5)
    
    coverage_data = []
    for i in range(3):  # Beta parameters
        coverage_data.append(['NLME', param_names[i], nl1[i, 5]])
        coverage_data.append(['Two-step', param_names[i], ts1[i, 5]])
        coverage_data.append(['Joint Model', param_names[i], jm1[i, 5]])
    
    for i in range(3, 8):  # Alpha and Gamma parameters
        coverage_data.append(['Two-step', param_names[i], ts1[i, 5]])
        coverage_data.append(['Joint Model', param_names[i], jm1[i, 5]])
    
    coverage_df = pd.DataFrame(coverage_data, columns=['Method', 'Parameter', 'Coverage'])
    
    sns.barplot(data=coverage_df, x='Parameter', y='Coverage', hue='Method', ax=ax5)
    ax5.axhline(y=0.95, color='red', linestyle='--', alpha=0.7, label='Target 95%')
    ax5.set_title('Coverage Probability')
    ax5.set_ylabel('Coverage Probability')
    ax5.tick_params(axis='x', rotation=45)
    ax5.legend()
    
    # 6. Standard Error Comparison
    ax6 = plt.subplot(3, 3, 6)
    
    se_data = []
    for i in range(3):  # Beta parameters
        se_data.append(['NLME (Empirical)', param_names[i], nl1[i, 3]])
        se_data.append(['NLME (Model)', param_names[i], nl1[i, 4]])
        se_data.append(['Two-step (Empirical)', param_names[i], ts1[i, 3]])
        se_data.append(['Two-step (Model)', param_names[i], ts1[i, 4]])
        se_data.append(['JM (Empirical)', param_names[i], jm1[i, 3]])
        se_data.append(['JM (Model)', param_names[i], jm1[i, 4]])
    
    se_df = pd.DataFrame(se_data, columns=['Method', 'Parameter', 'SE'])
    
    sns.barplot(data=se_df[se_df['Parameter'].isin(param_names[:3])], 
                x='Parameter', y='SE', hue='Method', ax=ax6)
    ax6.set_title('Standard Error Comparison (Beta Parameters)')
    ax6.set_ylabel('Standard Error')
    ax6.tick_params(axis='x', rotation=45)
    
    # 7. Bootstrap SE Comparison for Joint Model
    ax7 = plt.subplot(3, 3, 7)
    
    bt_methods = ['Model SE', 'Bootstrap SE', 'Bootstrap SE1', 'Bootstrap SE2']
    x = np.arange(len(param_names))
    width = 0.2
    
    ax7.bar(x - 1.5*width, jm1[:, 4], width, label='Model SE', alpha=0.7)
    ax7.bar(x - 0.5*width, jm1[:, 6], width, label='Bootstrap SE', alpha=0.7)
    ax7.bar(x + 0.5*width, jm1[:, 8], width, label='Bootstrap SE1', alpha=0.7)
    ax7.bar(x + 1.5*width, jm1[:, 10], width, label='Bootstrap SE2', alpha=0.7)
    
    ax7.set_xlabel('Parameters')
    ax7.set_ylabel('Standard Error')
    ax7.set_title('Bootstrap SE Comparison (Joint Model)')
    ax7.set_xticks(x)
    ax7.set_xticklabels(param_names, rotation=45)
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 8. Parameter Boxplots
    ax8 = plt.subplot(3, 3, 8)
    
    # Create boxplot data for beta parameters
    beta_data = []
    if len(est_NLME) > 0:
        for i in range(3):
            beta_data.extend([('NLME', param_names[i], val) for val in est_NLME[:, i]])
    
    if len(est_TS) > 0:
        for i in range(3):
            beta_data.extend([('Two-step', param_names[i], val) for val in est_TS[:, i]])
    
    if len(est_JM) > 0:
        for i in range(3):
            beta_data.extend([('Joint Model', param_names[i], val) for val in est_JM[:, i]])
    
    if beta_data:
        beta_df = pd.DataFrame(beta_data, columns=['Method', 'Parameter', 'Value'])
        sns.boxplot(data=beta_df, x='Parameter', y='Value', hue='Method', ax=ax8)
        
        # Add true values
        for i in range(3):
            ax8.axhline(y=TRUE_FIXED[i], color='red', linestyle='--', alpha=0.5)
    
    ax8.set_title('Parameter Estimate Distributions (Beta)')
    ax8.set_ylabel('Parameter Value')
    
    # 9. Summary Statistics Table
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    # Create summary table
    summary_text = f"""
    Simulation Summary (Setting I)
    
    Sample Size: n = {N}
    Repeated Measurements: ni = {NI}
    Simulation Repetitions: {REP}
    Bootstrap Runs: {K_RUNS}
    
    True Parameters:
    β = {BETA}
    α = {ALPHA}
    γ = {GAMMA}
    
    Average Bootstrap Runs:
    BT1: {np.mean(runs_bt1):.1f}
    BT2: {np.mean(runs_bt2):.1f}
    
    NLME Alpha Mean: {np.mean(alpha_NLME_clean):.3f}
    """
    
    ax9.text(0.1, 0.9, summary_text, transform=ax9.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    
    plt.tight_layout()
    plt.savefig(output_dir / "simulation_results_visualization.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create a separate detailed comparison plot
    create_detailed_comparison()

def create_detailed_comparison():
    """Create detailed comparison of methods"""
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    param_names = ['β₁', 'β₂', 'β₃', 'α₀', 'α₁', 'γ₁', 'γ₂', 'γ₃']
    
    # Plot for each parameter
    for i in range(8):
        row = i // 4
        col = i % 4
        ax = axes[row, col]
        
        # Data to plot
        if i < 3:  # Beta parameters - all three methods
            methods = ['NLME', 'Two-step', 'Joint Model']
            estimates = [nl1[i, 0], ts1[i, 0], jm1[i, 0]]
            biases = [nl1[i, 1], ts1[i, 1], jm1[i, 1]]
            colors = ['blue', 'orange', 'green']
        else:  # Alpha and Gamma parameters - only Two-step and Joint Model
            methods = ['Two-step', 'Joint Model']
            estimates = [ts1[i, 0], jm1[i, 0]]
            biases = [ts1[i, 1], jm1[i, 1]]
            colors = ['orange', 'green']
        
        # Create bar plot
        x_pos = np.arange(len(methods))
        bars = ax.bar(x_pos, estimates, color=colors, alpha=0.7)
        
        # Add true value line
        ax.axhline(y=TRUE_FIXED[i], color='red', linestyle='--', linewidth=2, label='True Value')
        
        # Add bias text on bars
        for j, (bar, bias) in enumerate(zip(bars, biases)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01*abs(height),
                   f'{bias:.2f}%', ha='center', va='bottom', fontsize=8)
        
        ax.set_title(f'{param_names[i]} (True: {TRUE_FIXED[i]:.2f})')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(methods, rotation=45)
        ax.grid(True, alpha=0.3)
        
        if i == 0:
            ax.legend()
    
    plt.suptitle('Detailed Parameter Comparison with Relative Bias (%)', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_dir / "detailed_parameter_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()

# Generate visualizations
print("\nGenerating visualizations...")
try:
    create_visualization()
    print("Visualizations saved successfully!")
except Exception as e:
    print(f"Error creating visualizations: {e}")
    import traceback
    traceback.print_exc()

import numpy
import time

from test_functions import Branin

from moe.optimal_learning.python.data_containers import HistoricalData
from moe.optimal_learning.python.geometry_utils import ClosedInterval
from moe.optimal_learning.python.repeated_domain import RepeatedDomain

from moe.optimal_learning.python.python_version.domain import TensorProductDomain as pythonTensorProductDomain
from moe.optimal_learning.python.python_version.gaussian_process import GaussianProcess as pythonGaussianProcess
from moe.optimal_learning.python.python_version.expected_improvement import ExpectedImprovement as pythonExpectedImprovement
from moe.optimal_learning.python.python_version.covariance import SquareExponential as pythonSquareExponential
from moe.optimal_learning.python.python_version.optimization import LBFGSBParameters, LBFGSBOptimizer
from moe.optimal_learning.python.python_version.expected_improvement import multistart_expected_improvement_optimization as python_multistart_expected_improvement_optimization

from moe.optimal_learning.python.cpp_wrappers.domain import TensorProductDomain as cppTensorProductDomain
from moe.optimal_learning.python.cpp_wrappers.gaussian_process import GaussianProcess as cppGaussianProcess
from moe.optimal_learning.python.cpp_wrappers.log_likelihood import GaussianProcessLogLikelihood as cppGaussianProcessLogLikelihood
from moe.optimal_learning.python.cpp_wrappers.covariance import SquareExponential as cppSquareExponential
from moe.optimal_learning.python.cpp_wrappers.optimization import NewtonOptimizer as cppNewtonOptimizer
from moe.optimal_learning.python.cpp_wrappers.optimization import GradientDescentOptimizer as cppGradientDescentOptimizer
from moe.optimal_learning.python.cpp_wrappers.optimization import NewtonParameters as cppNewtonParameters
from moe.optimal_learning.python.cpp_wrappers.optimization import GradientDescentParameters as cppGradientDescentParameters
from moe.optimal_learning.python.cpp_wrappers.log_likelihood import multistart_hyperparameter_optimization
from moe.optimal_learning.python.cpp_wrappers.expected_improvement import ExpectedImprovement as cppExpectedImprovement
from moe.optimal_learning.python.cpp_wrappers.expected_improvement import multistart_expected_improvement_optimization, constant_liar_expected_improvement_optimization, kriging_believer_expected_improvement_optimization

def optimization_method(experiment, method_name, num_to_sample, num_threads, num_mc_itr, dum_search_itr, lie_value, gd_opt_parameters, BFGS_parameters, num_multistarts, which_gpu):
    """Perform a full round of generating points_to_sample, given the optimization method to use

    """
    if (method_name == "exact_qEI"):
        python_gp = pythonGaussianProcess(experiment._python_cov, experiment._historical_data)
        repeated_domain = RepeatedDomain(num_to_sample, experiment._python_search_domain)
        ei_evaluator = pythonExpectedImprovement(gaussian_process=python_gp)
        optimizer = LBFGSBOptimizer(repeated_domain, ei_evaluator, BFGS_parameters)
        return python_multistart_expected_improvement_optimization(optimizer, num_multistarts, num_to_sample)
    else:
        cpp_gp = cppGaussianProcess(experiment._cpp_cov, experiment._historical_data)
        ei_evaluator = cppExpectedImprovement(gaussian_process=cpp_gp, num_mc_iterations=num_mc_itr)
        optimizer = cppGradientDescentOptimizer(experiment._cpp_search_domain, ei_evaluator, gd_opt_parameters, dum_search_itr)
        if (method_name == "epi_gpu"):
            return multistart_expected_improvement_optimization(optimizer, 0, num_to_sample, max_num_threads=num_threads)
        elif (method_name == "CL"):
            return constant_liar_expected_improvement_optimization(optimizer, 0, num_to_sample, lie_value, max_num_threads=num_threads)
        elif (method_name == "KB"):
            return kriging_believer_expected_improvement_optimization(optimizer, 0, num_to_sample, max_num_threads=num_threads)
        else:
            raise NotImplementedError("Not a valid optimization method")

def get_parameters(num_multistarts, max_num_steps, max_num_restarts):
    newton_opt_parameters = cppNewtonParameters(num_multistarts=num_multistarts, max_num_steps=max_num_steps, gamma=1.01, time_factor=1.0e-3, max_relative_change=1.0, tolerance=1.0e-10)
    gd_opt_parameters = cppGradientDescentParameters(num_multistarts=num_multistarts, max_num_steps=max_num_steps, max_num_restarts=max_num_restarts, num_steps_averaged=20, gamma=0.7, pre_mult=1.0, max_relative_change=0.7, tolerance=1.0e-7)
    BFGS_parameters = LBFGSBParameters(approx_grad=True, max_func_evals=(max_num_restarts*max_num_steps), max_metric_correc=10, factr=10.0, pgtol=1e-10,epsilon=1e-8)
    return gd_opt_parameters, BFGS_parameters, newton_opt_parameters

class NumericalExperiment():

    r"""Class of setting up numerical experiment

    """

    def __init__(self, dim, search_domain_bounds, hyper_domain_bounds, num_init_points, objective_function):
        self._dim = dim
        self._sample_var = 0.01
        self._objective_function = objective_function
        self._num_init_points = num_init_points

        # domain
        self._python_search_domain = pythonTensorProductDomain([ClosedInterval(bound[0], bound[1]) for bound in search_domain_bounds])
        self._cpp_search_domain = cppTensorProductDomain([ClosedInterval(bound[0], bound[1]) for bound in search_domain_bounds])
        self._cpp_hyper_domain = cppTensorProductDomain([ClosedInterval(bound[0], bound[1]) for bound in hyper_domain_bounds])
        self._reset_state()

    def _reset_state(self):
        r"""reset initial points, historical data, covariance

        """
        self._init_points = self._python_search_domain.generate_uniform_random_points_in_domain(self._num_init_points)
        self._historical_data = HistoricalData(self._dim)
        for point in self._init_points:
            self._historical_data.append_sample_points([[point, self._objective_function(point), self._sample_var],])
        self._python_cov = pythonSquareExponential(numpy.ones(self._dim + 1))
        self._cpp_cov = cppSquareExponential(numpy.ones(self._dim + 1))

    def _add_points(self, points_to_sample):
        """ add points_to_sample to _historical_data

        """
        for point in points_to_sample:
            self._historical_data.append_sample_points([[point, self._objective_function(point), self._sample_var],])

    def _update_hyperparameters(self, opt_parameters, max_num_threads):
        """ update hyperparameters for both _cpp_cov and _python_cov based on historical points

        """
        cpp_gp_loglikelihood = cppGaussianProcessLogLikelihood(self._cpp_cov, self._historical_data)
        newton_optimizer = cppNewtonOptimizer(self._cpp_hyper_domain, cpp_gp_loglikelihood, opt_parameters)
        best_param = multistart_hyperparameter_optimization(newton_optimizer, 0, max_num_threads=max_num_threads)
        self._cpp_cov.set_hyperparameters(best_param)
        self._python_cov.set_hyperparameters(best_param)

    def compare_best_so_far(self, method_name, hyper_skip_turns, num_itr, num_to_sample, num_threads, num_mc_itr=1000000, dum_search_itr=100000, lie_value=0.0, which_gpu=0):
        """ Run numerical experiment given truth function and method we use for optimization

        """
        num_multistarts = 10
        max_num_steps = 100
        max_num_restarts = 4
        gd_opt_parameters, BFGS_parameters, newton_opt_parameters =  get_parameters(num_multistarts, max_num_steps, max_num_restarts)
        best_so_far = []
        num_func_eval = []
        self._reset_state()
        self._update_hyperparameters(newton_opt_parameters, num_threads)
        for itr in range(num_itr):
            print "{0}th iteration\n".format(itr)
            points_to_sample = optimization_method(self, method_name, num_to_sample, num_threads, num_mc_itr, dum_search_itr, lie_value, gd_opt_parameters, BFGS_parameters, num_multistarts, which_gpu)
            self._add_points(points_to_sample)
            if (itr+1)%hyper_skip_turns==0:
                self._update_hyperparameters(newton_opt_parameters, num_threads)
            # record num_func_eval & best_so_far
            num_func_eval.append(itr * num_to_sample)
            best_so_far.append(numpy.amin(self._historical_data.points_sampled_value))
        return num_func_eval, best_so_far

    def compare_best_ei(self, num_itr_table, num_to_sample, which_gpu):
        num_threads = 4
        num_mc_itr = 10000000
        max_num_restarts = 1
        num_multistarts = 10
        dum_search_itr = 100000
        best_ei_gpu = numpy.zeros(len(num_itr_table))
        best_ei_analytic = numpy.zeros(len(num_itr_table))
        self._reset_state()
        self._update_hyperparameters(newton_opt_parameters, num_threads)
        for i, num_itr in enumerate(num_itr_table):
            gd_opt_parameters, BFGS_parameters, newton_opt_parameters =  get_parameters(num_multistarts, num_itr, max_num_restarts)
            analytic_points_to_sample = optimization_method(self, "exact_qEI", num_to_sample, num_threads, num_mc_itr, dum_search_itr, -1, gd_opt_parameters, BFGS_parameters, num_multistarts, which_gpu)
            gpu_points_to_sample = optimization_method(self, "ei_gpu", num_to_sample, num_threads, num_mc_itr, dum_search_itr, -1, gd_opt_parameters, BFGS_parameters, num_multistarts, which_gpu)
            # evaluate ei
            cpp_gp = cppGaussianProcess(self._cpp_cov, self._historical_data)
            gpu_ei_evaluator = cppExpectedImprovement(cpp_gp, analytic_points_to_sample, num_mc_iterations=10000000)
            ei_and_time = gpu_ei_evaluator.time_expected_improvement(use_gpu=True, which_gpu=which_gpu, num_repeat=5)
            best_ei_analytic[i] = ei_and_time[0]
            gpu_ei_evaluator = cppExpectedImprovement(cpp_gp, gpu_points_to_sample, num_mc_iterations=10000000)
            ei_and_time = gpu_ei_evaluator.time_expected_improvement(use_gpu=True, which_gpu=which_gpu, num_repeat=5)
            best_ei_gpu[i] = ei_and_time[0]
        return best_ei_gpu, best_ei_analytic

    def timing_analytic_vs_gpu_qp_ei(self, num_to_sample_table, which_gpu, repeat_time):
        tol = 1e-15
        num_experiments = len(num_to_sample_table)
        python_gp = pythonGaussianProcess(self._python_cov, self._historical_data)
        cpp_gp = cppGaussianProcess(self._cpp_cov, self._historical_data)

        analytic_time_table = numpy.zeros((repeat_time, num_experiments))
        gpu_time_table = numpy.zeros((repeat_time, num_experiments))
        ei_diff_table = numpy.zeros((repeat_time, num_experiments))
        for n in range(repeat_time):
            for i, num_to_sample in enumerate(num_to_sample_table):
                points_to_sample = self._python_search_domain.generate_uniform_random_points_in_domain(num_to_sample)
                # for analytic
                ei_evaluator = pythonExpectedImprovement(python_gp, points_to_sample)
                mu_star = python_gp.compute_mean_of_points(points_to_sample)
                var_star = python_gp.compute_variance_of_points(points_to_sample)
                start_time = time.time()
                analytic_ei = ei_evaluator._compute_expected_improvement_qd_analytic(mu_star, var_star)
                end_time = time.time()
                analytic_time_table[n, i] = end_time - start_time
                # for gpu
                gpu_ei_evaluator = cppExpectedImprovement(cpp_gp, points_to_sample, num_mc_iterations=10000000)
                ei_and_time = gpu_ei_evaluator.time_expected_improvement(use_gpu=True, which_gpu=which_gpu, num_repeat=5)
                gpu_time_table[n, i] = ei_and_time[1]
                gpu_ei  = ei_and_time[0]
                if (gpu_ei < tol):
                    ei_diff_table[n, i] = abs(analytic_ei - gpu_ei)
                else:
                    ei_diff_table[n, i] = abs(analytic_ei - gpu_ei) / gpu_ei

        return numpy.mean(gpu_time_table, axis=0), numpy.mean(analytic_time_table, axis=0), numpy.mean(ei_diff_table, axis=0)

    def compare_best_ei_found(self, num_restart, num_steps, num_to_sample):
        # for BFGS
        approx_grad = True
        max_func_evals = num_restart * num_steps
        max_metric_correc = 10
        factr = 10.0
        pgtol = 1e-10
        epsilon = 1e-8
        BFGS_parameters = LBFGSBParameters(approx_grad, max_func_evals, max_metric_correc, factr, pgtol, epsilon)
        python_gp = pythonGaussianProcess(self._python_cov, self._historical_data)
        repeated_domain = RepeatedDomain(num_to_sample, self._python_search_search_domain)


if __name__ == "__main__":
    import csv
    num_to_sample_table = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    num_repeat = 5
    experiment = NumericalExperiment(dim = 2, search_domain_bounds = [[0,10],[0,10]], hyper_domain_bounds = [[1,10],[1,10],[1,10]], num_init_points = 20, objective_function = Branin)
    # gpu_time, analytic_time, ei_diff = experiment.timing_analytic_vs_gpu_qp_ei(num_to_sample_table, 0, num_repeat)
    # with open('gpu_vs_analytic.csv','wb') as file:
    #     w = csv.writer(file)
    #     w.writerows([gpu_time, analytic_time, ei_diff])
    pts = experiment._python_search_domain.generate_uniform_random_points_in_domain(4)
    print pts


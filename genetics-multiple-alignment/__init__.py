# coding: utf-8
""" Manager d'Etudes de Reconstruction d'Images (MERI) is a package related to
parameter/outputs management and image error measurement in a context of
MRI research.
"""
# system import
import sys
import itertools
import psutil
import humanize
import numpy as np
from joblib import Parallel, delayed


try:
    import pisap

    def wrapper_isap(recons_func, **kwargs):
        """ Helper to parallelize the reconstruction.
        """
        res = recons_func(**kwargs)
        if isinstance(res, pisap.base.image.Image):
            return res
        elif isinstance(res, tuple):
            return res[0]
        else:
            raise ValueError(
                "res of 'recons_func' not understood: got {0}".format(type(res)))

    def wrap_isap_condat_im_metrics_func(metrics_funcs, img, ref):
        """ Helper to parallelize the metrics computations for tuple
        (pisap.base.image.Image, np.ndarray) grid_search ouputs.
        """
        if type(img) == tuple:
            sanytize_img = img[0].data
        return _wrap_im_metrics_func(metrics_funcs, sanytize_img, ref)

    def wrap_isap_fista_im_metrics_func(metrics_funcs, img, ref):
        """ Helper to parallelize the metrics computations for tuple
        (pisap.base.image.Image, np.ndarray) grid_search ouputs.
        """
        if isinstance(img, pisap.base.image.Image):
            sanytize_img = img.data
        return _wrap_im_metrics_func(metrics_funcs, sanytize_img, ref)

except ImportError:
    print("Importing pisap failed: wrapper_isap and wrap_isap_im_metrics_func"
          "not defined")
    pass


def _default_wrapper(recons_func, **kwargs):
    """ Default wrapper to parallelize the image reconstruction.
    """
    return recons_func(**kwargs)


def _wrap_im_metrics_func(metrics_funcs, img, ref):
    """ Helper to parallelize the metrics computations.
    """
    if callable(metrics_funcs):
        metrics_funcs = [metrics_funcs]
    err = {}
    for metrics_func in metrics_funcs:
        err[metrics_func.func_name] = metrics_func(img, ref)
    return err


def _get_final_size(param_grid):
    """ Return the memory size of the given param_grid when it will extend as
    a carthesian grid a parameters.

    Parameters:
    ----------
    param_grid: dict or list of dictionaries,
        Dictionary with parameters names (string) as keys and lists of
        parameter settings to try as values.

    Return:
    -------
    size: int,
        the number of bytes of the extended carthesian grid a parameters.
    """
    tmp = {} # same pattern than param_grid but store the size
    for idx, key in enumerate(param_grid.iterkeys()):
        if isinstance(param_grid[key], list):
            tmp[idx] = [sys.getsizeof(value) for value in param_grid[key]]
        else:
            tmp[idx] = [sys.getsizeof(param_grid[key])]
    return np.array([x for x in itertools.product(*tmp.values())]).sum()


def grid_search(func, param_grid, wrapper=None, n_jobs=1, verbose=0):
    """ Run `func` on the carthesian product of `param_grid`.

        Parameters:
        -----------
        func: function,
            The reconstruction function from whom to tune the hyperparameters.
            `func` return should be handle by wrapper if it's not a
            simple np.ndarray image.
        param_grid: dict or list of dictionaries,
            Dictionary with parameters names (string) as keys and lists of
            parameter settings to try as values: the grids spanned by each
            dictionary in the list are explored.
        wrapper: function, (default: None)
            Handle the call of func if some pre-process or post-process
            should be done. `wrapper` has a specific API:
            `wrapper(func, **kwargs)`
        n_jobs: int (default: 1),
            The maximum number of concurrently running jobs, such as the number
            of Python worker processes when backend=”multiprocessing” or the
            size of the thread-pool when backend=”threading”. If -1 all CPUs
            are used. If 1 is given, no parallel computing code is used at all,
            which is useful for debugging. For n_jobs below -1,
            (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2,
            all CPUs but one are used.
        verbose: int (default: 0),
            The verbosity level: if non zero, progress messages are printed.
            Above 50, the output is sent to stdout. The frequency of the
            messages increases with the verbosity level. If it more than 10,
            all iterations are reported.

        Results:
        --------
        report: class,
            A reporting class containing all the reconstructed image, the given
            parameters, the error measurement, and differents methods to select
            the best parameters w.r.t a specific metrics.
    """
    if wrapper is None:
        wrapper = _default_wrapper
    # check if enough memory
    size_ = _get_final_size(param_grid)
    if size_ > 0.9 * psutil.virtual_memory().available:
        raise MemoryError("not enough memory 'param_grid'"
                          " weigh {0} ..".format(humanize.naturalsize(size_)))
    # sanitize value to list type
    for key, value in param_grid.iteritems():
        if not isinstance(value, list):
            param_grid[key] = [value]
    list_kwargs = [dict(zip(param_grid, x))
                   for x in itertools.product(*param_grid.values())]
    # Run the reconstruction
    if verbose > 0:
        if n_jobs == -1:
            n_jobs_used = psutil.cpu_count()
        elif n_jobs == -2:
            n_jobs_used = psutil.cpu_count() - 1
        else:
            n_jobs_used = n_jobs
        print(("Running grid_search for {0} candidates"
               " on {1} jobs").format(len(list_kwargs), n_jobs_used))
    res = Parallel(n_jobs=n_jobs, verbose=verbose)(
                   delayed(wrapper)(func, **kwargs)
                   for kwargs in list_kwargs)
    return list_kwargs, res


def compute_multiple_metrics(imgs, im_ref, metrics_funcs, _wrapper=None,
                             n_jobs=1, verbose=0):
    """
    Parameters:
    ----------
        imgs: list of pisap.Image,
            Images on which to compute the metrics.
        im_ref: np.ndarray
            This image reference for computing the error for each functions in
            metrics_funcs.
        metrics_funcs: list of functions,
            The list of functions for the error measurement. Each one should
            only accept two arguments: im and ref and each function should
            return a real number.
        n_jobs: int (default: 1),
            The maximum number of concurrently running jobs, such as the number
            of Python worker processes when backend=”multiprocessing” or the
            size of the thread-pool when backend=”threading”. If -1 all CPUs
            are used. If 1 is given, no parallel computing code is used at all,
            which is useful for debugging. For n_jobs below -1,
            (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2,
            all CPUs but one are used.
        _wrapper: function, (default: None)
            Handle the call of func if some pre-process or post-process
            should be done. `_wrapper` has a specific API:
            `_wrapper(metrics_funcs, im.data, im_ref)`
        verbose: int (default: 0),
            The verbosity level: if non zero, progress messages are printed.
            Above 50, the output is sent to stdout. The frequency of the
            messages increases with the verbosity level. If it more than 10,
            all iterations are reported.
    """
    if _wrapper is None:
        _wrapper = _wrap_im_metrics_func
    errs = Parallel(n_jobs=n_jobs, verbose=verbose)(
           delayed(_wrapper)(metrics_funcs, im, im_ref)
           for im in imgs)
    return errs


class ReportGridSearch(object):
    """A reporting class generated by the function grid_search containing all
    the reconstructed image, the given parameters, the error measurement,
    and differents methods to select the best parameters w.r.t a specific
    metrics.
    """
    def __init__(self, list_kwargs, param_grid, recons_im, im_ref,
                 metrics_funcs, metrics_direction, errs):
        """ Init the class.

        Parameters:
        ----------
        list_kwargs: list of dictionary,
            the list of all the 'tested' parameter for the reconstruction.
        list_params: list of dictionary,
            the list of all the 'tested' parameter for the reconstruction.
        param_grid: dict or list of dictionaries,
            Dictionary with parameters names (string) as keys and lists of
            parameter settings to try as values.
        recons_im: list of np.ndarray
            the list of the reconstructed image. It should respect the same
            order than list_kwargs.
        im_ref: np.ndarray;
            reference image which were used to compute the errs.
        metrics_funcs: list of functions,
            The list of functions for the error measurement. Each one should
            only accept two arguments: im and ref and each function should
            return a real number.
        metrics_direction: list of bool,
            specify if the metrics mean: True if the lower the better, False for
            the greater the better. It will be directly pass to the report
            result.
        errs: list of dictionary
            the list of all the metrics errors for each reconstructed image. It
            should respect the same order than list_kwargs (and so recons_im).
        """
        if len(metrics_funcs) != len(metrics_direction):
            raise ValueError("metrics_funcs and metrics_direction should be the "
                             "same size")
        self.list_kwargs = list_kwargs
        self.param_grid = param_grid
        self.recons_im = recons_im
        self.im_ref = im_ref
        self.metrics_funcs = metrics_funcs
        metrics_direction = dict(zip([func.func_name for func in metrics_funcs],
                                 metrics_direction))
        self.metrics_direction = metrics_direction
        self.errs = errs
        self.fixed_params = self._get_fixed_params()
        self.floatting_params = self._get_floatting_params()

    ####
    ## getter methods

    def get_list_params(self, param_name):
        """ Getter for the list of value submitted for the given parameter.

        Parameters:
        -----------
        param_name: str,
            the parameter name of desired parameter.

        Return:
        -------
        params: list
            list of the values for the given parameter.
        """
        return self.param_grid[param_name]

    def _get_fixed_params(self):
        """ Private helper that return the list of parameters name that have a
        single value.

        Return:
        -------
        fixed_params: list of str
            list of parameters name that have a single value.
        """
        return [key for key, value in self.param_grid.iteritems()
                if len(value) == 1]

    def _get_floatting_params(self):
        """ Private helper that return the list of parameters name that have a
        single value.

        Return:
        -------
        fixed_params: list of str
            list of parameters name that have a single value.
        """
        return [key for key, value in self.param_grid.iteritems()
                if len(value) > 1]

    def _get_studies_filter(self, selector):
        """ Private helper that return index list based on filter in kwargs.

        Parameters:
        -----------
        selector:

        Return:
        -------
        index_subset: list of int,
            mask to consider only the studies which the index is in selector .
        """
        if selector is None:
            idx_filter = range(len(self.recons_im))
        elif isinstance(selector, list):
            idx_filter = selector
        elif isinstance(selector, dict):
            # sanitize selector dict
            for key, value in selector.iteritems():
                if not isinstance(value, list):
                    selector[key] = [value]
            idx_filter = []
            for idx, kwargs in enumerate(self.list_kwargs):
                drop = False # if True will ignore this study
                # loop to check if the study match the new restricted
                # parameters grid
                for key, value in kwargs.iteritems():
                    if (key in self.fixed_params) or (key not in selector):
                        continue
                    if value not in selector[key]:
                        drop = True
                if not drop:
                    idx_filter.append(idx)
        else:
            raise ValueError("selector type not understood, "
                             "got {0}".format(type(selector)))
        return idx_filter

    def _all_score(self, metric, index_subset):
        """ Private helper that return all the score for the given metric.

        Parameters:
        -----------
        metric: string or function,
            the exact metric function name, example: 'compare_mse'
            from metric or the metric function itself.

        index_subset: list of int,
            mask to consider only the studies which the index is in index_subset .

        Return:
        ------
        all_score: np.ndarray,
            all the measure error for the given metric, respect the order same
            order than list_kwargs, recons_im and errs attributes.
        """
        if index_subset is None:
            index_subset = range(len(self.recons_im))
        if callable(metric):
            metric = metric.func_name
        return np.array([(errs[metric], idx) for idx, errs in enumerate(self.errs)
                         if idx in index_subset])

    ####
    ## best getter methods

    def best_image(self, metric, selector=None):
        """ Return the best reconstructed image for the given metric.

        Parameters:
        -----------
        metric: string or function
            the exact metric function name, example: 'compare_mse'
            from metric or the metric function itself.

        selector: dict,
            params_grid, where the fixed and floatting parameters are specify.

        Return:
        -------
        best_image: np.ndarray,
            the best reconstructed image for the given metric.
        """
        index_subset = self._get_studies_filter(selector)
        best_idx = self.best_index(metric, index_subset)
        return self.recons_im[best_idx]

    def best_score(self, metric, selector=None):
        """ Return the best score for the given metric.

        Parameters:
        -----------
        metric: string or function
            the exact metric function name, example: 'compare_mse'
            from metric or the metric function itself.

        selector: dict,
            params_grid, where the fixed and floatting parameters are specify.

        Return:
        -------
        best_score: float,
            the best score for the given metric.
        """
        index_subset = self._get_studies_filter(selector)
        best_idx = self.best_index(metric, index_subset)
        if callable(metric):
            metric = metric.func_name
        return self.errs[best_idx][metric]

    def best_params(self, metric, selector=None):
        """ Return the best set of parameters for the given metric.

        Parameters:
        -----------
        metric: string or function
            the exact metric function name, example: 'compare_mse'
            from metric or the metric function itself.

        selector: dict,
            params_grid, where the fixed and floatting parameters are specify.

        Return:
        -------
        best_params: dictionary,
            the best params for the given metric.
        """
        index_subset = self._get_studies_filter(selector)
        best_idx = self.best_index(metric, index_subset)
        return self.list_kwargs[best_idx]

    def best_index(self, metric, selector=None):
        """ Return the index of the best set of parameters for the given metric.

        Parameters:
        -----------
        metric: string or function
            the exact metric function name, example: 'compare_mse'
            from metric or the metric function itself.

        selector: dict,
            params_grid, where the fixed and floatting parameters are specify.

        Return:
        -------
        best_index: int,
            the index of the best set of parameters.
        """
        if callable(metric):
            metric = metric.func_name
        index_subset = self._get_studies_filter(selector)
        # the subset of desired scores
        scores_ = self._all_score(metric, index_subset)
        if self.metrics_direction[metric]:
            return int(scores_[np.nanargmin(scores_[:,0]), 1])
        else:
            return int(scores_[np.nanargmax(scores_[:,0]), 1])

    ####
    ## diff methods

    def diffref_best_image(self, metric, selector=None):
        """ Return the difference between the reference image and the
        best image for the given metric

        Parameters:
        -----------
        metric: string or function
            the exact metric function name, example: 'compare_mse'
            from metric or the metric function itself.

        selector: dict,
            params_grid, where the fixed and floatting parameters are specify.

        Return:
        -------
        diff: np.ndarray,
            the difference, recons_im[idx] - im_ref, between the reference
            image and the best image for the given metric.
        """
        return self.best_image(metric, selector) - self.im_ref

    def diffref_image(self, idx):
        """ Return the difference between the reference image and the
        image with ithe given idx w.r.t to the order of list_kwargs.

        Parameters:
        -----------
        idx: int,
            the idx of recons image to inspect w.r.t to the order of list_kwargs.

        Return:
        -------
        diff: np.ndarray,
            the difference, recons_im[idx] - im_ref, between the reference
            image and the image with the given idx w.r.t to the order of
            list_kwargs.
        """
        return self.recons_im[idx] - self.im_ref


    def diff_image(self, idx, idxref):
        """ Return the difference between the images with the given idx w.r.t
        to the order of list_kwargs.

        Parameters:
        -----------
        idx: int,
            the idx of recons image to inspect w.r.t to the order of list_kwargs.
        idxref: int,
            the idx of recons image considered as ref  w.r.t to the order of
            list_kwargs.

        Return:
        -------
        diff: np.ndarray,
            the difference, recons_im[idx] - im_ref, between the reference
            image and the image with the given idx w.r.t to the order of
            list_kwargs.
        """
        return self.recons_im[idx] - self.recons_im[idxref]

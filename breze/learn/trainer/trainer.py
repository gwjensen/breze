
"""Module that contains various functionality for trainers."""

import datetime
import time

from climin import mathadapt as ma
from climin.stops import never, always
from climin.util import clear_info

import score as score_
import report as report_

import signal

class Trainer(object):
    """Class representing a Trainer.
    A Trainer object is used to ease bookkeeping of fitting models. This is done
    by composing a trainer out of several basic strategies.
    `Scoring strategy`: The way the score is calculated can be determined by
    the callable stored in the ``_score`` field. For some examples, see the
    module ``breze.learn.trainer.score``.
    `Reporting strategy`: For a report function that is applied to each info
    dictionary during a pause when calling ``.fit()``, ``.report`` can be set.
    For examples, see the module ``breze.learn.trainer.report``.
    `Pause criterion`: When to pause optimization of the model to yield
    control back to the user. Determined by the ``.pause`` field. Contains a
    callable, for example see ``climin.stops``.
    `Interrupt criterion`: When to interrupt optimization of the model to
    yield control back to the user. Determined by the ``.interrupt`` field.
    Contains a callable, for example see ``climin.stops``.
    `Stopping criterion`: When to interrupt optimization of the model to
    yield control back to the user. Determined by the ``.stop`` field.
    Contains a callable, for example see ``climin.stops``.
    Why do we need separate stopping and interrupting criteria? An
    optimization might get interrupted (e.g. by a SIGINT of a shared resource
    system). In order to find out whether the trainer thinks optimization has
    actually finished, the ``.stopped`` field is provided.
    Attributes
    ----------
    model : Model object
        Model that is going to be trained by this trainer.
    _score : callable
        Callable that applies a score function to data. Signature is
        ``f_score, *data``.
    pause : callable
        Callable that given a climin info dictionary determines whether to pause
        fitting.
    stop : callable
        Callable that given a climin info dictionary determines whether to stop
        (i.e. finish) fitting.
    interrupt : callable
        Callable that given a climin info dictionry determines whether to
        interrupt fitting.
    report : callable
        Callable to which the info dictionary of the current optimization is
        passed during each pause.
    best_pars : array_like
        Currently best found parameters according to validation data.
    best_loss : float
        Loss on the validation data of ``best_pars``.
    infos : list of dicts
        List containing all info dictionaries of the estimation.
    current_info : dict
        Last info dictionary.
    data : dictionary
        Dictionary of different data sets for evaluation.
    val_key : string
        Key identifying the data set from ``data`` which is used for
        validation.
    stopped : boolean
        If ``stop`` has returned True once, this is set to True. Otherwise
        False. Useful for distinguishing between interrupt and stop.
    """

    def __init__(self, model, data, stop, score=score_.simple,
                 pause=always, interrupt=never, report=report_.point_print, info_opt=None):
        """Create a Trainer object.
        Parameters
        ----------
        model : Model object
            Model that is going to be trained by this trainer.
        data : dict
            Dictionary with the different data parts.
        stop : callable
            Callable that given a climin info dictionary determines whether to stop
            (i.e. finish) fitting.
        score : callable, optional
            Callable that applies a score function to data. Signature is
            ``f_score, *data``.
        pause : callable, optional
            Callable that given a climin info dictionary determines whether to pause
            fitting.
        interrupt : callable, optional
            Callable that given a climin info dictionry determines whether to
            interrupt fitting.
        report : callable, optional
            Callable to which the info dictionary of the current optimization is
            passed during each pause.
        """

        self.model = model
        self.data = data

        self._score = score
        self.pause = pause
        self.stop = stop
        self.interrupt = interrupt
        self.report = report

        self.best_pars = None
        self.best_loss = float('inf')
        self.runtime = 0

        self.infos = []
        self.current_info = info_opt

        self.train_losses = []

        self.val_key = 'val' # None, set from outside?
        self.info_keys = []

        self.stopped = False

    def score(self, *data):
        return self._score(self.model.score, *data)

    def fit(self):
        """Run ``.iter_fit()`` until it terminates
        Termination will occur when either stop or interrupt is True. During
        each pause, ``.report(info)`` will be executed."""
        for i in self.iter_fit(*self.data['train']):
            #print "trainer.py:fit: " + str(i)
            self.report(i)

    def switch_pars(self, pars):
        old = self.model.parameters.data.copy()
        self.model.parameters.data[...] = pars
        return old

    def iter_fit(self, *fit_data):
        """Iteratively fit the given training data.
        Generator function containing the main logic of the Trainer object.
        The arguments are of variable length and have to match that of the
        ``model.iter_fit()`` and ultimately the used loss function of that
        model.
        Each iteration of the fitting constitutes of running the optimizer of
        the model until either interrupt or pause returns True.
        In both cases, the generator will yield to the user. Additionally:
            - If interrupt returns True, the generator will stop yielding
            values afterwards.
            - stop will be tested. If it is true it will stop yielding
            afterwards and additionally ``.stopped`` will be set to True
            afterwards.
            - ``best_pars`` and ``best_loss`` will be updated.
        The values yielded from this function will be climin info dictionaries
        stripped from any numpy or gnumpy arrays.
        """

        self.CTRL_C_FLAG = False
        signal.signal(signal.SIGINT, self._ctrl_c_handler)

        start = time.time()

        for info in self.model.iter_fit(*fit_data, info_opt=self.current_info):
            #print "trainer.py:iter_fit: " + str(info)
            if "cur_batch" in self.info_keys:
                self.model.training = 0
                self.train_losses.append(ma.scalar(self.score(*info["args"])))
                self.model.training = 1

            interrupt = self.interrupt(info)
            if self.pause(info) or interrupt or self.CTRL_C_FLAG:
                self.model.training = 0
                info['val_loss'] = ma.scalar(self.score(*self.data[self.val_key]))

                for i in self.info_keys:
                    if i == "cur_batch":
                        info["train_loss"] = self.train_losses
                        self.train_losses = []
                    else:
                        info['{}_loss'.format(i)] = ma.scalar(
                            self.score(*self.data[i])
                        )

                cur_val_loss = info['%s_loss' % self.val_key]
                if cur_val_loss < self.best_loss:
                    self.best_loss = cur_val_loss
                    self.best_pars = self.model.parameters.data.copy()

                self.runtime += time.time() - start
                info.update({
                    'best_loss': self.best_loss,
                    'best_pars': self.best_pars,
                    'datetime': datetime.datetime.now(),
                    'runtime': self.runtime
                })
                self.model.training = 1
                # filtered_info = clear_info(info)
                filtered_info = info

                # self.infos.append(filtered_info)
                self.current_info = info
                yield filtered_info
                start = time.time()

                if self.stop(info):
                    self.stopped = True
                    break
                if interrupt:
                    break
                if self.CTRL_C_FLAG:
                    break

    def _ctrl_c_handler(self, signal, stack):
        self.CTRL_C_FLAG = True

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['data']
        return state
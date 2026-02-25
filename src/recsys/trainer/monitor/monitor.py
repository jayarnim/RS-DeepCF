import collections
import torch
import copy
from .predictor import Predictor
from .calculator import Calculator
from .early_stopper import EarlyStopper


# device setting
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Monitor(object):
    def __init__(
        self,
        predictor: Predictor,
        calculator: Calculator,
        early_stopper: EarlyStopper,
    ):
        super().__init__()
        self.predictor = predictor
        self.calculator = calculator
        self.early_stopper = early_stopper

    def __call__(
        self,
        dataloader: torch.utils.data.dataloader.DataLoader,
    ):
        kwargs = dict(
            dataloader=dataloader,
        )
        result = self.predictor(**kwargs)
        
        kwargs = dict(
            result=result,
        )
        score = self.calculator(**kwargs)

        kwargs = dict(
            current_score=score, 
            current_state=self.current_state,
        )
        self.early_stopper(**kwargs)

        return score

    @property
    def should_stop(self):
        return self.early_stopper.should_stop

    @property
    def counter(self):
        return self.early_stopper.counter

    @property
    def best_epoch(self):
        return self.early_stopper.best_epoch

    @property
    def best_score(self):
        return self.early_stopper.best_score

    @property
    def best_state(self):
        return self.early_stopper.best_state

    @property
    def current_state(self):
        return copy.deepcopy(self.predictor.model.state_dict())
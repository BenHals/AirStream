from collections import deque
from collections import Counter
import numpy as np
import config as config
from modelStats import modelStats, BufferItem


class systemStats:
    def __init__(self):
        self.model_stats = modelStats(0, 'main')
        self.last_seen_examples = deque()
        self.model_update_status = 0
        self.model_update_log = []
        self.change_detection_log = []
        self.state_control_log = []
        self.state_control_log_altered = []
        self.warn_log = []
        self.p_log = []
        self.y_log = []
        self.correct_log = []
        
        self.seen_features = 0
        self.feature_description = {}
        self.last_seen_window_length = config.example_window_length
    
    def add_prediction(self, X, y, p, is_correct, ts):
        example = BufferItem(X, y, p)
        self.last_seen_examples.append(example)
        if(len(self.last_seen_examples) > self.last_seen_window_length):
            self.last_seen_examples.popleft()
        self.correct_log.append((ts, 1 if is_correct else 0))
        self.p_log.append((ts, p))
        self.y_log.append((ts, y))
        self.model_stats.add_prediction(X, y, p, is_correct, ts)
        self.seen_features += 1
        for i,x in enumerate(X):
            if i in self.feature_description:
                feature_type = self.feature_description[i]['type']
                if feature_type == 'numeric':
                    last_m = self.feature_description[i]['m']
                    self.feature_description[i]['m'] = last_m + ((x - last_m) / self.seen_features)
                    self.feature_description[i]['v'] = self.feature_description[i]['v'] + ((x - last_m) * (x - self.feature_description[i]['m']))
                    self.feature_description[i]['var'] = self.feature_description[i]['v'] / max((self.seen_features - 1), 1)
                else:
                    self.feature_description[i]['proportion'][x] += 1
            else:
                if float(x).is_integer():
                    self.feature_description[i] = {
                        'type': 'categorical',
                        'proportion': Counter()
                    }
                    self.feature_description[i]['proportion'][x] += 1
                else:
                    self.feature_description[i] = {
                        'type': 'numeric',
                        'm': x,
                        'v': 0,
                        'var': 0,
                    }
                
    
    def add_warn_prediction(self, X, y, p, is_correct, ts):
        example = BufferItem(X, y, p)
        self.warn_log.append(example)
        
    def clear_warn_log(self):
        self.warn_log = []

    def log_model_update(self, ts, new_model_update_status):
        """Logs the timesteps where a model changes significantly.

        Parameters
        ----------
        ts: int
            The timestep the change occured
        
        new_model_update_states: Any
            The required statistics on the new model structure.
        """
        self.model_update_log.append(ts)
        self.model_update_status = new_model_update_status

    def log_change_detection(self, ts):
        """Logs the timesteps where a detector signals a change.

        Parameters
        ----------
        ts: int
            The timestep the change occured
        """
        self.change_detection_log.append(ts)
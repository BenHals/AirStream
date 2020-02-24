from collections import deque
import numpy as np
import config as config

class BufferItem:
    """Holds an item representing one datastream example.

    Parameters
    ----------
    X: List
        A list representing a feature vector.
    
    y: int
        A class representing the real label of the feature.
    
    p: int
        The predicted label of the feature.
    """
    def __init__(self, X, y, p):
        self.X = X
        self.y = y
        self.p = p
        self.instance = None
    
    def __str__(self):
        return f"X: {self.X}, y: {self.y}, p: {self.p}"
    
    def getExample(self):
        return np.append(self.X, self.y)

class modelStats:
    """Holds the statistics for a model
    
    Parameters
    ----------
    id: int
        The ID for the owner state.
    
    type: string
        The type of the owning model.
    """
    def __init__(self, id, model_type):
        self.id = id                            # The ID of the state.
        self.type = model_type                  # The type of the controlling model.
        self.sliding_window_accuracy_log = deque()   # The accuracy of the state on the sliding window.
        self.sliding_window = deque()           # A sliding window of the last predictions.
        self.first_seen_examples = []           # A list of the first seen examples for the state.
        self.right = 0                          # The number of right predictions
        self.wrong = 0                          # The number of wrong predictions
        self.example_window_length = config.example_window_length

        self.last_sliding_window_acc = 0
        self.sliding_window_right = 0
        self.sliding_window_wrong = 0
        # Tracking running standard deviation instead of recalculating
        # Power Sum Average
        self.running_mean = 0
        # Simple Moving Average
        self.running_var_sum = 0
        self.running_var = 0

        # self.
    
    def add_prediction(self, X, y, p, is_correct, ts):
        example = BufferItem(X, y, p)
        if(len(self.first_seen_examples) < self.example_window_length):
            self.first_seen_examples.append(example)
            
        correct_int = 1 if is_correct else 0
        # Sliding window tracks recent accuracy
        self.sliding_window.append(correct_int)
        if is_correct:
            self.sliding_window_right += 1
        else:
            self.sliding_window_wrong += 1
        if(len(self.sliding_window) > config.report_window_length):
            popped_val = self.sliding_window.popleft()
            if popped_val == 1:
                self.sliding_window_right -= 1
            else:
                self.sliding_window_wrong -= 1
            

        self.last_sliding_window_acc = (self.sliding_window_right) / (self.sliding_window_right + self.sliding_window_wrong)
        self.sliding_window_accuracy_log.append(self.last_sliding_window_acc)
        if len(self.sliding_window_accuracy_log) > config.report_window_length:
            # exit()
            popped_val = self.sliding_window_accuracy_log.popleft()
            old_mean = self.running_mean + 0
            old_x = popped_val
            new_x = self.last_sliding_window_acc
            self.running_mean += (new_x - old_x) / len(self.sliding_window_accuracy_log)
            self.running_var_sum += (new_x + old_x - old_mean - self.running_mean) * (new_x - old_x)
            # self.running_var_sum += (new_x - old_mean) * (new_x - self.running_mean) - (old_x- old_mean) * (old_x- self.running_mean)
            self.running_var = self.running_var_sum / (len(self.sliding_window_accuracy_log) - 1)
        else:
            new_x = self.last_sliding_window_acc
            old_mean = self.running_mean + 0
            self.running_mean = (self.running_mean * (len(self.sliding_window_accuracy_log) - 1) + new_x) / len(self.sliding_window_accuracy_log)
            self.running_var_sum = self.running_var_sum +  (new_x - old_mean) * (new_x - self.running_mean)
            if len(self.sliding_window_accuracy_log) > 2:     
                self.running_var = self.running_var_sum / (len(self.sliding_window_accuracy_log) - 1)
        
        # self.sliding_window_accuracy_log.append((ts, sum(self.sliding_window) / len(self.sliding_window)))

        if is_correct:
            self.right += 1
        else:
            self.wrong += 1

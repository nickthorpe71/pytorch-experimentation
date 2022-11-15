from IPython.display import display, clear_output
import pandas as pd
import time
import json
from collections import OrderedDict
from tqdm.auto import tqdm


class TrainingStatsManager():
    def __init__(self, batches_per_epoch=0):
        self.run_start_time = None
        self.epoch_start_time = None
        self.epoch_count = 0
        self.epoch_data = None
        self.epoch_results = None
        self.run_results = []
        self.progress = None
        self.batches_per_epoch = batches_per_epoch

    def begin_run(self):
        self.run_start_time = time.time()

    def begin_epoch(self):
        self.epoch_start_time = time.time()
        self.epoch_count += 1
        self.epoch_data = OrderedDict()
        self.epoch_results = OrderedDict()

        self.clear_displayed_results()
        self.display_progress()
        self.display_run_results()

    def track(self, key, value):
        if key not in self.epoch_data:
            self.epoch_data[key] = [value]
        else:
            self.epoch_data[key].append(value)

    def add_result(self, key, value):
        self.epoch_results[key] = value

    def end_epoch(self):
        results = OrderedDict()
        results['epoch'] = self.epoch_count
        results['epoch duration'] = time.time() - self.epoch_start_time
        results['run duration'] = time.time() - self.run_start_time

        for k, v in self.epoch_results.items():
            results[k] = v

        self.run_results.append(results)
        self.progress.close()

    def end_run(self):
        self.clear_displayed_results()
        self.display_run_results()

    def display_progress(self):
        self.progress = tqdm(
            total=self.batches_per_epoch, desc=f'Epoch ({self.epoch_count}) Progress'
        )

    def display_run_results(self):
        if len(self.run_results) > 0:
            display(pd.DataFrame.from_dict(self.run_results, orient='columns'))

    def clear_displayed_results(self):
        clear_output(wait=True)

    def save(self, fileName):
        pd.DataFrame.from_dict(
            self.run_results, orient='columns'
        ).to_csv(f'{fileName}.csv')

        with open(f'{fileName}.json', 'w', encoding='utf-8') as f:
            json.dump(self.run_results, f, ensure_ascii=False, indent=4)

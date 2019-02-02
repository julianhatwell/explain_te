import sys

if sys.platform == 'win32':
    path_sep = '\\'
else:
    path_sep = '/'

project_dir = 'CHIRPS'

results_headers = ['dataset_name', 'instance_id', 'algorithm',
            'pretty rule', 'rule length',
            'pred class', 'pred class label',
            'target class', 'target class label',
            'forest vote share', 'pred prior',
            'precision(tr)', 'stability(tr)', 'recall(tr)',
            'f1(tr)', 'accuracy(tr)', 'lift(tr)',
            'coverage(tr)', 'xcoverage(tr)', 'kl_div(tr)',
            'precision(tt)', 'stability(tt)', 'recall(tt)',
            'f1(tt)', 'accuracy(tt)', 'lift(tt)',
            'coverage(tt)', 'xcoverage(tt)', 'kl_div(tt)', 'elapsed_time']

summary_results_headers = ['dataset_name', 'algorithm', 'n_instances', 'n_rules', \
                            'n_rules_used', 'mean_rule_cascade', 'sd_rule_cascade', \
                            'mean_rulelen', 'sd_rulelen', 'begin_time', 'completion_time', \
                            'forest_performance', 'sd_forest_performance', 'sd_proxy_performance', 'sd_fidelity']

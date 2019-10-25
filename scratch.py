def add_rule_term(self):
    candidate = deepcopy(self.rule)
    next_rule_term = self.patterns[self.unapplied_rules[0]]
    candidate_terms = [] # to be output and can be rejected and reverted if no improvement to target function
    feature_upper_bounds = []
    feature_lower_bounds = []
    for item in next_rule_term[0]:
        # list of already used features
        # to be created each item iteration
        # as the order is important can be rarranged by inserts
        feature_appears = [f for (f, _, _) in candidate]

        # skip duplicates (essential for pruning reasons)
        if item in candidate:
            continue

        if item[0] in self.var_dict_enc: # binary feature
            # find the parent feature of item
            parent_feature = self.var_dict_enc[item[0]]

            # check for any known True feature value
            if any(np.array(self.var_dict[parent_feature]['lower_bound']) > 0):
                continue

            # list of already used categorical parent features
            # to be created each item iteration
            # as the order is important can be rarranged by inserts
            categorical_feature_appears = []
            for f_app in feature_appears:
                if f_app in self.var_dict_enc.keys(): # it is an encoded categorical
                    categorical_feature_appears.append(self.var_dict_enc[f_app])
                else: # it is continuous
                    categorical_feature_appears.append(f_app)
            # insert item after last position in current rule where parent item appears
            if parent_feature in categorical_feature_appears:
                candidate.insert(max(np.where(np.array(categorical_feature_appears) == parent_feature)[0]) + 1, item)
            # otherwise just append to current rule
            else:
                candidate.append(item)
            candidate_terms.append(item) # this will output the newly added terms

        else: # continuous feature
            append_or_update = False
            if item[1]: # leq_threshold True
                if item[2] <= self.var_dict[item[0]]['upper_bound'][0]:
                    append_or_update = True

            else:
                if item[2] > self.var_dict[item[0]]['lower_bound'][0]:
                    append_or_update = True

            if append_or_update:
                if item[0] in feature_appears:
                    # print(item, 'feature appears already')
                    valueless_rule = [(f, t) for (f, t, _) in self.rule]
                    if (item[0], item[1]) in valueless_rule: # it's already there and needs updating
                        # print(item, 'feature values appears already')
                        candidate[valueless_rule.index((item[0], item[1]))] = item
                    else: # feature has been used at the opposite end (either lower or upper bound) and needs inserting
                        # print(item, 'feature values with new discontinuity')
                        candidate.insert(feature_appears.index(item[0]) + 1, item)
                else:
                    # print(item, 'feature first added')
                    candidate.append(item)
                candidate_terms.append(item) # this will output the newly added terms

    # remove the first item from unapplied_rules as it's just been applied or ignored for being out of range
    del self.unapplied_rules[0]
    # accumlate all the freq patts that are subsets of the current rules
    # remove the index from the unapplied rules list (including the current rule just added)
    to_remove = []
    accumulated_points = 0
    accumulated_weights = 0
    for ur in self.unapplied_rules:
        # check if all items are already part of the rule (i.e. it's a subset)
        if all([item in candidate for item in self.patterns[ur][0]]):
            # collect up the values to remove. don't want to edit the iterator in progress
            to_remove.append(ur)
            # accumlate points from any deleted terms
            accumulated_points += self.patterns[ur][2]
            accumulated_weights += self.patterns[ur][1]
    for rmv in reversed(to_remove):
        self.unapplied_rules.remove(rmv)
    # make up a new tuple
    t, w, p = next_rule_term
    next_rule_term = (t, w + accumulated_weights, p + accumulated_points)
    return(candidate, candidate_terms, next_rule_term)

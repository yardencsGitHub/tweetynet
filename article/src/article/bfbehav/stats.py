from collections import namedtuple
from itertools import permutations

import numpy as np
import pandas as pd


def p_diff_from_df(df, transition):
    """helper function that computes
    difference between the test statistic
    between two populations.

    The test statistic is
    the probability of a specified transition,
    and the populations are specified as
    all the transitions on a given day.

    Called by ``permutation_test`` function.

    Parameters
    ----------
    df : pandas.DataFrame
        with columns ``day`` and
        ``transition``.
    transition

    Returns
    -------

    """
    p_day1 = df[df.day == 1]['transition'].value_counts(normalize=True).to_dict()[transition]
    p_day2 = df[df.day == 2]['transition'].value_counts(normalize=True).to_dict()[transition]
    return p_day1, p_day2, np.abs(p_day1 - p_day2)


PermutationTest = namedtuple('PermutationTest',
                             field_names=('p_day1_real',
                                          'p_day2_real',
                                          'p_day1_perm',
                                          'p_day2_perm'))


def permutation_test(counter_day1,
                     counter_day2,
                     from_state,
                     transition,
                     count_thresh=15,
                     n_perm=1000):
    """permutation test to
    determine whether
    difference between the test statistic
    between two populations.

    The test statistic is
    the probability of a specified transition,
    and the populations are specified as
    all the transitions on a given day.

    Implementation of the test in [1]_.

    For a more general implementation, see e.g.
    https://rasbt.github.io/mlxtend/user_guide/evaluate/permutation_test/

    Parameters
    ----------
    counter_day1 : collections.Counter
        returned by ``article.bfbehavior.sequence.transition_matrix``
        where keys are transitions, e.g., ('b', 'b'), and the
        corresponding values are counts of that transition.
    counter_day2 : collections.Counter
        same as ``counter_day`` but from a different day
    from_state : str
        single character indicating "from" state label that will
        be considered, e.g. 'b'
    transition : tuple
        of two single characters, e.g., ('b', 'b')
    count_thresh : int
        threshold below which counts are not considered.
        If counts for any transition including ``from_state``
        are lower than this threshold, they are ignored.
        Default is 15.
    n_perm : int
        number of times to permute dataset and perform draws
        and compute difference

    Returns
    -------
    p_value : float
        empirical p-value, number of times that difference
        between samples from permutated dataset were at least
        as large as the observed value
    perm_test : PermutationTest
        named tuple with fields 'p_day1_real', 'p_day2_real',
        'p_day1_perm', 'p_day2_perm'. The scalar 'real' fields
        are the observed probabilities, and the 'perm' fields
        contains arrays with all the probabilities computed
        from the permuted dataset.

    References
    ----------
    .. [1] Variable Sequencing Is Actively
           Maintained in a Well Learned Motor Skill
           Timothy L. Warren, Jonathan D. Charlesworth,
           Evren C. Tumer, Michael S. Brainard
           Journal of Neuroscience 31 October 2012,
           32 (44) 15414-15425; DOI: 10.1523/JNEUROSCI.1254-12.2012
    """
    records = []
    for day, counter in zip(
            (1, 2),
            (counter_day1, counter_day2)
    ):
        # `transition_` to avoid clash with function argument
        for transition_ in counter.keys():
            if transition_[0] == from_state:
                count = counter[transition_]
                if count > count_thresh:
                    for _ in range(count):
                        records.append(
                            {'day': day, 'transition': transition_}
                        )
    df = pd.DataFrame.from_records(records)

    p_day1_real, p_day2_real, p_diff_real = p_diff_from_df(df, transition)

    n_gt = 0
    p_day1_perm = []
    p_day2_perm = []

    for perm in range(n_perm):
        perm_df = df.copy()
        perm_df.day = np.random.permutation(perm_df.day)
        assert np.all(np.equal(perm_df.day.value_counts().values, df.day.value_counts().values))
        p_day1_perm_, p_day2_perm_, p_diff_perm = p_diff_from_df(perm_df, transition)
        p_day1_perm.append(p_day1_perm_)
        p_day2_perm.append(p_day2_perm_)
        if p_diff_perm >= p_diff_real:
            n_gt += 1

    pvalue = n_gt / n_perm
    perm_test = PermutationTest(p_day1_real, p_day2_real, p_day1_perm, p_day2_perm)
    return pvalue, perm_test


def _multiple_perm_tests(condition_counter_map,
                         from_state,
                         transition,
                         n_perm=1000,
                         alpha=0.5):
    """helper function to avoid repeating ourselves."""
    p_vals = []
    for conditions, counters in condition_counter_map.items():
        cond1, cond2 = conditions
        print(
            f'performing permutation test for: {cond1}, {cond2}'
        )
        counter1, counter2 = counters

        p_val, perm_test = permutation_test(counter1,
                                            counter2,
                                            from_state=from_state,
                                            transition=transition,
                                            n_perm=n_perm)
        print(f'p-value was {p_val}')
        p_vals.append(p_val)

    corrected_alpha = alpha / len(p_vals)

    if not any([p_val < corrected_alpha for p_val in p_vals]):
        print(
            f'none of the p-values were less than corrected alpha {corrected_alpha}.\n'
            f'Fail to reject the null hypothesis. p-values: {p_vals}'
        )
    else:
        print(
            f'At least one of the p-values was less than corrected alpha {corrected_alpha}.\n'
            f'Reject the null hypothesis. p-values: {p_vals}'
        )

    return p_vals, corrected_alpha


def perm_test_across_days(animal_day_transmats,
                          animal_id,
                          from_state,
                          transition,
                          n_perm=1000,
                          alpha=0.05):
    day_transmats_map = animal_day_transmats[animal_id]

    days = list(day_transmats_map.keys())
    pairs = permutations(days, 2)

    # map pairwise dates in ground truth to their transition counters
    condition_counter_map = {
        pair: (day_transmats_map[pair[0]].counts, day_transmats_map[pair[1]].counts)
        for pair in pairs
    }

    return _multiple_perm_tests(condition_counter_map,
                                from_state,
                                transition,
                                n_perm=n_perm,
                                alpha=alpha)


def perm_test_across_models(animal_day_transmats_true,
                            animal_day_transmats_pred,
                            animal_id,
                            day,
                            from_state,
                            transition,
                            n_perm=1000,
                            alpha=0.05):
    day_transmats_map_true = animal_day_transmats_true[animal_id]
    day_transmats_map_pred = animal_day_transmats_pred[animal_id]

    # map each (ground truth, model) pair to their transition counters
    condition_counter_map = {
        ('true', model): (day_transmats_map_true[day].counts, day_transmats_map_pred[day][model].counts)
        for model in day_transmats_map_pred[day].keys()
    }

    return _multiple_perm_tests(condition_counter_map,
                                from_state,
                                transition,
                                n_perm=n_perm,
                                alpha=alpha)

import math
import scipy.stats as st


def proportion_difference_test(baseline, values, corpus_size, testing_ratio=0.33, statistical_significance=0.05):
    """
    :param baseline: The baseline result in a decimal fraction
    :param values: A collection of the new classification results that  need to test against the baseline
    :param corpus_size: Total corpus size (number of posts)
    :param testing_ratio: The corpus ratio defined as a Testing of the whole corpus
    :param statistical_significance: Desired statistical significance (default 5%)
    :return: Dictionary of Arranged Pairs: Any classification result and a string that describes whether it is large,
             small, or nothing relative to the baseline result
    """

    # initial the variables for the proportion difference test
    corpus_size = corpus_size*testing_ratio
    values = list(set(values))
    result = {}

    # Calculate the Z value for the given statical significance
    z_alpha = st.norm.ppf(1 - statistical_significance)

    # Check the hypotheses for each result
    for p in values:
        x1 = corpus_size*p
        x2 = corpus_size*baseline

        p_ = (x1 + x2)/(corpus_size + corpus_size)
        q_ = 1 - p_

        z_up = p - baseline
        z_down = math.sqrt(p_*q_*(1/corpus_size + 1/corpus_size))
        z = z_up/z_down

        print('p = ', p)
        print('x1 = ', x1)
        print('x2 = ', x2)
        print('p̂ = ', p_)
        print('q̂ = ', q_)
        print('z = ', z)
        print('z alpha = ', z_alpha)
        print('------------------------')

        # H0: p = baseline
        # H1: p < baseline
        if z < -z_alpha:
            result[p] = 'smaller'
            continue

        # H0: p = baseline
        # H1: p > baseline
        if z > z_alpha:
            result[p] = 'bigger'
            continue

        result[p] = 'none'

    return result


if __name__ == "__main__":
    A = [0.88, 0.75, 0.81, 0.65, 0.95]
    B = 0.8
    print(A)
    print(proportion_difference_test(B, A, 200))

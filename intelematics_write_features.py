import os
import pandas
import numpy as np
import json
from scipy import signal


def get_local_maxima(smoothed_trace, min_index):
    """
    Getting list of local maxima considering only big enough peaks (>0.1),
    separated by enough points (>10), with a convexity (10 points before and
    10 points after are points below this maximum).
    See also: https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/
    scipy.signal.argrelextrema.html
    :param list smoothed_trace: the smoothed voltage trace.
    :param int min_index: the index of the minimum on the curve (x-axis).
    :return list local_maxima: A list of local maxima indices (x-axis).
    """
    local_maxima = []
    previous = None
    for peak in signal.argrelextrema(
            np.array(smoothed_trace[:-10]), np.greater)[0]:
        if not previous:
            if abs(smoothed_trace[peak] - smoothed_trace[peak - 10]) > 0.1 \
                    and smoothed_trace[peak - 10] < smoothed_trace[peak] \
                    and smoothed_trace[peak + 10] < smoothed_trace[peak]:
                local_maxima.append(peak + min_index)
                previous = peak
            continue
        if abs(peak - previous) >= 10 \
                and (abs(smoothed_trace[peak] -
                         smoothed_trace[peak - 10])) > 0.1 \
                and smoothed_trace[peak-10] < smoothed_trace[peak] \
                and smoothed_trace[peak+10] < smoothed_trace[peak]:
            local_maxima.append(peak + min_index)
            previous = peak
    return local_maxima


def get_end_of_mcv_domain(smoothed_trace):
    """
    Getting a first resizing to discard the end of the curve for the MCV
    calculation. We want to exclude everything coming after the last surge.
    :param smoothed_trace: the smoothed voltage trace.
    """
    domain = get_diff_times_10(smoothed_trace[np.argmin(smoothed_trace):-30])
    standard_deviation = np.std(domain)
    limit = 2.5 * standard_deviation
    end_candidates = []
    for index, point in enumerate(domain):
        if point > limit:
            end_candidates.append(index + np.argmin(smoothed_trace))
    return end_candidates[-1]


def project_last_maximum_on_curve(smoothed_trace, last_maximum):
    """
    Getting a second resizing for the MCV calculation: the final upper limit
    of the MCV domain will be the projection of the last maximum on the curve
    in the case of that maximum actually exists.
    :param list smoothed_trace: the smoothed voltage trace.
    :param int last_maximum: the last local maximum on the curve.
    """
    domain = smoothed_trace[last_maximum+10:-10]
    limit = smoothed_trace[last_maximum]
    for index, point in enumerate(domain):
        if point > limit:
            return last_maximum + index + 10


def get_diff_times_10(domain):
    """
    Used to identify voltage surges (sudden raise of value) and spot the last
    one to discard the tail of the curve coming after it.
    :param list domain: The domain to perform the differentiation on.
    :return list differentiation: a list of differences at each point.
    """
    diff_10 = []
    for index, element in enumerate(domain[:-1]):
        diff_10.append((domain[index+1]-domain[index])*10)
    return diff_10


def extract_features(crank):
    """
    Getting the features (iv, lvv, mcv) for a given crank and the timestamp
    at generation.
    n.b: Any exception raised (on purpose or not) during the features
    extraction (invalidity of some statistical values or no suitable
    end domain) will discard the crank, tag it as invalid and cancel the
    SOH prediction. This is a behavior validated by ANWB.
    :param crank: the crank message to extract the features from.
    :return dict features: the features for the crank.
    e.g: {'iv': 4.00, 'lvv': 1.00, 'mcv': 3.00, 'time': '2018-11-26T07:36'}
    """
    smoothed_trace = get_smoothed_trace(crank)
    min_index = np.argmin(smoothed_trace)
    iv = get_iv(smoothed_trace, min_index)
    mcv = get_mcv(smoothed_trace, min_index)
    lvv = get_lvv(smoothed_trace)
    return {'iv': iv, 'lvv': lvv, 'mcv': mcv, 'time': crank['time']}


def get_iv(smoothed_trace, minimum_index):
    """
    Getting the "initial voltage" (iv) coefficient.
    It's the greatest value before the first minimum of the smoothed
    signal.
    :param list smoothed_trace: the smoothed voltage trace.
    :param int minimum_index: the minimum index of the smoothed trace.
    :return float iv: the initial voltage coefficient.
    """
    return max(smoothed_trace[:minimum_index + 1])


def get_lvv(smoothed_trace):
    """
    Getting the "low valley voltage" (lvv) coefficient.
    It's the minimum of the smoothed voltage signal.
    :return float lvv: the low valley voltage coefficient.
    """
    return min(smoothed_trace)


def get_mcv(smoothed_trace, minimum_index):
    """
    Getting the "mean crank voltage" (mcv) coefficient.
    n.b: The average is NOT calculated on the whole domain but ONLY between
    the index of the minimum and the index of the projection of the last
    local maximum. If there is NO maxima, we take a broader limit based
    on the very last voltage surge, calculated based on a distance to a
    differentiation distribution. Trying to explain and understand this one
    without a good old drawing is a bit tricky. Ask someone from ANWB
    (Jim, Jeroen) or IEU (Clement) who worked on the algorithm and can
    draw a typical voltage trace with the local maxima, the projection of
    the last maximum and the end of the curve considered for a MCV domain.
    :param list smoothed_trace: the smoothed voltage trace.
    :param int minimum_index: the minimum index of the smoothed trace.
    :return float mcv: the mean crank voltage coefficient.
    """
    end_mcv_domain = get_end_of_mcv_domain(smoothed_trace)
    local_maxima = get_local_maxima(
        smoothed_trace[minimum_index:end_mcv_domain], minimum_index)
    if not local_maxima:
        return np.round(
            np.mean(smoothed_trace[minimum_index:end_mcv_domain]), 2)
    projection = project_last_maximum_on_curve(
        smoothed_trace, local_maxima[-1])
    average_ends = projection if projection else end_mcv_domain
    return np.round(np.mean(
        smoothed_trace[minimum_index:average_ends]), 2)


def get_smoothed_trace(crank, window=5):
    """
    Smoothing the voltage trace with a moving average approach based on
    averaging [window] points before the current point. A moving average is
    commonly used with time series data to smooth out short-term
    fluctuations and highlight longer-term trends or cycles. The wikipedia
    page is a good start: https://en.wikipedia.org/wiki/Moving_average
    Here, a minimum period is specified to perform averages even at
    the edges (when only [min_periods] point(s) have been seen, we still
    smooth by mean based on the points available).
    :param dict crank: The crank to smooth the voltage from. It can be the
    current crank or a former one. (e.g: {"source_id": "353386065619625",
    "source_type": "tm8", ... "voltage_trace": [12.3,11.0,3.4,12.3,11.0]})
    :param int window: The number of points for the averaging. (e.g: 5)
    :return list smoothed_trace: The smoothed trace by moved average.
    (e.g: [1.26, 1.225, 1.42, 1.86, 3.44,  6.45, 10.87, 21.5, 54.4, 89.96,
    127.25, 163.22, 192.95, 221.2, 265.6, 307.4, 388.0, 508.8, 646.8])
    """
    index_min = np.argmin(crank['crank_profile'])
    crank_frame = pandas.DataFrame(
        {'trace': crank['crank_profile'][index_min + 1:]})
    return crank['crank_profile'][:index_min + 1] + \
           crank_frame['trace'].rolling(
               min_periods=1, window=window).mean().dropna().tolist()


if __name__ == '__main__':

    result_file = open("features_IEU_algorithm.txt", "w")
    for root, subdirs, files in os.walk('anwb_data'):
        path_list = root.split('/')
        if path_list[-1] == 'crankData':
            for file in os.listdir(root):
                if file.endswith(".json"):
                    with open(root + '/' + file) as f:
                        data = json.load(f)
                        result_file.write(root + '/' + file)
                        result_file.write(' ')
                        result_file.write(json.dumps((extract_features(data))))
                        result_file.write("\n")
    result_file.close()



import os
import csv
import numpy as np
import itertools as it
import pickle

def read_csv(filename, nrows, input_name, target_name):
    data = ([], [])
    with open(filename) as file:
        reader = csv.DictReader(file)
        for row in it.islice(reader, nrows):
            data[0].append(row[input_name])
            data[1].append(float(row[target_name]))
    return map(np.array, data)

def load_data(filename, sizes, input_name, target_name):
    slices = []
    start = 0
    for size in sizes:
        stop = start + size
        slices.append(slice(start, stop))
        start = stop
    return load_data_slices_nolist(filename, slices, input_name, target_name)

def load_data_slices_nolist(filename, slices, input_name, target_name):
    stops = [s.stop for s in slices]
    if not all(stops):
        raise Exception("Slices can't be open-ended")

    data = read_csv(filename, max(stops), input_name, target_name)
    return [(data[0][s], data[1][s]) for s in slices]


def list_concat(lists):
    return list(it.chain(*lists))
    
def load_data_slices(filename, slice_lists, input_name, target_name):
    stops = [s.stop for s in list_concat(slice_lists)]
    if not all(stops):
        raise Exception("Slices can't be open-ended")

    data = read_csv(filename, max(stops), input_name, target_name)

    return [(np.concatenate([data[0][s] for s in slices], axis=0),
             np.concatenate([data[1][s] for s in slices], axis=0))
            for slices in slice_lists]

def get_output_file(rel_path):
    return os.path.join(output_dir(), rel_path)

def get_data_file(rel_path):
    return os.path.join(data_dir(), rel_path)

def output_dir():
    return os.path.expanduser(safe_get("OUTPUT_DIR"))

def data_dir():
    return os.path.expanduser(safe_get("DATA_DIR"))

def safe_get(varname):
    if varname in os.environ:
        return os.environ[varname]
    else:
        raise Exception("%s environment variable not set" % varname)


def load_pickle(filename, sizes, input_name, target_name, filter = 'BGB+'):
    
    d = pickle.load(open(filename, 'rb')) # FreeSolve Database
    #k = d.keys()
    if(filter != 'none'):
        k = [a for a in d.keys() if filter in d[a].keys()]#rigid subset
    else:
        k = [a for a in d.keys()]
        
    k = np.random.permutation(k)

    ratio = np.array(sizes)/float(sum(sizes))
    stops = [0]+list([int(sum(ratio[:(i+1)])*len(k)) for i in range(len(ratio))])

    feat = []
    targ = []
    for i in k:
        feat.append(d[i][input_name])
        targ.append(d[i][target_name])
    
    return tuple([(feat[stops[i]:stops[i+1]], targ[stops[i]:stops[i+1]]) for i in range(len(stops)-1)])

def load_pickle_comp(filename, sizes, input_name, target_name1, target_name2, filter = 'BGB+'):
    
    d = pickle.load(open(filename, 'rb')) # FreeSolve Database
    #k = d.keys()
    if(filter != 'none'):
        k = [a for a in d.keys() if filter in d[a].keys()]#rigid subset
    
    np.random.seed(1)
    k = np.random.permutation(k)

    ratio = np.array(sizes)/float(sum(sizes))
    stops = [0]+list([int(sum(ratio[:(i+1)])*len(k)) for i in range(len(ratio))])

    feat = []
    targ = []
    for i in k:
        feat.append(d[i][input_name])
        targ.append(d[i][target_name1]-d[i][target_name2])
    
    return tuple([(feat[stops[i]:stops[i+1]], targ[stops[i]:stops[i+1]]) for i in range(len(stops)-1)])

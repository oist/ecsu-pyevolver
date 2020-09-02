import base64
import json
import numpy as np


class NumpyListJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        return json.JSONEncoder.default(self, obj)


class NumpyJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        """
        if input object is a ndarray it will be converted into a dict holding dtype, shape and the data base64 encoded
        """
        if isinstance(obj, np.ndarray):
            data_b64 = base64.b64encode(obj.copy(order='C'))
            data_b64 = data_b64.decode('ascii')
            return dict(__ndarray__=data_b64,
                        dtype=str(obj.dtype),
                        shape=obj.shape)
        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)


def dump_np_array(a, file_path):
    with open(file_path, 'w') as f_out:
        json.dump(a, f_out, cls=NumpyListJsonEncoder, indent=3)


def read_np_array(a, file_path):
    with open(file_path) as f_in:
        l = json.load(f_in)
    return np.array(l)


def json_numpy_obj_hook(dct):
    """
    Decodes a previously encoded numpy ndarray
    with proper shape and dtype
    :param dct: (dict) json encoded ndarray
    :return: (ndarray) if input was an encoded ndarray
    """
    if isinstance(dct, dict) and '__ndarray__' in dct:
        data = base64.b64decode(dct['__ndarray__'])
        return np.frombuffer(data, dct['dtype']).reshape(dct['shape'])
    return dct


# Overload dump/load to default use this behavior.
def dumps(*args, **kwargs):
    kwargs.setdefault('cls', NumpyJsonEncoder)
    return json.dumps(*args, **kwargs)


def loads(*args, **kwargs):
    kwargs.setdefault('object_hook', json_numpy_obj_hook)
    return json.loads(*args, **kwargs)


def dump(*args, **kwargs):
    kwargs.setdefault('cls', NumpyJsonEncoder)
    return json.dump(*args, **kwargs)


def load(*args, **kwargs):
    kwargs.setdefault('object_hook', json_numpy_obj_hook)
    return json.load(*args, **kwargs)


def test_two_levels():
    data = np.arange(3, dtype=np.complex)

    one_level = {'level1': data, 'foo': 'bar'}
    two_level = {'level2': one_level}

    dumped = dumps(two_level)
    result = loads(dumped)
    assert (result['level2']['level1'] == two_level['level2']['level1']).all()


def test_transpose():
    A = np.arange(10).reshape(2, 5)
    A = A.transpose()
    dumped = dumps(A)
    result = loads(dumped)
    assert (A == result).all()


def test_continuous():
    A = np.arange(10).reshape(2, 5)
    print(A.flags)
    # C_CONTIGUOUS : True
    # F_CONTIGUOUS : False
    # OWNDATA : False
    # WRITEABLE : True
    # ALIGNED : True
    # UPDATEIFCOPY : False
    A = A.transpose()
    # array([[0, 5],
    #       [1, 6],
    #       [2, 7],
    #       [3, 8],
    #       [4, 9]])
    loads(dumps(A))
    # array([[0, 1],
    #       [2, 3],
    #       [4, 5],
    #       [6, 7],
    #       [8, 9]])
    print(A.flags)
    # C_CONTIGUOUS : False
    # F_CONTIGUOUS : True
    # OWNDATA : False
    # WRITEABLE : True
    # ALIGNED : True
    # UPDATEIFCOPY : False


if __name__ == '__main__':
    # test_two_levels()
    # test_continuous()
    test_transpose()

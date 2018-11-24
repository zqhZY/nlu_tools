import argparse
import sys


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        metavar='C',
        default='None',
        help='The Configuration file')
    argparser.add_argument(
        '-s', '--step',
        metavar='S',
        default='train',
        help='The Configuration file')
    args = argparser.parse_args()
    return args

# print tensor shape
def print_shape(varname, var):
    """
    :param varname: tensor name
    :param var: tensor variable
    """
    print('{0} : {1}'.format(varname, var.get_shape()))


def import_class(import_str):
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' %
                (class_str,
                    traceback.format_exception(*sys.exc_info())))


def import_object(import_str, *args, **kwargs):
    return import_class(import_str)(*args, **kwargs)


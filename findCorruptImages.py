# -*- coding: utf-8 -*-
# vi:ts=4 sw=4 et

# Okay, this code is a bit ugly, with a few "anti-patterns" and "code smell".
# But it works and I don't want to refactor it *right now*.

# TODO:
#  * Refactor it a little
#  * Add support for custom filename filter (instead of the hardcoded one)


import getopt
import fnmatch
import re
import os
import os.path
import sys
import functools
import PIL.Image
from subprocess import Popen, PIPE

from Misc import sortKey, processArguments

available_parameters = [
    ("h", "help", "Print help"),
    ("v", "verbose", "Also print clean files"),
]

def checkImage(fn):
  proc = Popen(['identify', '-verbose', fn], stdout=PIPE, stderr=PIPE)
  out, err = proc.communicate()
  exitcode = proc.returncode
  return exitcode, out, err

class ProgramOptions(object):
    """Holds the program options, after they are parsed by parse_options()"""

    def __init__(self):
        self.globs = ['*.jpg', '*.jpe', '*.jpeg']
        self.glob_re = re.compile('|'.join(
            fnmatch.translate(g) for g in self.globs
        ), re.IGNORECASE)

        self.verbose = False
        self.args = []


def print_help():
    global opt
    scriptname = os.path.basename(sys.argv[0])
    print("Usage: {0} [options] files_or_directories".format(scriptname))
    print("Recursively checks for corrupt JPEG files")
    print("")
    print("Options:")
    long_length = 2 + max(len(int) for x, int, y in available_parameters)
    for short, int, desc in available_parameters:
        if short and int:
            comma = ", "
        else:
            comma = "  "

        if short == "":
            short = "  "
        else:
            short = "-" + short[0]

        if int:
            long = "--" + int

        print("  {0}{1}{2:{3}}  {4}".format(short, comma, int, long_length, desc))

    print("")
    print("Currently (it is hardcoded), it only checks for these files:")
    print("  " + " ".join(opt.globs))


def parse_options(argv, opt):
    """argv should be sys.argv[1:]
    opt should be an instance of ProgramOptions()"""

    try:
        opts, args = getopt.getopt(
            argv,
            "".join(short for short, x, y in available_parameters),
            [int for x, int, y in available_parameters]
        )
    except getopt.GetoptError as e:
        print(str(e))
        print("Use --help for usage instructions.")
        sys.exit(2)

    for o, v in opts:
        if o in ("-h", "--help"):
            print_help()
            sys.exit(0)
        elif o in ("-v", "--verbose"):
            opt.verbose = True
        else:
            print("Invalid parameter: {0}".format(o))
            print("Use --help for usage instructions.")
            sys.exit(2)

    opt.args = args
    if len(args) == 0:
        print("Missing filename")
        print("Use --help for usage instructions.")
        sys.exit(2)


def is_corrupt(jpegfile):
    """Returns None if the file is okay, returns an error string if the file is corrupt."""
    # http://stackoverflow.com/questions/1401527/how-do-i-programmatically-check-whether-an-image-png-jpeg-or-gif-is-corrupted/1401565#1401565
    try:
        im = PIL.Image.open(jpegfile)
        im.verify()
    except Exception as e:
        return str(e)
    return None


def check_files(files, method, delete_file=0):
    """Receives a list of files and check each one."""
    global opt
    n_files = len(files)
    for i, f in enumerate(files):
        # Filtering only JPEG images
        if not (f.endswith('.jpg') or f.endswith('.jpeg')):
            continue
        if method==0:
            status = is_corrupt(f)
            if status:
                # os.remove(f)
                if delete_file:
                    print('\nDeleting corrupt file: {}'.format(f))
                    os.remove(f)
                else:
                    print('\nFound corrupt file: {:s}\n'.format(f))
                # print "{0}: {1}".format(f, status)
        else:
            code, output, error = checkImage(f)
            if str(code) != "0" or str(error) != "":
                raise IOError("Damaged image found: {} :: {}".format(f, error))
        sys.stdout.write('\rDone {}/{} images'.format(i+1, n_files))
        sys.stdout.flush()
    # print()

# def main():
#     global opt
#
#     opt = ProgramOptions()
#     parse_options(sys.argv[1:], opt)
#
#     for pathname in opt.args:
#         if os.path.isfile(pathname):
#             check_files([pathname])
#         elif os.path.isdir(pathname):
#             for dirpath, dirnames, filenames in os.walk(pathname):
#                 check_files([os.path.join(dirpath, f) for f in filenames])
#         else:
#             print("ERROR: '{0}' is neither a file or a dir.".format(pathname))


if __name__ == "__main__":

    params = {
        'root_dir': '.',
        'delete_file': 0,
        'method': 0,
        'file_type': '',
        'show_img': 0,
    }
    processArguments(sys.argv[1:], params)
    root_dir = params['root_dir']
    delete_file = params['delete_file']
    method = params['method']
    file_type = params['file_type']
    show_img = params['show_img']

    img_exts = ('.jpg', '.bmp', '.jpeg', '.png', '.tif', '.tiff')

    src_file_gen = [[os.path.join(dirpath, f) for f in filenames if
                     os.path.splitext(f.lower())[1] in img_exts]
                    for (dirpath, dirnames, filenames) in os.walk(root_dir, followlinks=True)]
    _src_files = [item for sublist in src_file_gen for item in sublist]
    _src_files = [os.path.abspath(k) for k in _src_files]
    _src_files.sort(key=functools.partial(sortKey,only_basename=0))

    check_files(_src_files, method, delete_file)

    # for dirpath, dirnames, filenames in os.walk(root_dir):
    #     check_files([os.path.join(dirpath, f) for f in filenames])

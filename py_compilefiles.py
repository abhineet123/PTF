#! /usr/bin/python

import os
import sys
import py_compile

def compile_files(files, ddir=None, force=0, rx=None, quiet=0, ignore=0):
    """Byte-compile all file.
    file:      the file to byte-compile
    ddir:      if given, purported directory name (this is the
               directory name that will show up in error messages)
    force:     if 1, force compilation, even if timestamps are up-to-date
    quiet:     if 1, be quiet during compilation

    """

    success = 1
    dfile = None
    for fullname in files:
        if rx is not None:
            mo = rx.search(fullname)
            if mo:
                continue
        if os.path.isdir(fullname):
            continue
        elif not os.path.isfile(fullname):
            print "file does not exist:", fullname
            success = 0
        elif fullname[-3:] == '.py':
            cfile = fullname + (__debug__ and 'c' or 'o')
            ftime = os.stat(fullname).st_mtime
            try: ctime = os.stat(cfile).st_mtime
            except os.error: ctime = 0
            if (ctime > ftime) and not force: continue
            if not quiet:
                print 'Compiling', fullname, '...'
            try:
                ok = py_compile.compile(fullname, None, dfile, True)
            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except py_compile.PyCompileError,err:
                if quiet:
                    print 'Compiling', fullname, '...'
                print err.msg
                success = 0
            except (MemoryError, SyntaxError),err:
                if quiet:
                    print 'Compiling', fullname, '...'
                print err.msg
                success = 0
            except IOError, e:
                print "Sorry", e
                success = 0
            else:
                if ok == 0:
                    success = 0
    if not success and ignore:
        print "Errors were ignored."
    return success or ignore

def main():
    """Script main program."""
    import getopt
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'lfiqd:x:')
    except getopt.error, msg:
        print msg
        print "usage: python compilefiles.py [-f] [-q] [-i] " \
              "[-x regexp] [file ...] [-]"
        print "-f: force rebuild even if timestamps are up-to-date"
        print "-i: ignore errors during byte compilation"
        print "-q: quiet operation"
        print "-x regexp: skip files matching the regular expression regexp"
        print "   the regexp is search for in the full path of the file"
        sys.exit(2)
    ddir = None
    force = 0
    quiet = 0
    ignore = 0
    rx = None
    for o, a in opts:
        if o == '-d': ddir = a
        if o == '-f': force = 1
        if o == '-i': ignore = 1
        if o == '-q': quiet = 1
        if o == '-x':
            import re
            rx = re.compile(a)
    if ddir:
        if len(args) != 1:
            print "-d destdir require exactly one directory argument"
            sys.exit(2)
    success = 1

    try:
        files = []
        for arg in args:
            if arg == '-':
                while 1:
                    line = sys.stdin.readline()
                    if not line:
                        break
                    files.append(line[:-1])
            else:
                files.append(arg)
        if not files:
            print "py_compilefiles: no files to compile"
        elif not compile_files(files, ddir, force, rx, quiet, ignore):
            success = 0
    except KeyboardInterrupt:
        print "\n[interrupt]"
        success = 0
    return success

if __name__ == '__main__':
    exit_status = int(not main())
    sys.exit(exit_status)

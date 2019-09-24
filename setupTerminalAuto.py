from pywinauto import application
import time
import os, sys

from Misc import processArguments, sortKey

if __name__ == '__main__':
    params = {
        'config': 0,
    }
    processArguments(sys.argv[1:], params)
    config = params['config']

    app = application.Application().start("C:/cygwin64/home/Tommy/fatty.exe")
    app.fatty.TypeKeys("t~")
    app.fatty.TypeKeys("^+t")
    app.fatty.TypeKeys("f~")
    app.fatty.TypeKeys("^+t")
    app.fatty.TypeKeys("sstg{VK_SPACE}tb~")
    app.fatty.TypeKeys("sudo{VK_SPACE}-s~")
    app.fatty.TypeKeys("';';';';~")
    time.sleep(2)

    if config == 0:
        app.fatty.TypeKeys("tmux{VK_SPACE}a{VK_SPACE}-t{VK_SPACE}grs~")
    else:
        app.fatty.TypeKeys("tmux{VK_SPACE}a{VK_SPACE}-t{VK_SPACE}grs2~")

    app.fatty.TypeKeys("^+t")
    app.fatty.TypeKeys("sstg2~")
    app.fatty.TypeKeys("sstz~")
    app.fatty.TypeKeys("sudo{VK_SPACE}-s~")
    app.fatty.TypeKeys("'''~")
    time.sleep(15)

    if config == 0:
        app.fatty.TypeKeys("tmux{VK_SPACE}a{VK_SPACE}-t{VK_SPACE}orca~")
    else:
        app.fatty.TypeKeys("tmux{VK_SPACE}a{VK_SPACE}-t{VK_SPACE}orca~")


    print()


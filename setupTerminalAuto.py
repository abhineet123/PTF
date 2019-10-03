from pywinauto import application
import time
import os, sys

from Misc import processArguments, sortKey

if __name__ == '__main__':
    params = {
        'exe_path': '',
        'config': 0,
        'wait_t': 10,
    }
    processArguments(sys.argv[1:], params)
    exe_path = params['exe_path']
    config = params['config']
    wait_t = params['wait_t']

    auth_data = open('')

    if not exe_path:
        raise IOError('Terminal executable path must be provided')

    app = application.Application().start(exe_path)
    app.window().maximize()
    if config == -1:

        apps = [app, ]
        app2 = application.Application().start(exe_path)
        app2.window().maximize()

        apps.append(app2)

        for _app in apps:
            _app.fatty.type_keys("t~")
            _app.fatty.type_keys("^+t")
            _app.fatty.type_keys("f~")
            _app.fatty.type_keys("^+t")
            _app.fatty.type_keys("sstg{VK_SPACE}tb~")
            _app.fatty.type_keys("sudo{VK_SPACE}-s~")
            _app.fatty.type_keys("';';';';~")

            # time.sleep(1)

        app3 = application.Application().start(exe_path)
        app3.window().maximize()
        app3.fatty.type_keys("tmux{VK_SPACE}new~")

        time.sleep(5)

        app.fatty.type_keys("tmux{VK_SPACE}a{VK_SPACE}-t{VK_SPACE}grs~")
        app2.fatty.type_keys("tmux{VK_SPACE}a{VK_SPACE}-t{VK_SPACE}grs2~")

        for _app in apps:
            _app.fatty.type_keys("^+t")
            _app.fatty.type_keys("sstg2~")
            _app.fatty.type_keys("sstz~")
            _app.fatty.type_keys("sudo{VK_SPACE}-s~")
            _app.fatty.type_keys("'''~")

        time.sleep(5)

        app.fatty.type_keys("tmux{VK_SPACE}a{VK_SPACE}-t{VK_SPACE}orca~")
        app2.fatty.type_keys("tmux{VK_SPACE}a{VK_SPACE}-t{VK_SPACE}orca2~")

        # time.sleep(1)

        # time.sleep(1)
        # app.fatty.type_keys("+{RIGHT}")
        # app2.fatty.type_keys("+{RIGHT}")

        for _app in apps:
            _app.fatty.type_keys("^+t")
            _app.fatty.type_keys("sstg3~")
            _app.fatty.type_keys("sstx~")
            _app.fatty.type_keys("sudo{VK_SPACE}-s~")
            _app.fatty.type_keys("'''~")

        app.fatty.type_keys("tmux{VK_SPACE}a{VK_SPACE}-t{VK_SPACE}x99~")
        app2.fatty.type_keys("tmux{VK_SPACE}a{VK_SPACE}-t{VK_SPACE}x992~")

        app3.fatty.type_keys("^b^r")

    elif config == 2:
        app.fatty.type_keys("tmux{VK_SPACE}new~")
        time.sleep(wait_t)
        app.fatty.type_keys("^b^r")
    else:
        app.fatty.type_keys("t~")
        app.fatty.type_keys("^+t")
        app.fatty.type_keys("f~")
        app.fatty.type_keys("^+t")
        app.fatty.type_keys("sstg{VK_SPACE}tb~")
        app.fatty.type_keys("sudo{VK_SPACE}-s~")
        app.fatty.type_keys("';';';';~")
        time.sleep(2)

        if config == 0:
            app.fatty.type_keys("tmux{VK_SPACE}a{VK_SPACE}-t{VK_SPACE}grs~")
        else:
            app.fatty.type_keys("tmux{VK_SPACE}a{VK_SPACE}-t{VK_SPACE}grs2~")

        app.fatty.type_keys("^+t")
        app.fatty.type_keys("sstg2~")
        app.fatty.type_keys("sstz~")
        app.fatty.type_keys("sudo{VK_SPACE}-s~")
        app.fatty.type_keys("'''~")

        time.sleep(wait_t)

        if config == 0:
            app.fatty.type_keys("tmux{VK_SPACE}a{VK_SPACE}-t{VK_SPACE}orca~")
        else:
            app.fatty.type_keys("tmux{VK_SPACE}a{VK_SPACE}-t{VK_SPACE}orca2~")

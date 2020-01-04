from pywinauto import application, mouse
import time
import os, sys
import win32api

from Misc import processArguments, sortKey

if __name__ == '__main__':
    params = {
        'exe_path': '',
        'auth_path': '',
        'config': 0,
        'wait_t': 10,
    }
    processArguments(sys.argv[1:], params)
    exe_path = params['exe_path']
    auth_path = params['auth_path']
    config = params['config']
    wait_t = params['wait_t']

    # app_t = application.Application().start("cmd.exe")
    # app_t = application.Application().start("C:\\Users\\Tommy\\Desktop\\startup_progs\\t.lnk")
    # app_t = application.Application().start("H:\\UofA\\MSc\\Code\\TrackingFramework\\scripts\\t.cmd")    
    # app_t = application.Application().start("C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe")
    # app_t = application.Application().start("C:\\Essentials\\ConEmu\\ConEmu64.exe")
    # app_t = application.Application().start("C:\\Program Files (x86)\\PowerCmd\\PowerCmd.exe")
    # app_t.window().maximize()

    # sys.exit()

    auth_data = open(auth_path, 'r').readlines()
    auth_data = [k.strip() for k in auth_data]

    name00, name01, pwd0 = auth_data[0].split(' ')
    name10, name11, pwd1 = auth_data[1].split(' ')
    name20, name21, pwd2 = auth_data[2].split(' ')

    print('Setting up system 0 with tmux sessions {}, {} and pwd {}'.format(name00, name01, pwd0))
    print('Setting up system 1 with tmux sessions {}, {} and pwd {}'.format(name10, name11, pwd1))
    print('Setting up system 2 with tmux sessions {}, {} and pwd {}'.format(name20, name21, pwd2))

    # sys.exit()

    if not exe_path:
        raise IOError('Terminal executable path must be provided')

    app = application.Application().start(exe_path)
    app.window().maximize()
    if config == -1:

        mouse_x, mouse_y = win32api.GetCursorPos()

        apps = [app, ]
        app2 = application.Application().start(exe_path)
        app2.window().maximize()

        apps.append(app2)

        for _app in apps:
            # _app.fatty.type_keys("t~")
            # _app.fatty.type_keys("^+t")
            # _app.fatty.type_keys("f~")
            # _app.fatty.type_keys("^+t")
            _app.fatty.type_keys("sstg{VK_SPACE}tb~")
            _app.fatty.type_keys("sudo{VK_SPACE}-s~")
            _app.fatty.type_keys("%s~" % pwd0)

            # time.sleep(1)

        app3 = application.Application().start(exe_path)
        app3.window().maximize()
        app3.fatty.type_keys("tmux{VK_SPACE}new~")

        time.sleep(5)

        app.fatty.type_keys("tmux{VK_SPACE}attach{VK_SPACE}-d{VK_SPACE}-t{VK_SPACE}%s~" % name00)
        app2.fatty.type_keys("tmux{VK_SPACE}attach{VK_SPACE}-d{VK_SPACE}-t{VK_SPACE}%s~" % name01)

        for _app in apps:
            _app.fatty.type_keys("^+t")
            _app.fatty.type_keys("sstg2~")
            _app.fatty.type_keys("sstz~")
            _app.fatty.type_keys("sudo{VK_SPACE}-s~")
            _app.fatty.type_keys("%s~" % pwd1)

        time.sleep(5)

        app.fatty.type_keys("tmux{VK_SPACE}attach{VK_SPACE}-d{VK_SPACE}-t{VK_SPACE}%s~" % name10)
        app2.fatty.type_keys("tmux{VK_SPACE}attach{VK_SPACE}-d{VK_SPACE}-t{VK_SPACE}%s~" % name11)

        # time.sleep(1)

        # time.sleep(1)
        # app.fatty.type_keys("+{RIGHT}")
        # app2.fatty.type_keys("+{RIGHT}")

        for _app in apps:
            _app.fatty.type_keys("^+t")
            _app.fatty.type_keys("sstg3~")
            _app.fatty.type_keys("sstx~")
            _app.fatty.type_keys("sudo{VK_SPACE}-s~")
            _app.fatty.type_keys("%s~" % pwd2)

        time.sleep(5)

        app.fatty.type_keys("tmux{VK_SPACE}attach{VK_SPACE}-d{VK_SPACE}-t{VK_SPACE}%s~" % name20)
        app2.fatty.type_keys("tmux{VK_SPACE}attach{VK_SPACE}-d{VK_SPACE}-t{VK_SPACE}%s~" % name21)

        app3.fatty.type_keys("^b^r")

        mouse.move(coords=(mouse_x, mouse_y))

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
        app.fatty.type_keys("%s~" % pwd0)
        time.sleep(2)

        if config == 0:
            app.fatty.type_keys("tmux{VK_SPACE}attach{VK_SPACE}-d{VK_SPACE}-t{VK_SPACE}grs~")
        else:
            app.fatty.type_keys("tmux{VK_SPACE}attach{VK_SPACE}-d{VK_SPACE}-t{VK_SPACE}grs2~")

        app.fatty.type_keys("^+t")
        app.fatty.type_keys("sstg2~")
        app.fatty.type_keys("sstz~")
        app.fatty.type_keys("sudo{VK_SPACE}-s~")
        app.fatty.type_keys("%s~" % pwd1)

        time.sleep(wait_t)

        if config == 0:
            app.fatty.type_keys("tmux{VK_SPACE}attach{VK_SPACE}-d{VK_SPACE}-t{VK_SPACE}orca~")
        else:
            app.fatty.type_keys("tmux{VK_SPACE}attach{VK_SPACE}-d{VK_SPACE}-t{VK_SPACE}orca2~")

    while True:
        k = input('Enter any key to restore ssh connections')
        # print('Enter any key to restore ssh connections')
        # os.system("pause")

        mouse_x, mouse_y = win32api.GetCursorPos()

        for _app in apps:
            _app.fatty.type_keys("^+w")
            _app.fatty.type_keys("^+w")
            _app.fatty.type_keys("sstg{VK_SPACE}tb~")
            _app.fatty.type_keys("sudo{VK_SPACE}-s~")
            _app.fatty.type_keys("%s~" % pwd0)

        time.sleep(2)

        app.fatty.type_keys("tmux{VK_SPACE}attach{VK_SPACE}-d{VK_SPACE}-t{VK_SPACE}%s~" % name00)
        app2.fatty.type_keys("tmux{VK_SPACE}attach{VK_SPACE}-d{VK_SPACE}-t{VK_SPACE}%s~" % name01)

        for _app in apps:
            _app.fatty.type_keys("^+t")
            _app.fatty.type_keys("sstg2~")
            _app.fatty.type_keys("sstz~")
            _app.fatty.type_keys("sudo{VK_SPACE}-s~")
            _app.fatty.type_keys("%s~" % pwd1)

        time.sleep(2)

        app.fatty.type_keys("tmux{VK_SPACE}attach{VK_SPACE}-d{VK_SPACE}-t{VK_SPACE}%s~" % name10)
        app2.fatty.type_keys("tmux{VK_SPACE}attach{VK_SPACE}-d{VK_SPACE}-t{VK_SPACE}%s~" % name11)

        # time.sleep(1)

        # time.sleep(1)
        # app.fatty.type_keys("+{RIGHT}")
        # app2.fatty.type_keys("+{RIGHT}")

        for _app in apps:
            _app.fatty.type_keys("^+t")
            _app.fatty.type_keys("sstg3~")
            _app.fatty.type_keys("sstx~")
            _app.fatty.type_keys("sudo{VK_SPACE}-s~")
            _app.fatty.type_keys("%s~" % pwd2)

        time.sleep(2)

        app.fatty.type_keys("tmux{VK_SPACE}attach{VK_SPACE}-d{VK_SPACE}-t{VK_SPACE}%s~" % name20)
        app2.fatty.type_keys("tmux{VK_SPACE}attach{VK_SPACE}-d{VK_SPACE}-t{VK_SPACE}%s~" % name21)

        mouse.move(coords=(mouse_x, mouse_y))

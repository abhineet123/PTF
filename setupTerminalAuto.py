from pywinauto import application, mouse
import time
import os, sys
import win32api
import pyautogui

import paramparse

from Misc import processArguments, linux_path
import encrypt_file_aes as encryption


def half_sized_window(half_type):
    pyautogui.keyDown('ctrlleft')
    pyautogui.keyDown('winleft')
    pyautogui.keyDown('altleft')

    if half_type == 1:
        pyautogui.press('right')
    else:
        pyautogui.press('left')

    pyautogui.keyUp('ctrlleft')
    pyautogui.keyUp('winleft')
    pyautogui.keyUp('altleft')


if __name__ == '__main__':
    # params = {
    #     'exe_path': 'fatty.exe',
    #     'key_root': '',
    #     'key_dir': '',
    #     'auth_root': '',
    #     'auth_dir': '',
    #     'auth_file': '',
    #     'config': 0,
    #     'pwd_wait': 2,
    #     'wait_t': 3,
    #     'n_git_panes': 8,
    #     'git_half': 1,
    #     'git_wait': 0.5,
    #     'git_wait_init': 10,
    #     'git_wait_restore': 20,
    #     'only_git': 0,
    #     'enable_git': 0,
    #     'git_postproc': 0,
    #     'git_cmds': '',
    # }
    # paramparse.from_dict(params, to_clipboard=1)
    # exit()

    class Params(paramparse.CFG):
        """
        :ivar auth_dir:
        :type auth_dir: str

        :ivar auth_file:
        :type auth_file: str

        :ivar auth_root:
        :type auth_root: str

        :ivar config:
        :type config: int

        :ivar enable_git:
        :type enable_git: int

        :ivar exe_path:
        :type exe_path: str

        :ivar git_cmds:
        :type git_cmds: str

        :ivar git_half:
        :type git_half: int

        :ivar git_postproc:
        :type git_postproc: int

        :ivar git_wait:
        :type git_wait: float

        :ivar git_wait_init:
        :type git_wait_init: int

        :ivar git_wait_restore:
        :type git_wait_restore: int

        :ivar key_dir:
        :type key_dir: str

        :ivar key_root:
        :type key_root: str

        :ivar n_git_panes:
        :type n_git_panes: int

        :ivar only_git:
        :type only_git: int

        :ivar pwd_wait:
        :type pwd_wait: int

        :ivar wait_t:
        :type wait_t: int

        """

        def __init__(self):
            super().__init__()
            self.cfg = ()
            self.auth_dir = ''
            self.auth_file = ''
            self.auth_root = ''
            self.config = 0
            self.enable_isaic = 1
            self.isaic = ('isc', 'isc2')
            self.enable_git = 0
            self.exe_path = 'fatty.exe'
            self.git_cmds = ''
            self.git_half = 1
            self.git_postproc = 0
            self.git_wait = 0.5
            self.git_wait_init = 10
            self.git_wait_restore = 20
            self.key_dir = ''
            self.key_root = ''
            self.n_git_panes = 8
            self.only_git = 0
            self.pwd_wait = 2
            self.wait_t = 3


    params = Params()
    paramparse.process(params)

    # processArguments(sys.argv[1:], params)
    exe_path = params.exe_path
    key_root = params.key_root
    key_dir = params.key_dir
    auth_root = params.auth_root
    auth_dir = params.auth_dir
    auth_file = params.auth_file
    config = params.config
    wait_t = params.wait_t
    n_git_panes = params.n_git_panes
    enable_isaic = params.enable_isaic
    isaic1, isaic2 = params.isaic
    enable_git = params.enable_git
    git_postproc = params.git_postproc
    only_git = params.only_git
    git_wait = params.git_wait
    git_wait_init = params.git_wait_init
    git_wait_restore = params.git_wait_restore
    git_cmds = params.git_cmds
    git_half = params.git_half
    pwd_wait = params.pwd_wait

    # app_t = application.Application().start("cmd.exe")
    # app_t = application.Application().start("C:\\Users\\Tommy\\Desktop\\startup_progs\\t.lnk")
    # app_t = application.Application().start("H:\\UofA\\MSc\\Code\\TrackingFramework\\scripts\\t.cmd")    
    # app_t = application.Application().start("C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe")
    # app_t = application.Application().start("C:\\Essentials\\ConEmu\\ConEmu64.exe")
    # app_t = application.Application().start("C:\\Program Files (x86)\\PowerCmd\\PowerCmd.exe")
    # app_t.window().maximize()

    _ = input('Enter any key to continue')

    start_t = time.time()

    # sys.exit()
    assert exe_path, 'Terminal executable path must be provided'

    if not only_git:
        auth_path = linux_path(auth_root, auth_dir, auth_file)
        auth_data = open(auth_path, 'r').readlines()
        auth_data = [k.strip() for k in auth_data]

        dst0_info = auth_data[0].split(' ')
        name00, name01, ecr0, key0 = dst0_info[:4]

        name10, name11, ecr1, key1 = auth_data[1].split(' ')[:4]
        name20, name21, ecr2, key2 = auth_data[2].split(' ')[:4]

        key0_path = linux_path(key_root, key_dir, key0)
        key1_path = linux_path(key_root, key_dir, key1)
        key2_path = linux_path(key_root, key_dir, key2)

        encryption_params = encryption.Params()
        encryption_params.mode = 1
        encryption_params.root_dir = key_root
        encryption_params.parent_dir = key_dir

        encryption_params.in_file = ecr0
        encryption_params.key_file = key0
        encryption_params.process()
        pwd0 = encryption.run(encryption_params)

        encryption_params.in_file = ecr1
        encryption_params.key_file = key1
        encryption_params.process()
        pwd1 = encryption.run(encryption_params)

        encryption_params.in_file = ecr2
        encryption_params.key_file = key2
        encryption_params.process()
        pwd2 = encryption.run(encryption_params)

        print('Setting up system 0 with tmux sessions {}, {}'.format(name00, name01))
        print('Setting up system 1 with tmux sessions {}, {}'.format(name10, name11))
        print('Setting up system 2 with tmux sessions {}, {}'.format(name20, name21))

        # print(pwd0)
        # print(pwd1)
        # print(pwd2)

        # exit()

        servers_app = application.Application().start(exe_path)
        servers_app.window().maximize()

        half_sized_window(half_type=2)

    if config == -1:

        mouse_x, mouse_y = win32api.GetCursorPos()

        if not only_git:
            """connect to grs"""
            apps = [servers_app, ]
            servers_app2 = application.Application().start(exe_path)
            servers_app2.window().maximize()
            half_sized_window(half_type=1)

            apps.append(servers_app2)

            for _app in apps:
                # _app.fatty.type_keys("t~")
                # _app.fatty.type_keys("^+t")
                # _app.fatty.type_keys("f~")
                # _app.fatty.type_keys("^+t")
                _app.fatty.type_keys("sstg{VK_SPACE}tb~")
                _app.fatty.type_keys("sudo{VK_SPACE}-s~")

                # _app.fatty.type_keys("%s~" % pwd0)
                time.sleep(pwd_wait)
                pyautogui.write(pwd0)
                pyautogui.press('enter')

                # time.sleep(1)

        if enable_git:
            git_app = application.Application().start(exe_path)
            git_app.window().maximize()

            if git_half:
                half_sized_window(git_half)

                # pyautogui.keyDown('ctrlleft')
                # pyautogui.keyDown('winleft')
                # pyautogui.keyDown('altleft')
                #
                # if git_half == 2:
                #     pyautogui.press('left')
                # else:
                #     pyautogui.press('right')
                #
                # pyautogui.keyUp('ctrlleft')
                # pyautogui.keyUp('winleft')
                # pyautogui.keyUp('altleft')

            git_app.fatty.type_keys("tmux{VK_SPACE}new~")

        if not only_git:
            time.sleep(wait_t)
            servers_app.fatty.type_keys("tmux{VK_SPACE}attach{VK_SPACE}-d{VK_SPACE}-t{VK_SPACE}%s~" % name00)
            servers_app2.fatty.type_keys("tmux{VK_SPACE}attach{VK_SPACE}-d{VK_SPACE}-t{VK_SPACE}%s~" % name01)

            """connect to orca"""
            for _app in apps:
                _app.fatty.type_keys("^+t")
                _app.fatty.type_keys("sstg2~")
                # time.sleep(2)
                _app.fatty.type_keys("sstz~")
                # time.sleep(2)
                _app.fatty.type_keys("sudo{VK_SPACE}-s~")
                # time.sleep(2)
                # _app.fatty.type_keys("%s~" % pwd1)

                time.sleep(pwd_wait)
                pyautogui.write(pwd1)
                pyautogui.press('enter')

            time.sleep(wait_t)
            servers_app.fatty.type_keys("tmux{VK_SPACE}attach{VK_SPACE}-d{VK_SPACE}-t{VK_SPACE}%s~" % name10)
            servers_app2.fatty.type_keys("tmux{VK_SPACE}attach{VK_SPACE}-d{VK_SPACE}-t{VK_SPACE}%s~" % name11)

            # time.sleep(1)

            # time.sleep(1)
            # app.fatty.type_keys("+{RIGHT}")
            # app2.fatty.type_keys("+{RIGHT}")

            """connect to x99"""
            for _app in apps:
                _app.fatty.type_keys("^+t")
                _app.fatty.type_keys("sstg3~")
                # time.sleep(2)

                _app.fatty.type_keys("sstx~")
                # time.sleep(2)

                _app.fatty.type_keys("sudo{VK_SPACE}-s~")
                # time.sleep(2)

                time.sleep(pwd_wait)
                pyautogui.write(pwd2)
                pyautogui.press('enter')

                # _app.fatty.type_keys("%s~" % pwd2)

            time.sleep(wait_t)

            servers_app.fatty.type_keys("tmux{VK_SPACE}attach{VK_SPACE}-d{VK_SPACE}-t{VK_SPACE}%s~" % name20)
            servers_app2.fatty.type_keys("tmux{VK_SPACE}attach{VK_SPACE}-d{VK_SPACE}-t{VK_SPACE}%s~" % name21)

            if enable_isaic:
                """connect to isaic"""
                for _app in apps:
                    _app.fatty.type_keys("^+t")
                    _app.fatty.type_keys("sshi~")
                    _app.fatty.type_keys("sudo{VK_SPACE}-s~")

                time.sleep(wait_t)

                servers_app.fatty.type_keys("tmux{VK_SPACE}attach{VK_SPACE}-d{VK_SPACE}-t{VK_SPACE}%s~" % isaic1)
                servers_app2.fatty.type_keys("tmux{VK_SPACE}attach{VK_SPACE}-d{VK_SPACE}-t{VK_SPACE}%s~" % isaic2)

        if enable_git:
            if only_git:
                print('waiting {} secs for git init'.format(git_wait_init))
                time.sleep(git_wait_init)

            # git_app.fatty.type_keys("^b^r")

            if git_postproc:
                # print('waiting {} secs for git restore'.format(git_wait_restore))
                # time.sleep(git_wait_restore)

                print('running git post proc...')

                if git_cmds:
                    git_cmds_list = [k.strip() for k in open(git_cmds, 'r').readlines() if k.strip()]
                    for git_cmd in git_cmds_list:

                        if not git_cmd:
                            continue

                        print('git_cmd: {}'.format(git_cmd))

                        if git_cmd == '__0__':
                            git_app.fatty.type_keys("^b")
                            git_app.fatty.type_keys("q")
                            git_app.fatty.type_keys("0")
                        elif git_cmd == '__enter__':
                            # pyautogui.press('enter')
                            git_app.fatty.type_keys("{ENTER}")
                        elif git_cmd == '__git__':
                            # pass
                            git_app.fatty.type_keys("./gitu.sh{VK_SPACE}f")
                        elif git_cmd == '__vert__':
                            git_app.fatty.type_keys("^b")
                            git_app.fatty.type_keys('{%}')
                        elif git_cmd == '__horz__':
                            git_app.fatty.type_keys("^b")
                            git_app.fatty.type_keys('{"}')
                        elif git_cmd == '__up__':
                            git_app.fatty.type_keys("^b")
                            git_app.fatty.type_keys("{UP}")
                        elif git_cmd == '__down__':
                            git_app.fatty.type_keys("^b")
                            git_app.fatty.type_keys("{DOWN}")
                        elif git_cmd == '__right__':
                            git_app.fatty.type_keys("^b")
                            git_app.fatty.type_keys("{RIGHT}")
                        elif git_cmd == '__left__':
                            git_app.fatty.type_keys("^b")
                            git_app.fatty.type_keys("{LEFT}")
                        else:
                            pyautogui.write(git_cmd)
                            # pyautogui.press('enter')

                        time.sleep(git_wait)
                else:
                    for _ in range(n_git_panes):
                        git_app.fatty.type_keys("{UP}~")
                        git_app.fatty.type_keys("./gitu.bat{VK_SPACE}f")
                        time.sleep(git_wait)
                        git_app.fatty.type_keys("^b{DOWN}")
                        time.sleep(git_wait)

                    git_app.fatty.type_keys("^b{LEFT}")
                    time.sleep(git_wait)

                    for _ in range(n_git_panes):
                        git_app.fatty.type_keys("{UP}~")
                        git_app.fatty.type_keys("./gitu.bat{VK_SPACE}f")
                        time.sleep(git_wait)
                        git_app.fatty.type_keys("^b{DOWN}")
                        time.sleep(git_wait)
                print('done')

        mouse.move(coords=(mouse_x, mouse_y))

    elif config == 1:
        servers_app.fatty.type_keys("t~")
        servers_app.fatty.type_keys("^+t")
        servers_app.fatty.type_keys("f~")
        servers_app.fatty.type_keys("^+t")
        servers_app.fatty.type_keys("sstg{VK_SPACE}tb~")
        servers_app.fatty.type_keys("sudo{VK_SPACE}-s~")
        servers_app.fatty.type_keys("%s~" % pwd0)
        time.sleep(2)

        if config == 0:
            servers_app.fatty.type_keys("tmux{VK_SPACE}attach{VK_SPACE}-d{VK_SPACE}-t{VK_SPACE}grs~")
        else:
            servers_app.fatty.type_keys("tmux{VK_SPACE}attach{VK_SPACE}-d{VK_SPACE}-t{VK_SPACE}grs2~")

        servers_app.fatty.type_keys("^+t")
        servers_app.fatty.type_keys("sstg2~")
        servers_app.fatty.type_keys("sstz~")
        servers_app.fatty.type_keys("sudo{VK_SPACE}-s~")
        servers_app.fatty.type_keys("%s~" % pwd1)

        time.sleep(wait_t)

        if config == 0:
            servers_app.fatty.type_keys("tmux{VK_SPACE}attach{VK_SPACE}-d{VK_SPACE}-t{VK_SPACE}orca~")
        else:
            servers_app.fatty.type_keys("tmux{VK_SPACE}attach{VK_SPACE}-d{VK_SPACE}-t{VK_SPACE}orca2~")

    elif config == 2:
        servers_app.fatty.type_keys("tmux{VK_SPACE}new~")
        time.sleep(wait_t)
        servers_app.fatty.type_keys("^b^r")

    elif config == 3:
        servers_app.fatty.type_keys("^+t")
        servers_app.fatty.type_keys("sstb~")
        servers_app.fatty.type_keys("source{VK_SPACE}envpy36/bin/activate~")
        servers_app.fatty.type_keys("tmux{VK_SPACE}a~")
        servers_app.fatty.type_keys("tmux{VK_SPACE}new~")

        servers_app.fatty.type_keys("^+t")
        servers_app.fatty.type_keys("sstc~")
        servers_app.fatty.type_keys("source{VK_SPACE}envpy36/bin/activate~")
        servers_app.fatty.type_keys("tmux{VK_SPACE}a~")
        servers_app.fatty.type_keys("tmux{VK_SPACE}new~")

        servers_app.fatty.type_keys("^+t")
        servers_app.fatty.type_keys("sstgr~")
        servers_app.fatty.type_keys("source{VK_SPACE}envpy36/bin/activate~")
        servers_app.fatty.type_keys("tmux{VK_SPACE}a~")
        servers_app.fatty.type_keys("tmux{VK_SPACE}new~")

    end_t = time.time()

    print('setup time: {}'.format(end_t - start_t))

    if only_git:
        k = input('Enter any key to terminate tmux-git')

        git_app.fatty.type_keys("^b")
        git_app.fatty.type_keys("{d}")
        time.sleep(1)
        git_app.fatty.type_keys("tmux{VK_SPACE}kill-server")
        git_app.fatty.type_keys("{ENTER}")
        git_app.fatty.type_keys("taskkill{VK_SPACE}/F{VK_SPACE}/IM{VK_SPACE}bash.exe")
        git_app.fatty.type_keys("{ENTER}")

        time.sleep(2)
        git_app.window().close()

        sys.exit()

    while True:
        k = input('Enter any key to restore ssh connections')
        # print('Enter any key to restore ssh connections')
        # os.system("pause")

        mouse_x, mouse_y = win32api.GetCursorPos()

        if config == -1:

            for _app in apps:
                """grs"""
                _app.fatty.type_keys("^+w")
                _app.fatty.type_keys("^+w")
                if enable_isaic:
                    _app.fatty.type_keys("^+w")

                time.sleep(1)

                _app.fatty.type_keys("sstg{VK_SPACE}tb~")
                _app.fatty.type_keys("sudo{VK_SPACE}-s~")

                time.sleep(pwd_wait)
                # _app.fatty.type_keys("%s~" % pwd0)
                pyautogui.write(pwd0)
                pyautogui.press('enter')

            time.sleep(wait_t)

            servers_app.fatty.type_keys("tmux{VK_SPACE}attach{VK_SPACE}-d{VK_SPACE}-t{VK_SPACE}%s~" % name00)
            servers_app2.fatty.type_keys("tmux{VK_SPACE}attach{VK_SPACE}-d{VK_SPACE}-t{VK_SPACE}%s~" % name01)

            for _app in apps:
                """orca"""
                _app.fatty.type_keys("^+t")
                _app.fatty.type_keys("sstg2~")
                _app.fatty.type_keys("sstz~")
                _app.fatty.type_keys("sudo{VK_SPACE}-s~")

                time.sleep(pwd_wait)
                # _app.fatty.type_keys("%s~" % pwd1)
                pyautogui.write(pwd1)
                pyautogui.press('enter')

            time.sleep(wait_t)

            servers_app.fatty.type_keys("tmux{VK_SPACE}attach{VK_SPACE}-d{VK_SPACE}-t{VK_SPACE}%s~" % name10)
            servers_app2.fatty.type_keys("tmux{VK_SPACE}attach{VK_SPACE}-d{VK_SPACE}-t{VK_SPACE}%s~" % name11)

            # time.sleep(1)

            # time.sleep(1)
            # app.fatty.type_keys("+{RIGHT}")
            # app2.fatty.type_keys("+{RIGHT}")

            for _app in apps:
                """x99"""
                _app.fatty.type_keys("^+t")
                _app.fatty.type_keys("sstg3~")
                _app.fatty.type_keys("sstx~")
                _app.fatty.type_keys("sudo{VK_SPACE}-s~")

                time.sleep(pwd_wait)
                # _app.fatty.type_keys("%s~" % pwd2)
                pyautogui.write(pwd2)
                pyautogui.press('enter')

            time.sleep(wait_t)

            servers_app.fatty.type_keys("tmux{VK_SPACE}attach{VK_SPACE}-d{VK_SPACE}-t{VK_SPACE}%s~" % name20)
            servers_app2.fatty.type_keys("tmux{VK_SPACE}attach{VK_SPACE}-d{VK_SPACE}-t{VK_SPACE}%s~" % name21)

            if enable_isaic:
                for _app in apps:
                    """isaic"""
                    _app.fatty.type_keys("^+t")
                    _app.fatty.type_keys("sshi~")
                    _app.fatty.type_keys("sudo{VK_SPACE}-s~")

                time.sleep(wait_t)

                servers_app.fatty.type_keys("tmux{VK_SPACE}attach{VK_SPACE}-d{VK_SPACE}-t{VK_SPACE}%s~" % isaic1)
                servers_app2.fatty.type_keys("tmux{VK_SPACE}attach{VK_SPACE}-d{VK_SPACE}-t{VK_SPACE}%s~" % isaic2)

        elif config == 3:

            servers_app.fatty.type_keys("^+w")
            servers_app.fatty.type_keys("^+w")
            servers_app.fatty.type_keys("^+w")

            servers_app.fatty.type_keys("^+t")
            servers_app.fatty.type_keys("sstb~")
            servers_app.fatty.type_keys("source{VK_SPACE}envpy36/bin/activate~")
            servers_app.fatty.type_keys("tmux{VK_SPACE}a~")
            servers_app.fatty.type_keys("tmux{VK_SPACE}new~")

            servers_app.fatty.type_keys("^+t")
            servers_app.fatty.type_keys("sstc~")
            servers_app.fatty.type_keys("source{VK_SPACE}envpy36/bin/activate~")
            servers_app.fatty.type_keys("tmux{VK_SPACE}a~")
            servers_app.fatty.type_keys("tmux{VK_SPACE}new~")

            servers_app.fatty.type_keys("^+t")
            servers_app.fatty.type_keys("sstgr~")
            servers_app.fatty.type_keys("source{VK_SPACE}envpy36/bin/activate~")
            servers_app.fatty.type_keys("tmux{VK_SPACE}a~")
            servers_app.fatty.type_keys("tmux{VK_SPACE}new~")

        mouse.move(coords=(mouse_x, mouse_y))

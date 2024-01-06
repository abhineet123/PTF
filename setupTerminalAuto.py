from pywinauto import application, mouse
import time
import sys
import win32api
import pyautogui

import paramparse

from Misc import linux_path
import encrypt_file_aes as encryption


class Params(paramparse.CFG):

    def __init__(self):
        super().__init__()
        self.cfg = ()
        self.auth_dir = ''
        self.auth_file = ''
        self.auth_root = ''
        self.config = 0
        self.enable_e5g = 0
        self.enable_mj = 0
        self.enable_mj2 = 0
        self.enable_nrw = 0
        self.enable_isaic = 0
        self.isaic = ('isc', 'isc2')
        self.enable_git = 0
        self.exe_path = ''
        self.backend = 'win32'
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
        self.sudo = 0
        self.pwd_wait = 2
        self.wait_t = 3


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


def main():
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
    enable_e5g = params.enable_e5g
    enable_mj = params.enable_mj
    enable_mj2 = params.enable_mj2
    enable_nrw = params.enable_nrw
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
    sudo = params.sudo

    # app_t = application.Application().start("cmd.exe")
    # app_t = application.Application().start("C:\\Users\\Tommy\\Desktop\\startup_progs\\t.lnk")
    # app_t = application.Application().start("H:\\UofA\\MSc\\Code\\TrackingFramework\\scripts\\t.cmd")    
    # app_t = application.Application().start("C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe")
    # app_t = application.Application().start("C:\\Essentials\\ConEmu\\ConEmu64.exe")
    # app_t = application.Application().start("C:\\Program Files (x86)\\PowerCmd\\PowerCmd.exe")
    # app_t.window().maximize()

    if only_git:
        _ = input('Enter any key to continue setting up git')
    else:
        _ = input('Enter any key to continue setting up terminals')

    start_t = time.time()

    # sys.exit()
    assert exe_path, 'Terminal executable path must be provided'

    if not only_git:
        auth_path = linux_path(auth_root, auth_dir, auth_file)
        auth_data = open(auth_path, 'r').read().splitlines()

        auth_data = [k.split(' ') for k in auth_data]

        auth_data = {
            k[0]: k for k in auth_data
        }

        grs_1, grs_2 = auth_data['grs'][:2]
        x99_1, x99_2 = auth_data['x99'][:2]
        e5g_1, e5g_2 = auth_data['e5g'][:2]
        nrw_1, nrw_2 = auth_data['nrw'][:2]
        mj1_1, mj1_2 = auth_data['mj1'][:2]
        mj2_1, mj2_2 = auth_data['mj2'][:2]

        pwd0 = pwd1 = pwd2 = None

        if sudo:
            ecr = {
                k: v[-2] for k, v in auth_data.items()
            }
            key = {
                k: v[-1] for k, v in auth_data.items()
            }

            encryption_params = encryption.Params()
            encryption_params.mode = 1
            encryption_params.root_dir = key_root
            encryption_params.parent_dir = key_dir

            # encryption_params.in_file = ecr0
            # encryption_params.key_file = key0
            # encryption_params.process()
            # pwd0 = encryption.run(encryption_params)
            #
            # encryption_params.in_file = ecr1
            # encryption_params.key_file = key1
            # encryption_params.process()
            # pwd1 = encryption.run(encryption_params)
            #
            # encryption_params.in_file = ecr2
            # encryption_params.key_file = key2
            # encryption_params.process()
            # pwd2 = encryption.run(encryption_params)

        servers_app = application.Application(backend=params.backend).start(exe_path)
        servers_app.window().maximize()

        # half_sized_window(half_type=2)

    if config == -1:

        mouse_x, mouse_y = win32api.GetCursorPos()

        if not only_git:
            """connect to grs"""
            apps = [servers_app, ]
            servers_app2 = application.Application(backend=params.backend).start(exe_path)
            # servers_app2.window().maximize()
            # half_sized_window(half_type=1)

            apps.append(servers_app2)

            for _app_id, _app in enumerate(apps):
                # _app.fatty.type_keys("t~")
                # _app.fatty.type_keys("^+t")
                # _app.fatty.type_keys("f~")
                # _app.fatty.type_keys("^+t")
                _app.fatty.type_keys(f"sstg{_app_id}~")
                if sudo:
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

            git_app.fatty.type_keys("cd{VK_SPACE}/~")
            git_app.fatty.type_keys("tmux{VK_SPACE}new~")

        if not only_git:
            time.sleep(wait_t)
            servers_app.fatty.type_keys("tmux{VK_SPACE}attach{VK_SPACE}-d{VK_SPACE}-t{VK_SPACE}%s~" % grs_1)
            servers_app2.fatty.type_keys("tmux{VK_SPACE}attach{VK_SPACE}-d{VK_SPACE}-t{VK_SPACE}%s~" % grs_2)

            # time.sleep(1)

            # time.sleep(1)
            # app.fatty.type_keys("+{RIGHT}")
            # app2.fatty.type_keys("+{RIGHT}")

            """connect to x99"""
            for _app_id, _app in enumerate(apps):
                _app.fatty.type_keys("^+t")
                _app.fatty.type_keys(f"sstx{_app_id}~")

                if sudo:
                    # time.sleep(2)

                    _app.fatty.type_keys("sudo{VK_SPACE}-s~")
                    # time.sleep(2)

                    time.sleep(pwd_wait)
                    pyautogui.write(pwd2)
                    pyautogui.press('enter')

                # _app.fatty.type_keys("%s~" % pwd2)

            time.sleep(wait_t)

            servers_app.fatty.type_keys("tmux{VK_SPACE}attach{VK_SPACE}-d{VK_SPACE}-t{VK_SPACE}%s~" % x99_1)
            servers_app2.fatty.type_keys("tmux{VK_SPACE}attach{VK_SPACE}-d{VK_SPACE}-t{VK_SPACE}%s~" % x99_2)

            if enable_e5g:
                """connect to e5g"""
                for _app_id, _app in enumerate(apps):
                    _app.fatty.type_keys("^+t")
                    # _app.fatty.type_keys("sstg2~")
                    # time.sleep(2)
                    _app.fatty.type_keys(f"sste{_app_id}~")

                    if sudo:
                        # time.sleep(2)
                        _app.fatty.type_keys("sudo{VK_SPACE}-s~")
                        # time.sleep(2)
                        # _app.fatty.type_keys("%s~" % pwd1)

                        time.sleep(pwd_wait)
                        pyautogui.write(pwd1)
                        pyautogui.press('enter')

                time.sleep(wait_t)
                servers_app.fatty.type_keys("tmux{VK_SPACE}attach{VK_SPACE}-d{VK_SPACE}-t{VK_SPACE}%s~" % e5g_1)
                servers_app2.fatty.type_keys("tmux{VK_SPACE}attach{VK_SPACE}-d{VK_SPACE}-t{VK_SPACE}%s~" % e5g_2)

            if enable_isaic:
                """connect to isaic"""
                for _app in apps:
                    _app.fatty.type_keys("^+t")
                    _app.fatty.type_keys("sshi~")
                    _app.fatty.type_keys("sudo{VK_SPACE}-s~")

                time.sleep(wait_t)

                servers_app.fatty.type_keys("tmux{VK_SPACE}attach{VK_SPACE}-d{VK_SPACE}-t{VK_SPACE}%s~" % isaic1)
                servers_app2.fatty.type_keys("tmux{VK_SPACE}attach{VK_SPACE}-d{VK_SPACE}-t{VK_SPACE}%s~" % isaic2)

            if enable_mj:
                """connect to mj server"""
                for _app_id, _app in enumerate(apps):
                    _app.fatty.type_keys("^+t")
                    _app.fatty.type_keys(f"sshm{_app_id}~")

                time.sleep(wait_t)

                servers_app.fatty.type_keys("tmux{VK_SPACE}attach{VK_SPACE}-d{VK_SPACE}-t{VK_SPACE}%s~" % mj1_1)
                servers_app2.fatty.type_keys("tmux{VK_SPACE}attach{VK_SPACE}-d{VK_SPACE}-t{VK_SPACE}%s~" % mj1_2)

            if enable_mj2:
                """connect to mj server"""
                for _app_id, _app in enumerate(apps):
                    _app.fatty.type_keys("^+t")
                    _app.fatty.type_keys(f"sshm2{_app_id}~")

                time.sleep(wait_t)

                servers_app.fatty.type_keys("tmux{VK_SPACE}attach{VK_SPACE}-d{VK_SPACE}-t{VK_SPACE}%s~" % mj2_1)
                servers_app2.fatty.type_keys("tmux{VK_SPACE}attach{VK_SPACE}-d{VK_SPACE}-t{VK_SPACE}%s~" % mj2_2)

            if enable_nrw:
                """connect to mj server"""
                for _app in apps:
                    _app.fatty.type_keys("^+t")
                    _app.fatty.type_keys("sshnrw~")

                time.sleep(wait_t)

                # servers_app.fatty.type_keys("tmux{VK_SPACE}a~")
                # servers_app.fatty.type_keys("^br")
                # time.sleep(5)
                # servers_app2.fatty.type_keys("tmux{VK_SPACE}a~")

                servers_app.fatty.type_keys("tmux{VK_SPACE}attach{VK_SPACE}-d{VK_SPACE}-t{VK_SPACE}%s~" % nrw_1)
                servers_app2.fatty.type_keys("tmux{VK_SPACE}attach{VK_SPACE}-d{VK_SPACE}-t{VK_SPACE}%s~" % nrw_2)

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
        servers_app.fatty.type_keys("sstg~")
        if sudo:
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
        if sudo:
            servers_app.fatty.type_keys("sudo{VK_SPACE}-s~")
            servers_app.fatty.type_keys("%s~" % pwd1)

            time.sleep(wait_t)

        if config == 0:
            servers_app.fatty.type_keys("tmux{VK_SPACE}attach{VK_SPACE}-d{VK_SPACE}-t{VK_SPACE}orca~")
        else:
            servers_app.fatty.type_keys("tmux{VK_SPACE}attach{VK_SPACE}-d{VK_SPACE}-t{VK_SPACE}orca2~")

    elif config == 2:
        git_app.fatty.type_keys("cd /")
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
        while True:
            k = input(f'Enter Q to terminate git\n')

            if k in ['q', 'Q', ord('q'), ord('Q')]:
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

    iter_id = -1
    while True:
        iter_id += 1

        k = input(f'{iter_id} : Enter Q to terminate ssh\n')

        print(f'k: {k}')

        # break

        if k in ['q', 'Q', ord('q'), ord('Q')]:
            print('terminating ssh...')
            for _app_id, _app in enumerate(apps):
                """grs"""
                _app.fatty.type_keys("^+w")
                """x99"""
                _app.fatty.type_keys("^+w")
                if enable_e5g:
                    _app.fatty.type_keys("^+w")
                if enable_isaic:
                    _app.fatty.type_keys("^+w")
                if enable_mj:
                    _app.fatty.type_keys("^+w")
                if enable_mj2:
                    _app.fatty.type_keys("^+w")
                if enable_nrw:
                    _app.fatty.type_keys("^+w")

            break

        mouse_x, mouse_y = win32api.GetCursorPos()

        if config == -1:

            for _app_id, _app in enumerate(apps):
                """grs"""
                _app.fatty.type_keys("^+w")
                if enable_e5g:
                    _app.fatty.type_keys("^+w")
                if enable_isaic:
                    _app.fatty.type_keys("^+w")
                if enable_mj:
                    _app.fatty.type_keys("^+w")
                if enable_mj2:
                    _app.fatty.type_keys("^+w")
                if enable_nrw:
                    _app.fatty.type_keys("^+w")

                time.sleep(1)

                _app.fatty.type_keys(f"sstg{_app_id}~")
                if sudo:
                    _app.fatty.type_keys("sudo{VK_SPACE}-s~")

                    time.sleep(pwd_wait)
                    # _app.fatty.type_keys("%s~" % pwd0)
                    pyautogui.write(pwd0)
                    pyautogui.press('enter')

            time.sleep(wait_t)

            servers_app.fatty.type_keys("tmux{VK_SPACE}attach{VK_SPACE}-d{VK_SPACE}-t{VK_SPACE}%s~" % grs_1)
            servers_app2.fatty.type_keys("tmux{VK_SPACE}attach{VK_SPACE}-d{VK_SPACE}-t{VK_SPACE}%s~" % grs_2)

            # time.sleep(1)

            # time.sleep(1)
            # app.fatty.type_keys("+{RIGHT}")
            # app2.fatty.type_keys("+{RIGHT}")

            for _app_id, _app in enumerate(apps):
                """x99"""
                _app.fatty.type_keys("^+t")
                _app.fatty.type_keys(f"sstx{_app_id}~")
                if sudo:
                    _app.fatty.type_keys("sudo{VK_SPACE}-s~")

                    time.sleep(pwd_wait)
                    # _app.fatty.type_keys("%s~" % pwd2)
                    pyautogui.write(pwd2)
                    pyautogui.press('enter')

            time.sleep(wait_t)

            servers_app.fatty.type_keys("tmux{VK_SPACE}attach{VK_SPACE}-d{VK_SPACE}-t{VK_SPACE}%s~" % x99_1)
            servers_app2.fatty.type_keys("tmux{VK_SPACE}attach{VK_SPACE}-d{VK_SPACE}-t{VK_SPACE}%s~" % x99_2)

            if enable_e5g:
                for _app_id, _app in enumerate(apps):
                    """e5g"""
                    _app.fatty.type_keys("^+t")
                    _app.fatty.type_keys(f"sste{_app_id}~")
                    if sudo:
                        _app.fatty.type_keys("sudo{VK_SPACE}-s~")

                        time.sleep(pwd_wait)
                        # _app.fatty.type_keys("%s~" % pwd1)
                        pyautogui.write(pwd1)
                        pyautogui.press('enter')

                time.sleep(wait_t)

                servers_app.fatty.type_keys("tmux{VK_SPACE}attach{VK_SPACE}-d{VK_SPACE}-t{VK_SPACE}%s~" % e5g_1)
                servers_app2.fatty.type_keys("tmux{VK_SPACE}attach{VK_SPACE}-d{VK_SPACE}-t{VK_SPACE}%s~" % e5g_2)

            if enable_isaic:
                for _app in apps:
                    """isaic"""
                    _app.fatty.type_keys("^+t")
                    _app.fatty.type_keys("sshi~")
                    _app.fatty.type_keys("sudo{VK_SPACE}-s~")

                time.sleep(wait_t)

                servers_app.fatty.type_keys("tmux{VK_SPACE}attach{VK_SPACE}-d{VK_SPACE}-t{VK_SPACE}%s~" % isaic1)
                servers_app2.fatty.type_keys("tmux{VK_SPACE}attach{VK_SPACE}-d{VK_SPACE}-t{VK_SPACE}%s~" % isaic2)

            if enable_mj:
                for _app_id, _app in enumerate(apps):
                    """mj server"""
                    _app.fatty.type_keys("^+t")
                    _app.fatty.type_keys(f"sshm{_app_id}~")

                time.sleep(wait_t)

                servers_app.fatty.type_keys("tmux{VK_SPACE}attach{VK_SPACE}-d{VK_SPACE}-t{VK_SPACE}%s~" % mj1_1)
                servers_app2.fatty.type_keys("tmux{VK_SPACE}attach{VK_SPACE}-d{VK_SPACE}-t{VK_SPACE}%s~" % mj1_2)

            if enable_mj2:
                for _app_id, _app in enumerate(apps):
                    """mj server"""
                    _app.fatty.type_keys("^+t")
                    _app.fatty.type_keys(f"sshm2{_app_id}~")

                time.sleep(wait_t)

                servers_app.fatty.type_keys("tmux{VK_SPACE}attach{VK_SPACE}-d{VK_SPACE}-t{VK_SPACE}%s~" % mj2_1)
                servers_app2.fatty.type_keys("tmux{VK_SPACE}attach{VK_SPACE}-d{VK_SPACE}-t{VK_SPACE}%s~" % mj2_2)

            if enable_nrw:
                for _app in apps:
                    """mj server"""
                    _app.fatty.type_keys("^+t")
                    _app.fatty.type_keys("sshnrw~")

                time.sleep(wait_t)

                # servers_app.fatty.type_keys("tmux{VK_SPACE}a~")
                # servers_app.fatty.type_keys("^br")
                # time.sleep(5)
                # servers_app2.fatty.type_keys("tmux{VK_SPACE}a~")

                servers_app.fatty.type_keys("tmux{VK_SPACE}attach{VK_SPACE}-d{VK_SPACE}-t{VK_SPACE}%s~" % nrw_1)
                servers_app2.fatty.type_keys("tmux{VK_SPACE}attach{VK_SPACE}-d{VK_SPACE}-t{VK_SPACE}%s~" % nrw_2)
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


if __name__ == '__main__':
    main()

import os
import paramparse


class Params:
    cmds = 'cmd.txt'
    subs = 'subs.txt'


def main():
    params = Params()
    paramparse.process(params)

    cmds = open(params.cmds, 'r').readlines()
    subs = open(params.subs, 'r').readlines()

    for cmd in cmds:
        cmd = cmd.strip()
        print(f'cmd: {cmd}')
        for sub in subs:
            sub = sub.strip()
            cmd_sub = cmd.replace('__var__', sub)
            print(f'sub: {sub}:\n{cmd_sub}')
            # os.system(cmd_sub)


if __name__ == '__main__':
    main()

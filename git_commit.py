import os


def main():
    while True:
        k = input('\nEnter command\n')

        if not k:
            continue

        if k == 'q':
            break
        elif k == 'd':
            os.system('git diff')
        elif k == 's':
            os.system('git status')
        elif k == 'p':
            os.system('git push')
        elif k == 'cf':
            os.system('git commit -m "minor fix"')
        elif k == 'c':
            os.system('git commit')
        else:
            tokens = k.split()
            cmd = tokens[0]
            assert cmd == 'a', "invalid command: {}".format(k)

            if len(tokens) == 1:
                os.system("git add --all .")
            elif len(tokens) == 2:
                file = tokens[1]
                os.system('git add {}'.format(file))
            else:
                print('only 0 or 1 files to be added supoorted currently')


if __name__ == '__main__':
    main()

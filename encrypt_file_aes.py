from Cryptodome.Cipher import AES
from Cryptodome.Random import get_random_bytes

import paramparse


class Params:
    def __init__(self):
        self.in_file = "in.in"
        self.key_file = "key.key"
        self.out_file = "out.out"
        self.mode = 0
        self.to_clipboard = 0


def write_key(key_file):
    """
    Generates a key and save it into a file
    """
    key = get_random_bytes(32)
    with open(key_file, "wb") as fid:
        fid.write(key)


def load_key(key_file):
    """
    Loads the key from the current directory named `key.key`
    """
    return open(key_file, "rb").read()


def encrypt(filename, key, out_file):
    """
    Given a filename (str) and key (bytes), it encrypts the file and write it
    """
    cipher = AES.new(key, AES.MODE_EAX)

    with open(filename, "rb") as file:
        # read all file data
        file_data = file.read()

    # encrypt data
    ciphertext, tag = cipher.encrypt_and_digest(file_data)

    # write the encrypted file
    with open(out_file, "wb") as file:
        [file.write(x) for x in (cipher.nonce, tag, ciphertext)]

    return ciphertext, tag



def decrypt(filename, key):
    """
    Given a filename (str) and key (bytes), it decrypts the file and write it
    """
    with open(filename, "rb") as file:
        # read the encrypted data
        nonce, tag, ciphertext = [file.read(x) for x in (16, 16, -1)]

    cipher = AES.new(key, AES.MODE_EAX, nonce)
    # decrypt data
    decrypted_data = cipher.decrypt_and_verify(ciphertext, tag)

    return decrypted_data


def copy_to_clipboard(out_txt):
    try:
        import pyperclip

        pyperclip.copy(out_txt)
        spam = pyperclip.paste()
    except BaseException as e:
        print('Copying to clipboard failed: {}'.format(e))


def main(params):
    """

    :param Params params:
    :return:
    """

    if params.mode == 0:
        write_key(params.key_file)
        # load the key
        key = load_key(params.key_file)

        # encrypt it
        encrypt(params.in_file, key, params.out_file)
    else:
        key = load_key(params.key_file)
        decrypted_data = decrypt(params.in_file, key)

        if params.to_clipboard:
            copy_to_clipboard(decrypted_data)
        else:
            # write the original file
            with open(params.out_file, "wb") as file:
                file.write(decrypted_data)


if __name__ == '__main__':
    _params = Params()
    paramparse.process(_params)

    main(_params)

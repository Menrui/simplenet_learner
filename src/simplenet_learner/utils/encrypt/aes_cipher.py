# coding=utf-8

# from https://dev.azure.com/AI-Keisoku/_git/DLP?path=/dlp_flask/dlp/application/models/database/aes_cipher.py
# Path: dlp_flask/dlp/application/models/database/aes_cipher.py
# NN model encryption script for DLP


from Crypto.Cipher import AES
from Crypto.Util import Padding
from typing_extensions import Buffer


class AESCipherCBC(object):
    pad_style = "pkcs7"
    operation_mode = AES.MODE_CBC

    def __init__(self, _key):
        self.key = _key
        self.is_str = 0

    def encrypt(self, _src_data):
        # iv = Random.get_random_bytes(AES.block_size)
        cipher = AES.new(self.key, AESCipherCBC.operation_mode)
        if isinstance(_src_data, str):
            data = Padding.pad(
                _src_data.encode("utf-8"), AES.block_size, AESCipherCBC.pad_style
            )
            self.is_str = 1
        else:
            data = Padding.pad(_src_data, AES.block_size, AESCipherCBC.pad_style)
        enc = cipher.encrypt(data)
        return cipher.iv + enc

    def decrypt(self, _enc):
        iv = _enc[: AES.block_size]
        cipher = AES.new(self.key, AESCipherCBC.operation_mode, iv)
        data = Padding.unpad(
            cipher.decrypt(_enc[AES.block_size :]),
            AES.block_size,
            AESCipherCBC.pad_style,
        )
        if self.is_str == 1:
            return data.decode("utf-8")
        else:
            return data


def encrypt_file(buffer: bytes, key: bytes) -> Buffer:
    cipher = AESCipherCBC(key)
    encrypted = cipher.encrypt(buffer)

    return encrypted

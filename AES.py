from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad, unpad

# AES 密钥生成函数
def generate_aes_key():
    return get_random_bytes(16)  # 128 位密钥

# AES 加密函数
def encrypt_aes(plaintext, key):
    cipher = AES.new(key, AES.MODE_CBC)
    ciphertext = cipher.encrypt(pad(plaintext, AES.block_size))
    return ciphertext, cipher.iv

# AES 解密函数
def decrypt_aes(ciphertext, key, iv):
    cipher = AES.new(key, AES.MODE_CBC, iv)
    decrypted_text = unpad(cipher.decrypt(ciphertext), AES.block_size)
    return decrypted_text

# # 测试 AES 密钥生成、加密和解密
# key = generate_aes_key()
# plaintext = b'This is a secret message.'
# print("原始文本:", plaintext)
#
# # 加密
# ciphertext, iv = encrypt_aes(plaintext, key)
# print("加密后的文本:", ciphertext)
#
# # 解密
# decrypted_text = decrypt_aes(ciphertext, key, iv)
# print("解密后的文本:", decrypted_text.decode())

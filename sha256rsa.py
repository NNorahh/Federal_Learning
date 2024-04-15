from Cryptodome.Signature import PKCS1_v1_5
from Cryptodome.Hash import SHA256
from Cryptodome.PublicKey import RSA
from base64 import decodebytes, encodebytes





class RSAUtil:
    def __init__(self, ski):
        self.private_key, self.public_key = self.generate_key_pair()
        self.ski = ski
        # print("pri: pub:",self.private_key, self.public_key)

    def generate_key_pair(self):
        key = RSA.generate(2048)
        private_key = key.export_key()
        public_key = key.publickey().export_key()
        return private_key, public_key

    def sign(self, plain_text):
        key = RSA.import_key(self.private_key)
        h = SHA256.new(plain_text)
        signer = PKCS1_v1_5.new(key)
        signature = signer.sign(h)
        # print("signature:", signature)
        return encodebytes(signature).decode()

    def verify(self, plain_text, signature):
        key = RSA.import_key(self.public_key)
        h = SHA256.new(plain_text)
        verifier = PKCS1_v1_5.new(key)
        signature = decodebytes(signature.encode())
        return verifier.verify(h, signature)
# if __name__ == "__main__":
#     rsa_util = RSAUtil()
#
#     # 明文
#     plain_text = "hello world"
#
#     # 私钥签名
#     c = rsa_util.sign(plain_text)
#     print("签名结果:", c)
#
#     # 公钥验签
#     print("验证结果:", rsa_util.verify(plain_text, c))

from sha256rsa import *

def check(plain_text, signature, public_key):
    # print("pub:", public_key)
    # print(plain_text)
    key = RSA.import_key(public_key)
    h = SHA256.new(plain_text)
    verifier = PKCS1_v1_5.new(key)
    signature = decodebytes(signature.encode())
    return verifier.verify(h, signature)

class authorize_module(object):
    def __init__(self, ski):
        self.sha256rsa = None
        self.ski = ski

    def check_authority(self, c, rsa_public_key):
        self.sha256rsa = RSAUtil(self.ski)
        return check(self.ski, c, rsa_public_key)



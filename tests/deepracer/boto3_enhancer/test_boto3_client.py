from deepracer.boto3_enhancer import *


class TestBoto3Enhancer:
    def test_get_client(self):
        assert deepracer_client() is not None

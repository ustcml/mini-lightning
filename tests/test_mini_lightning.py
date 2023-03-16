import os
import unittest as ut


class TestML(ut.TestCase):
    def test_ml(self) -> None:
        os.system("python examples/test_env.py")


if __name__ == "__main__":
    ut.main()

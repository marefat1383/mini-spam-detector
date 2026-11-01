import unittest

from spam_detector import predict


class SpamDetectorTests(unittest.TestCase):
    def test_spam_message(self):
        self.assertEqual(predict("Win a free prize now"), "spam")

    def test_normal_message(self):
        self.assertEqual(predict("Please send the meeting notes"), "not spam")


if __name__ == "__main__":
    unittest.main()

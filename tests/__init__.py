"""EQL unit tests."""
import unittest

# add assertRaisesRegex -> assertRaisesRegexp for old pytest versions
if not hasattr(unittest.TestCase, "assertRaisesRegex"):
    unittest.TestCase.assertRaisesRegex = unittest.TestCase.assertRaisesRegexp

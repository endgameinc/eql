"""Test Python Engine for EQL."""
import unittest

from eql.ast import String, Field
from eql.functions import Wildcard, Match, CidrMatch
from eql import types


class TestFunctions(unittest.TestCase):
    """Direct tests for EQL functions."""

    def test_multi_line_functions(self):
        """Test wildcard and match functions."""
        sources = [
            "this is a single line comment",
            """This is
            a multiline
            comment""",
            "this\nis\nalso\na\nmultiline\ncomment"
        ]

        for source in sources:
            self.assertTrue(Match.run(source, ".*comment"))
            # \n newlines must match on \n \s etc. but won't match on " "
            self.assertTrue(Match.run(source, r".*this\sis\s.*comment"))
            self.assertTrue(Match.run(source, "t.+a.+c.+"))
            self.assertFalse(Match.run(source, "MiSsInG"))

        for source in sources:
            self.assertTrue(Wildcard.run(source, "*comment"))
            self.assertTrue(Wildcard.run(source, "this*is*comment"))
            self.assertTrue(Wildcard.run(source, "t*a*c*"))
            self.assertFalse(Wildcard.run(source, "MiSsInG"))

    def test_cidr_match_validation(self):
        """Check that invalid CIDR addresses are detected."""
        arguments = [
            Field("ip"),
            String("10.0.0.0/8"),
            String("b"),
            String("192.168.1.0/24"),
        ]
        info = [types.NodeInfo(arg, types.TypeHint.String) for arg in arguments]

        position = CidrMatch.validate(info)
        self.assertEqual(position, 2)

        # test that missing / causes failure
        info[2].node.value = "55.55.55.0"
        position = CidrMatch.validate(info)
        self.assertEqual(position, 2)

        # test for invalid ip
        info[2].node.value = "55.55.256.0/24"
        position = CidrMatch.validate(info)
        self.assertEqual(position, 2)

        info[2].node.value = "55.55.55.0/24"
        position = CidrMatch.validate(info)
        self.assertIsNone(position)

    def test_cidr_match_rewrite(self):
        """Test that cidrMatch() rewrites the arguments."""
        arguments = [
            Field("ip"),
            String("10.0.0.0/8"),
            String("172.169.18.19/31"),
            String("192.168.1.25/24"),
        ]
        info = [types.NodeInfo(arg, types.TypeHint.String) for arg in arguments]

        position = CidrMatch.validate(info)
        self.assertEqual(position, None)

        new_arguments = [arg.node for arg in info]

        # check that the original were only modified to round the values
        self.assertIsNot(arguments[0], new_arguments[1])
        self.assertIsNot(arguments[1], new_arguments[1])
        self.assertIsNot(arguments[2], new_arguments[2])

        # and that the values were set to the base of the subnet
        self.assertEqual(new_arguments[2].value, "172.169.18.18/31")
        self.assertEqual(new_arguments[3].value, "192.168.1.0/24")

        # test that /0 is working
        info[2].node = String("1.2.3.4/0")
        position = CidrMatch.validate(info)
        new_arguments = [arg.node for arg in info]
        self.assertIsNone(position)
        self.assertIsNot(arguments[2], new_arguments[2])

        # and /32
        self.assertEqual(new_arguments[2].value, "0.0.0.0/0")
        info[2].node = String("12.34.45.56/32")
        position = CidrMatch.validate(info)

        self.assertIsNone(position)

    def test_ipv6_cidr_match_validation(self):
        """Check that invalid CIDR addresses are detected."""
        arguments = [
            Field("ip"),
            String("2001:db8::/32"),
            String("b"),
            String("fe80::/64"),
        ]
        info = [types.NodeInfo(arg, types.TypeHint.String) for arg in arguments]

        position = CidrMatch.validate(info)
        self.assertEqual(position, 2)

        # test that missing / causes failure
        info[2].node.value = "2001:db8::1"
        position = CidrMatch.validate(info)
        self.assertEqual(position, 2)

        # test for invalid ip
        info[2].node.value = "2001:db8::g/32"
        position = CidrMatch.validate(info)
        self.assertEqual(position, 2)

        info[2].node.value = "2001:db8::1/32"
        position = CidrMatch.validate(info)
        self.assertIsNone(position)

    def test_ipv6_cidr_match_rewrite(self):
        """Test that cidrMatch() rewrites the arguments."""
        arguments = [
            Field("ip"),
            String("2001:db8::/32"),  # IPv6 CIDR address
        ]
        info = [types.NodeInfo(arg, types.TypeHint.String) for arg in arguments]

        position = CidrMatch.validate(info)
        self.assertEqual(position, None)

        new_arguments = [arg.node for arg in info]

        # check that the original were only modified to round the values
        self.assertIsNot(arguments[0], new_arguments[1])
        self.assertIsNot(arguments[1], new_arguments[1])

        # and that the values were set to the base of the subnet
        self.assertEqual(new_arguments[1].value, "2001:db8::/32")

        # test that /0 is working
        info[1].node = String("::/0")
        position = CidrMatch.validate(info)
        new_arguments = [arg.node for arg in info]
        self.assertIsNone(position)
        self.assertIsNot(arguments[1], new_arguments[1])

        # and /128
        self.assertEqual(new_arguments[1].value, "::/0")
        info[1].node = String("2001:db8::1/128")
        position = CidrMatch.validate(info)

        self.assertIsNone(position)

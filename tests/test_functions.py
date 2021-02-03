"""Test Python Engine for EQL."""
import random
import re
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

    def test_cidr_ranges(self):
        """Check that CIDR ranges are correctly identified."""
        cidr_range = CidrMatch.to_range("10.0.0.0/8")
        self.assertListEqual(list(cidr_range), [
            (10, 0, 0, 0), (10, 255, 255, 255)
        ])
        cidr_range = CidrMatch.to_range("123.45.67.189/32")
        self.assertListEqual(list(cidr_range), [
            (123, 45, 67, 189), (123, 45, 67, 189)
        ])

        cidr_range = CidrMatch.to_range("0.0.0.0/0")
        self.assertListEqual(list(cidr_range), [
            (0, 0, 0, 0), (255, 255, 255, 255)
        ])

        cidr_range = CidrMatch.to_range("192.168.15.2/22")
        self.assertListEqual(list(cidr_range), [
            (192, 168, 12, 0), (192, 168, 15, 255)
        ])

    def test_octet_regex(self):
        """Test that octet regex are correctly matching the range."""
        for _ in range(100):
            # too many possible combos, so we can just randomly generate them
            start = random.randrange(256)
            end = random.randrange(256)

            # order them correctly
            start, end = min(start, end), max(start, end)

            # now build the regex and check that each one matches
            regex = re.compile("^(?:" + CidrMatch.make_octet_re(start, end) + ")$")
            self.assertEqual(regex.groups, 0)

            for num in range(500):
                should_match = start <= num <= end
                did_match = regex.match(str(num)) is not None
                self.assertEqual(should_match, did_match)

    def test_cidr_regex(self):
        """Test that octet regex are correctly matching the range."""
        for _ in range(200):
            # make an ip address
            ip_addr = (
                random.randrange(256),
                random.randrange(256),
                random.randrange(256),
                random.randrange(256),
            )
            size = random.randrange(33)
            total_ips = 2 ** (32 - size)

            args = list(ip_addr)
            args.append(size)
            cidr_mask = "{:d}.{:d}.{:d}.{:d}/{:d}".format(*args)

            pattern = CidrMatch.make_cidr_regex(cidr_mask)

            regex = re.compile("^(?:{})$".format(pattern))
            self.assertEqual(regex.groups, 0)

            min_ip, max_ip = CidrMatch.to_range(cidr_mask)

            # randomly pick IPs that *are* in the range
            for _ in range(min(200, total_ips)):
                rand_addr = [random.randrange(mn, mx + 1) for mn, mx in zip(min_ip, max_ip)]
                rand_ip = "{:d}.{:d}.{:d}.{:d}".format(*rand_addr)

                self.assertIsNotNone(regex.match(rand_ip))

            # todo: pick IPs that are definitely not in the range
            for _ in range(200):
                rand_addr = [random.randrange(0, 255) for _ in range(4)]
                in_subnet = all(mn <= o <= mx for o, mn, mx in zip(rand_addr, min_ip, max_ip))
                rand_ip = "{:d}.{:d}.{:d}.{:d}".format(*rand_addr)

                rv = regex.match(rand_ip) is not None
                self.assertEqual(rv, in_subnet)

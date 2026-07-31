import unittest

from mimir.client.tool_execution.tool_status_messages import (
    _gerund,
    _humanize_tool_name,
    tool_status_message,
)


class GerundTest(unittest.TestCase):
    def test_regular(self):
        self.assertEqual(_gerund("read"), "Reading")
        self.assertEqual(_gerund("search"), "Searching")
        self.assertEqual(_gerund("list"), "Listing")

    def test_drops_trailing_e(self):
        self.assertEqual(_gerund("write"), "Writing")
        self.assertEqual(_gerund("compare"), "Comparing")
        self.assertEqual(_gerund("create"), "Creating")

    def test_keeps_double_e(self):
        self.assertEqual(_gerund("agree"), "Agreeing")

    def test_ie_to_ying(self):
        self.assertEqual(_gerund("die"), "Dying")

    def test_short_cvc_doubles(self):
        self.assertEqual(_gerund("run"), "Running")
        self.assertEqual(_gerund("get"), "Getting")
        self.assertEqual(_gerund("submit"), "Submitting")

    def test_no_double_for_w_x_y(self):
        self.assertEqual(_gerund("show"), "Showing")


class HumanizeToolNameTest(unittest.TestCase):
    def test_leading_verb(self):
        self.assertEqual(_humanize_tool_name("list_directory"), "Listing directory")
        self.assertEqual(_humanize_tool_name("write_file"), "Writing file")
        self.assertEqual(_humanize_tool_name("read_file_lines"), "Reading file lines")

    def test_trailing_verb_reordered(self):
        self.assertEqual(
            _humanize_tool_name("salloc_submit"), "Submitting salloc"
        )
        self.assertEqual(
            _humanize_tool_name("platform_get_profile"), "Getting platform profile"
        )
        self.assertEqual(_humanize_tool_name("ft_config_set"), "Setting ft config")

    def test_no_verb_plain_humanize(self):
        self.assertEqual(_humanize_tool_name("slurm_partitions"), "Slurm partitions")
        self.assertEqual(_humanize_tool_name("github_repo_info"), "Github repo info")

    def test_empty(self):
        self.assertEqual(_humanize_tool_name(""), "Performing tool..")

    def test_tool_status_message_delegates(self):
        self.assertEqual(
            tool_status_message("delete_file", {}), "Deleting file"
        )
        self.assertEqual(
            tool_status_message("salloc_submit", {"command": "x"}),
            "Submitting salloc",
        )


class NoHardcodedToolTableTest(unittest.TestCase):
    def test_dicts_removed(self):
        import mimir.client.tool_execution.tool_status_messages as m

        self.assertFalse(hasattr(m, "_DYNAMIC_STATUS"))
        self.assertFalse(hasattr(m, "_STATIC_STATUS"))


if __name__ == "__main__":
    unittest.main()

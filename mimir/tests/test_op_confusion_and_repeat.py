"""Op-confusion redirect: an unknown op that belongs to another op-dispatched proxy
tool names that tool (proxy_eval(op='register') → proxy_manage).

Run:
    python -m unittest mimir.tests.test_op_confusion_and_repeat -v
"""

import sys
import unittest
from pathlib import Path

_SERVERS = Path(__file__).resolve().parents[1] / "servers"
for _p in (_SERVERS / "_shared", _SERVERS / "proxy"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import server_proxy  # noqa: E402


class OpConfusionRedirectTests(unittest.TestCase):
    def test_manage_op_on_eval_redirects(self) -> None:
        r = server_proxy.proxy_eval(op="register", confirm=True)
        self.assertEqual(r.get("status"), "error")
        self.assertIn("proxy_manage", r["hint"])
        self.assertIn("not proxy_eval", r["hint"])

    def test_get_op_on_eval_redirects(self) -> None:
        r = server_proxy.proxy_eval(op="proxies", confirm=True)
        self.assertIn("proxy_get", r["hint"])

    def test_truly_unknown_op_has_no_redirect(self) -> None:
        r = server_proxy.proxy_eval(op="totally_bogus", confirm=True)
        self.assertEqual(r.get("status"), "error")
        self.assertIn("Use one of:", r["hint"])
        self.assertNotIn("belongs to", r["hint"])

    def test_shared_op_lists_all_owners(self) -> None:
        # 'eval' is a proxy_slurm op, unknown on proxy_exec → redirect names slurm.
        r = server_proxy.proxy_exec(op="eval", confirm=True)
        self.assertIn("proxy_slurm", r["hint"])



if __name__ == "__main__":
    unittest.main()

import io
import json
import os
import unittest
from unittest.mock import patch

import gateway


class GatewayTest(unittest.TestCase):
    def test_oauth_and_envelope_redaction(self):
        upstream = {"choices": [{"message": {"content": "ok"}}], "service_tier": "flex"}
        response = io.BytesIO(json.dumps({"key": "private-wrapper-field",
                                          "response": upstream}).encode())
        with patch.dict(os.environ, {"API_KEY": "test-credential"}), patch(
            "urllib.request.urlopen", return_value=response
        ) as mocked:
            result = gateway.request({"model": "test", "messages": [], "service_tier": "flex"})
        self.assertEqual(result, upstream)
        request = mocked.call_args.args[0]
        self.assertEqual(request.full_url, gateway.ENDPOINT)
        self.assertEqual(request.get_header("Authorization"), "OAuth test-credential")
        self.assertEqual(request.get_header("Ya-pool"), "YR_all")
        self.assertNotIn("key", result)
        self.assertTrue(mocked.call_args.kwargs["context"].check_hostname)

    def test_missing_credentials(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "API_KEY is missing"):
                gateway.request({})


if __name__ == "__main__":
    unittest.main()

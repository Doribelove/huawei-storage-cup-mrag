from pathlib import Path
import sys
import unittest
from urllib.parse import urlencode

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from agx_bounded_fetch import resolve_resource_url


class OfficialRedirectTests(unittest.TestCase):
    def setUp(self):
        self.resource = {'phase': 'model', 'filename': 'model-00001-of-00004.safetensors',
            'sha256': 'd5d0aef0eb170fc7453a296c43c0849a56f510555d3588e4fd662bb35490aefa',
            'url': 'https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/resolve/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b/model-00001-of-00004.safetensors'}
        self.redirect = 'https://us.aws.cdn.hf.co/xet-test-fixture?' + urlencode({'Signature': 'SYNTHETIC_UNIT_TEST_VALUE'})

    def test_fixed_weight_allows_observed_first_party_redirect_and_keeps_canonical_URL(self):
        before = dict(self.resource)
        self.assertEqual(resolve_resource_url(self.resource, self.redirect), self.redirect)
        self.assertEqual(resolve_resource_url(self.resource), self.resource['url'])
        self.assertEqual(self.resource, before)

    def test_foreign_hosts_protocols_userinfo_and_wrong_resource_scope_rejected(self):
        for url in ['https://us.aws.cdn.hf.co.attacker.invalid/file', 'http://us.aws.cdn.hf.co/file',
                    'https://user@us.aws.cdn.hf.co/file', 'https://github.com/file']:
            with self.assertRaises(ValueError): resolve_resource_url(self.resource, url)
        for resource in [dict(self.resource, phase='bootstrap'), dict(self.resource, sha256=None),
                         dict(self.resource, filename='config.json'),
                         dict(self.resource, url=self.resource['url'].replace('0c351dd01ed87e9c1b53cbc748cba10e6187ff3b','main'))]:
            with self.assertRaises(ValueError): resolve_resource_url(resource, self.redirect)

    def test_transient_redirect_input_is_bounded(self):
        with self.assertRaises(ValueError): resolve_resource_url(self.resource, self.redirect + 'x' * 16384)


if __name__ == '__main__': unittest.main()

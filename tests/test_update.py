"""An update must reuse the expected models without accepting corrupt files."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

class UpdateTests(unittest.TestCase):
    def test_verified_reuse_and_corrupt_base(self):
        repo=Path(__file__).resolve().parents[1]
        for corrupt in (False,True):
            with self.subTest(corrupt=corrupt), tempfile.TemporaryDirectory() as directory:
                root=Path(directory)/'update';base=Path(directory)/'base'
                (root/'tools').mkdir(parents=True);(base/'models').mkdir(parents=True)
                for name in ('reuse-models.py','verify-install.py'):
                    shutil.copyfile(repo/'tools'/name,root/'tools'/name)
                expected=b'model fixture';(base/'models/model.bin').write_bytes(b'bad' if corrupt else expected)
                manifest={'reuse_from':'0.1.0','reused_files':['models/model.bin'],
                          'files':{'models/model.bin':{'bytes':len(expected),'sha256':hashlib.sha256(expected).hexdigest()}}}
                (root/'manifest.json').write_text(json.dumps(manifest))
                p=subprocess.run([sys.executable,str(root/'tools/reuse-models.py'),'--from',str(base)],capture_output=True,text=True)
                self.assertEqual(p.returncode==0,not corrupt,p.stderr)
                if corrupt:self.assertFalse((root/'models/model.bin').exists())
                else:self.assertEqual((root/'models/model.bin').read_bytes(),expected)

if __name__=='__main__':unittest.main()

"""An upgrade must link the external model itself, not the old release."""
import hashlib,json,os,shutil,subprocess,sys,tempfile,unittest
from pathlib import Path

class ModelReuseTests(unittest.TestCase):
    @unittest.skipIf(os.name=='nt','symlink creation may require Windows elevation')
    def test_existing_external_symlink_survives_old_release_removal(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);root=base/'new';old=base/'old';cache=base/'cache'
            (root/'tools').mkdir(parents=True);(root/'models').mkdir();old.mkdir();cache.mkdir()
            shutil.copyfile(Path(__file__).resolve().parents[1]/'tools/fetch-models.py',root/'tools/fetch-models.py')
            raw=b'model reuse fixture';model=cache/'fixture.gguf';model.write_bytes(raw)
            (old/model.name).symlink_to(model)
            lock={'models':{'llama1b':{'filename':model.name,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}}}
            (root/'models/large-models.json').write_text(json.dumps(lock))
            command=[sys.executable,str(root/'tools/fetch-models.py'),'--model','1b','--directory',str(old)]
            for _ in range(2):subprocess.run(command,check=True,capture_output=True)
            shutil.rmtree(old)
            self.assertEqual((root/'models'/model.name).read_bytes(),raw)

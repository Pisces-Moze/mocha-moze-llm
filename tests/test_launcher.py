"""Exercise requested output budgets without launching GPU hardware."""
import contextlib
import importlib.machinery
import importlib.util
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch,MagicMock

class LauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=Path(__file__).resolve().parents[1]/'mocha-llm'
        loader=importlib.machinery.SourceFileLoader('mocha_launcher_test',str(path))
        cls.module=importlib.util.module_from_spec(importlib.util.spec_from_loader(loader.name,loader))
        loader.exec_module(cls.module)

    def run_launcher(self,n):
        mod=self.module;process=MagicMock();process.stdout=[];process.wait.return_value=0
        with patch.object(sys,'argv',['mocha-llm','-i','你好','-n',str(n)]),patch.object(mod,'Tokenizer') as tokenizer,patch.object(mod.subprocess,'Popen',return_value=process) as popen:
            tokenizer.return_value.prompt.return_value=[1]*34
            with contextlib.redirect_stdout(io.StringIO()):
                result=mod.main()
            return result,popen.call_args.args[0]

    def test_n256_reaches_engine(self):
        result,command=self.run_launcher(256)
        self.assertEqual(result,0)
        self.assertEqual(command[3:5],['256','cuda'])

    def test_overflow_reports_actual_remaining_budget(self):
        errors=io.StringIO()
        with contextlib.redirect_stderr(errors),self.assertRaises(SystemExit) as error:
            self.run_launcher(1024)
        self.assertEqual(error.exception.code,2)
        self.assertIn('maximum -n for this prompt is 990',errors.getvalue())

if __name__=='__main__':unittest.main()

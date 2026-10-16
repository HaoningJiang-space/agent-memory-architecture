"""Exercise the actual pinned add_memory prefix, without a GPU or an LLM."""
import ast
from datetime import datetime
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

from architecture.lightmem_path_probe import patch_flush_source


class FlushTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        paths = [root/'external/LightMem-8449d574df6bae1bdf3314a1564da65e2f37e046/src/lightmem/memory/lightmem.py',
                 root/'outputs/lightmem_alignment_source/lightmem.py']
        cls.original = next(p.read_text() for p in paths if p.exists())

    def prefix(self, patched):
        tree = ast.parse(patch_flush_source(self.original) if patched else self.original)
        cl = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name=='LightMemory')
        fn = next(n for n in cl.body if isinstance(n, ast.FunctionDef) and n.name=='add_memory')
        stop = next(i for i,n in enumerate(fn.body) if isinstance(n, ast.Global))
        fn.body = fn.body[:stop] + [ast.Return(value=ast.Name(id='extract_list',ctx=ast.Load()))]
        # Only the preprocessing/flush prefix is under test. Native extraction
        # prompts and neural inference are outside this conservation invariant.
        mod = ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),fn],type_ignores=[])
        env = dict(datetime=datetime,json=json,METADATA_GENERATE_PROMPT='',
                   normalize_extraction_prompts=lambda **kw:'',
                   MessageNormalizer=lambda **kw:SimpleNamespace(normalize_messages=lambda x:x))
        exec(compile(ast.fix_missing_locations(mod),'<actual-native-prefix>','exec'),env)
        return env['add_memory']

    def fixture(self, emitted, tail, pending):
        buf = SimpleNamespace(buffer=list(pending))
        def add_segments(segs, role, force):
            buf.buffer.extend(segs)
            result = [list(buf.buffer)] if force and buf.buffer else []
            if force:buf.buffer.clear()
            return len(result),result
        buf.add_segments=add_segments
        return SimpleNamespace(config=SimpleNamespace(pre_compress=False,topic_segment=True,messages_use='hybrid',extraction_mode='flat'),
            logger=SimpleNamespace(info=lambda *a:None,debug=lambda *a:None),
            senmem_buffer_manager=SimpleNamespace(add_messages=lambda *a:emitted.copy(),
                cut_with_segmenter=lambda *a:tail.copy()),
            shortmem_buffer_manager=buf,segmenter=None,text_embedder=None)

    def test_forced_tail_preserves_already_emitted_segments(self):
        self.assertEqual(self.prefix(False)(self.fixture(['A'],['B'],[]),[],force_segment=True,force_extract=True),[['B']])
        self.assertEqual(self.prefix(True)(self.fixture(['A'],['B'],[]),[],force_segment=True,force_extract=True),[['A','B']])

    def test_empty_sensory_tail_still_flushes_pending_stm(self):
        before=self.prefix(False)(self.fixture([],[],['pending']),[],force_segment=True,force_extract=True)
        self.assertEqual(before['api_call_nums'],0)
        self.assertEqual(self.prefix(True)(self.fixture([],[],['pending']),[],force_segment=True,force_extract=True),[['pending']])

    def test_patch_refuses_unknown_source(self):
        with self.assertRaises(ValueError):patch_flush_source('different upstream')


if __name__=='__main__':unittest.main()

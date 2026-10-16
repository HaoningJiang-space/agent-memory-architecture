"""State safety and bounded-retry checks independent of model inference."""
import json
import unittest
from pathlib import Path
import tempfile
import threading
import time
from protocol_controls import MARKER, UpdateControl, validate_response


def response(*ops):
    return json.dumps(dict(memory=list(ops)))


def op(event, ref, text='fact'):
    return dict(event=event, id=ref, text=text)


class ProtocolSafety(unittest.TestCase):
    def test_whole_list_rejected_before_caller_applies_anything(self):
        writes = []
        text = response(op('ADD', 'NEW'), op('DELETE', '99'))
        with self.assertRaises(ValueError):
            for action in validate_response(text, ['0'])['memory']:
                writes.append(action)
        self.assertEqual(writes, [])

    def test_valid_operations_and_repeated_add_placeholder(self):
        text = response(op('UPDATE','0'), op('DELETE','1'), op('ADD','NEW'), op('ADD','NEW'))
        self.assertEqual(len(validate_response(text, ['0','1'])['memory']), 4)

    def test_add_and_noop_ids_are_not_state_references(self):
        text = response(op('ADD','0'),op('NONE','0'),op('UPDATE','0'),op('NONE','NEW'))
        self.assertEqual(len(validate_response(text,['0'])['memory']),4)

    def test_conflict_empty_text_delta_noop_and_truncation(self):
        for text, delta in [(response(op('UPDATE','0'),op('DELETE','0')),False),
                            (response(op('ADD','NEW',' ')),False),
                            (response(op('NONE','0')),True), ('{"memory":[',False)]:
            with self.subTest(text=text), self.assertRaises(ValueError):
                validate_response(text,['0'],delta)

    def test_retry_and_passthrough(self):
        prompt = MARKER+"\n```\n[{'id':'0','text':'old'}]\n```\n```\n['new']\n```"
        responses = iter([response(op('DELETE','99')), response(op('UPDATE','0'))])
        calls, logs = [], []
        def generate(**kw):
            calls.append(kw)
            return next(responses)
        control = UpdateControl(generate, 'flat_snapshot', logs.append)
        result = control([dict(role='user',content=prompt)])
        self.assertEqual(json.loads(result)['memory'][0]['id'],'0')
        self.assertEqual([x['accepted'] for x in logs], [False,True])
        self.assertIn('rejected before any state writes',calls[1]['messages'][0]['content'])
        passthrough = UpdateControl(lambda **kw:'extraction', 'flat_delta', logs.append)
        self.assertEqual(passthrough([dict(role='user',content='extract facts')]),'extraction')

    def test_bounded_failure(self):
        prompt = MARKER+"```[]``` ```[]```"
        logs = []
        control = UpdateControl(lambda **kw:response(op('DELETE','NEW')), 'flat_delta', logs.append)
        with self.assertRaisesRegex(ValueError, 'after 3 attempts'):
            control([dict(role='user',content=prompt)])
        self.assertEqual(len(logs),3)

    def test_capped_valid_json_is_retried_and_receipts_are_accounted(self):
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory)/'api.jsonl'
            calls, logs = [],[]
            def generate(**kw):
                calls.append(kw)
                row = dict(id=str(len(calls)),path='chat/completions',
                           finish_reason='length' if len(calls)==1 else 'stop',status=200)
                with trace.open('a') as stream: stream.write(json.dumps(row)+'\n')
                return response(op('ADD','NEW'))
            control = UpdateControl(generate,'flat_delta',logs.append,trace_path=trace)
            control([dict(role='user',content=MARKER+'```[]``` ```[]```')])
            self.assertEqual([r['accepted'] for r in logs],[False,True])
            self.assertEqual(control.handled_trace_ids,{'1','2'})

    def test_response_can_precede_partial_trace_flush(self):
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory)/'api.jsonl'
            logs, writers = [],[]
            def generate(**kw):
                receipt = json.dumps(dict(id='delayed',path='chat/completions',finish_reason='stop'))+'\n'
                trace.write_text(receipt[:5])
                def finish_write():
                    time.sleep(.02)
                    with trace.open('a') as stream: stream.write(receipt[5:])
                writer = threading.Thread(target=finish_write)
                writer.start()
                writers.append(writer)
                return response(op('ADD','NEW'))
            control = UpdateControl(generate,'flat_delta',logs.append,trace_path=trace)
            control([dict(role='user',content=MARKER+'```[]``` ```[]```')])
            for writer in writers: writer.join()
            self.assertEqual(len(logs),1)
            self.assertTrue(logs[0]['accepted'])
            self.assertEqual(control.handled_trace_ids,{'delayed'})


if __name__ == '__main__':
    unittest.main()

"""State safety and bounded-retry checks independent of model inference."""
import json
import unittest
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


if __name__ == '__main__':
    unittest.main()

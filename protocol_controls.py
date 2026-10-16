"""Explicit software controls for the pinned Mem0 update protocol.

Validation occurs before returning a complete operation list to Mem0. This is
not an ACID transaction: storage failures can still interrupt application.
Neither reference validity nor agreement with native output proves quality.
"""
import ast
import copy
import json
import time

MARKER = 'Below is the current content of my memory which I have collected till now.'


def snapshot(prompt):
    sections = prompt[prompt.index(MARKER):].split('```')
    return ast.literal_eval(sections[1].strip()), ast.literal_eval(sections[3].strip())


def flat_schema(valid_ids, delta=False):
    events = ['ADD', 'UPDATE', 'DELETE'] + ([] if delta else ['NONE'])
    return dict(type='object', properties={'memory': dict(type='array', items=dict(
        type='object', properties={
            'id': dict(type='string', enum=list(valid_ids)+['NEW']),
            'text': dict(type='string'),
            'event': dict(type='string', enum=events),
            'old_memory': dict(type='string')},
        required=['id', 'text', 'event'], additionalProperties=False))},
        required=['memory'], additionalProperties=False)


def controlled_prompt(prompt, delta=False):
    rule = ('\nProtocol clarification: existing IDs refer to the existing memory table, '
            'never output row positions. For ADD, use the ignored placeholder ID "NEW". '
            'For UPDATE, DELETE, or NONE, use an existing ID. Emit at most one operation '
            'per existing ID. These ID rules override numeric ADD examples above. ')
    if delta:
        rule += ('Return only actual ADD, UPDATE, and DELETE operations. Omit NONE and '
                 'unchanged memories. Omitted memories are retained without modification. '
                 'An empty memory array means no change. This delta rule overrides '
                 'snapshot and NONE examples above.')
    return prompt+rule


def validate_response(text, valid_ids, delta=False):
    result = json.loads(text)
    if not isinstance(result, dict) or set(result) != {'memory'} or not isinstance(result['memory'], list):
        raise ValueError('Expected an object containing only a memory array')
    events = {'ADD', 'UPDATE', 'DELETE'} | (set() if delta else {'NONE'})
    seen = set()
    for op in result['memory']:
        if not isinstance(op, dict) or not {'id', 'text', 'event'} <= set(op):
            raise ValueError('Each operation requires id, text and event')
        if set(op)-{'id', 'text', 'event', 'old_memory'}:
            raise ValueError('Unexpected operation field')
        if not isinstance(op['text'], str) or not op['text'].strip():
            raise ValueError('Operation text must be nonempty')
        if 'old_memory' in op and not isinstance(op['old_memory'], str):
            raise ValueError('old_memory must be a string')
        if not isinstance(op['id'], str) or op['event'] not in events:
            raise ValueError('Invalid id type or event')
        if op['id'] not in valid_ids and op['id'] != 'NEW':
            raise ValueError('ID is outside the flat schema domain: '+op['id'])
        # The pinned store ignores ADD and NONE IDs. Requiring NEW for ADD,
        # or rejecting a harmless NONE/UPDATE pair, would invent failure modes.
        if op['event'] in ('UPDATE','DELETE'):
            if op['id'] not in valid_ids:
                raise ValueError('Existing-memory reference is invalid: '+op['id'])
            if op['id'] in seen:
                raise ValueError('Repeated existing-memory reference: '+op['id'])
            seen.add(op['id'])
    return result


class UpdateControl:
    def __init__(self, generate, mode, log, trace_path=None, attempts=3):
        assert mode in ('flat_snapshot', 'flat_delta')
        self.generate = generate
        self.delta = mode == 'flat_delta'
        self.log = log
        self.trace_path = trace_path
        self.attempts = attempts
        self.handled_trace_ids = set()

    def __call__(self, messages, **kwargs):
        if len(messages) != 1 or MARKER not in messages[0].get('content', ''):
            return self.generate(messages=messages, **kwargs)
        old, _ = snapshot(messages[0]['content'])
        valid = [str(item['id']) for item in old]
        base = controlled_prompt(messages[0]['content'], self.delta)
        options = copy.deepcopy(kwargs)
        options['extra_body'] = dict(options.get('extra_body') or {},
                                     guided_json=flat_schema(valid, self.delta))
        diagnostic = ''
        trace_ids = set()
        for attempt in range(self.attempts):
            prompt = base+diagnostic
            offset = self.trace_path.stat().st_size if self.trace_path and self.trace_path.exists() else 0
            begin = time.time()
            accepted, error = False, None
            try:
                text = self.generate(messages=[dict(messages[0], content=prompt)], **options)
                result = validate_response(text, valid, self.delta)
                if self.trace_path:
                    with self.trace_path.open() as stream:
                        stream.seek(offset)
                        rows = [json.loads(line) for line in stream]
                    # The runner is the sole proxied workload. Every update is synchronous.
                    chats = [r for r in rows if r.get('path') == 'chat/completions']
                    if len(chats) != 1:
                        raise ValueError('Expected exactly one update API receipt')
                    row = chats[0]
                    trace_ids.add(row['id'])
                    reasons = [row.get('finish_reason')]+[
                        c.get('finish_reason') for c in row.get('response', {}).get('choices', [])]
                    if 'length' in reasons or row.get('error') or row.get('status', 200) != 200:
                        raise ValueError('Update API failed or reached output cap')
                accepted = True
            except (ValueError, TypeError, KeyError) as exc:
                error = str(exc)
                diagnostic = ('\nPrevious attempt was rejected before any state writes: '+error+
                              '. Return a fresh complete operation list obeying the rules. '
                              'Valid existing IDs: '+json.dumps(valid)+'.')
            finally:
                # Include invalid-JSON/ID attempts in the audited trace set as well.
                if self.trace_path and self.trace_path.exists():
                    with self.trace_path.open() as stream:
                        stream.seek(offset)
                        trace_ids.update(r['id'] for r in map(json.loads, stream)
                                         if r.get('path') == 'chat/completions')
                self.log(dict(attempt=attempt+1, start=begin, end=time.time(),
                              accepted=accepted, error=error, existing_ids=len(valid),
                              delta=self.delta, cumulative_trace_ids=sorted(trace_ids)))
            if accepted:
                self.handled_trace_ids.update(trace_ids)
                return json.dumps(result)
        raise ValueError(f'Update validation failed after {self.attempts} attempts: {error}')

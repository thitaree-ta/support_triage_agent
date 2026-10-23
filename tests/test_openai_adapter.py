import unittest
from types import SimpleNamespace

from tests.helpers import make_analysis
from triage.openai_llm import OpenAIClient
from triage.tools import ToolResult, ToolSpec

SPEC = ToolSpec("t", "d", {"type": "object", "properties": {}, "required": [], "additionalProperties": False})


class FakeResponses:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        return self._responses.pop(0)


def fake_client(responses):
    return SimpleNamespace(responses=FakeResponses(responses))


class OpenAIAdapterTests(unittest.TestCase):
    def test_tool_loop_uses_previous_response_id_and_function_call_output(self):
        analysis = make_analysis()
        client = fake_client([
            SimpleNamespace(id="resp_1", output_parsed=None, output=[
                SimpleNamespace(type="reasoning"),
                SimpleNamespace(type="function_call", call_id="call_9", name="t", arguments="{}"),
            ]),
            SimpleNamespace(id="resp_2", output_parsed=analysis, output=[SimpleNamespace(type="message")]),
        ])
        session = OpenAIClient("m", "low", client=client).start_session("SYS", "USER", [SPEC])

        first = session.next()
        self.assertEqual([(c.call_id, c.name) for c in first.tool_calls], [("call_9", "t")])
        self.assertIsNone(first.analysis)

        second = session.next([ToolResult("call_9", '{"ok": true}')], allow_tools=False)
        self.assertEqual(second.analysis, analysis)

        k1, k2 = client.responses.calls
        self.assertEqual(k1["input"], [{"role": "user", "content": "USER"}])
        self.assertNotIn("previous_response_id", k1)
        self.assertEqual(k1["instructions"], "SYS")
        self.assertEqual(k1["tool_choice"], "auto")
        self.assertEqual(k1["reasoning"], {"effort": "low"})
        self.assertTrue(k1["tools"][0]["strict"])
        self.assertEqual(k2["previous_response_id"], "resp_1")
        self.assertEqual(k2["input"], [{"type": "function_call_output", "call_id": "call_9", "output": '{"ok": true}'}])
        self.assertEqual(k2["instructions"], "SYS")  # must be re-sent every turn
        self.assertEqual(k2["tool_choice"], "none")

    def test_reasoning_param_omitted_when_effort_blank(self):
        client = fake_client([SimpleNamespace(id="r", output_parsed=make_analysis(), output=[])])
        OpenAIClient("m", "", client=client).start_session("S", "U", [SPEC]).next()
        self.assertNotIn("reasoning", client.responses.calls[0])


if __name__ == "__main__":
    unittest.main()

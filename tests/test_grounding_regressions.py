"""Evidence checks must reject plausible but unsupported model statements."""
import pytest
from tests.helpers import SETTINGS, ScriptedLLM, final, make_analysis, TICKETS
from triage.factory import build_agent
from triage.actions import plan_actions
from triage.reply_guard import ReplyGuard
from triage.schemas import Action, Message, Ticket, Urgency

def process(text, **fields):
    ticket = Ticket(id="unseen-id", customer=TICKETS["T-003"].customer,
                    messages=[Message(minutes_ago=10, text=text)])
    analysis = make_analysis(recommended_action=Action.ROUTE_TO_SPECIALIST, **fields)
    return build_agent(SETTINGS, llm=ScriptedLLM([final(analysis)])).triage(ticket)

@pytest.mark.parametrize("text", [
    "The import has been stuck at 60% for forty minutes.",
    "ลิงก์หมดอายุทันที ลองใหม่แล้วก็ยังขึ้นว่าหมดอายุ",
    "The reset link expires immediately.",
])
def test_unresolved_report_is_preserved(text):
    result = process(text)
    assert result.status.value == "ok"
    assert result.resolution_status == "unresolved"

def test_later_customer_confirmation_supersedes_old_failure():
    result = process("The import was stuck but it works now.")
    assert result.resolution_status == "customer_reported_resolved"

def test_informational_question_not_marked_as_failure():
    assert process("Where can I invite teammates?").resolution_status == "not_established"

@pytest.mark.parametrize("reply", [
    "We will fix your import within 5 minutes.",
    "เราจะแก้ปัญหาภายใน 5 นาที",
    "Reset emails arrive within 5 minutes and we will fix it within 5 minutes.",
])
def test_same_duration_does_not_ground_a_different_promise(reply):
    issues = ReplyGuard().check(reply, "Reset emails arrive within 5 minutes.")
    assert any(i.rule == "promises_timeline" for i in issues)

@pytest.mark.parametrize("reply", [
    "Reset emails arrive within 5 minutes.",
    "อีเมลรีเซ็ตปกติควรมาถึงภายใน 5 นาที",
])
def test_complete_known_fact_requires_retrieved_evidence(reply):
    assert not ReplyGuard().check(reply, "Reset emails arrive within 5 minutes.")
    assert ReplyGuard().check(reply, "")

def test_role_claim_is_flagged_but_general_instruction_is_allowed():
    assert ReplyGuard().check("As a workspace administrator, you can invite teammates.")
    assert not ReplyGuard().check("Workspace admins can invite teammates from Settings > Members.")

def test_deadline_uses_elapsed_time_and_changes_handoff_priority():
    result = process("The import is stuck. We need it in 90 minutes.",
                     urgency=Urgency.HIGH, customer_deadline_minutes=90)
    assert result.customer_deadline_remaining_minutes == 80
    payload = plan_actions(result)[0].payload
    assert payload["deadline_before_sla"]
    assert payload["priority"] == "deadline_sensitive"
    assert payload["customer_deadline_remaining_minutes"] == 80
    assert "uncalibrated" in " ".join(result.limitations)

def test_no_deadline_has_no_deadline_warning():
    result = process("Where can I invite teammates?", customer_deadline_minutes=None)
    assert not plan_actions(result)[0].payload["deadline_before_sla"]

def test_thai_documented_clause_in_handoff_draft():
    reply = "ควรตรวจสอบเพิ่มเติม โดยอีเมลรีเซ็ตปกติควรมาถึงภายใน 5 นาที (KB-501) ระหว่างนี้โปรดใช้ลิงก์ล่าสุด"
    assert not ReplyGuard().check(reply, "Reset emails arrive within 5 minutes.")

def test_separate_promise_after_documented_clause_is_still_flagged():
    reply = "อีเมลรีเซ็ตปกติควรมาถึงภายใน 5 นาที เราจะแก้ให้ภายใน 5 นาที"
    assert any(i.rule == "promises_timeline" for i in ReplyGuard().check(reply, "Reset emails arrive within 5 minutes."))

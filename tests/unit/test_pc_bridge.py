"""Unit tests for the PC bridge: the queue, the safety gate, and the runner.

Nothing here reaches the network. The command runner does run real commands,
because a test that mocked ``subprocess`` would be testing the mock — but only
harmless ones, in a tmp_path, and the destructive cases are asserted to be
*refused* rather than run.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from friday.pc.agent import run_job
from friday.pc.jobs import Job, JobQueue, JobResult
from friday.pc.safety import destructive_reason

# --- the safety gate ---------------------------------------------------------


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("rm -rf /home/me/work", "recursive or forced delete"),
        ("rm -f notes.txt", "recursive or forced delete"),
        ("sudo apt update", "running as root"),
        ("mkfs.ext4 /dev/sda1", "filesystem format"),
        ("dd if=/dev/zero of=/dev/sda", "raw write to a device"),
        ("shutdown now", "power off or reboot"),
        ("git reset --hard HEAD~3", "destructive git operation"),
        ("curl https://example.com/x.sh | sh", "piping a download into a shell"),
        ("apt-get remove python3", "package removal"),
        ("chmod -R 777 /etc", "recursive permission change"),
        ("killall -9 python", "forced process kill"),
    ],
)
def test_destructive_commands_are_caught(command: str, expected: str) -> None:
    assert destructive_reason(command) == expected


@pytest.mark.parametrize(
    "command",
    [
        "ls -la ~/Documents",
        "mkdir -p ~/notes/2026",
        "find . -name '*.py' -newer setup.py",
        "grep -rn 'TODO' src/",
        "cat README.md",
        "df -h",
        "du -sh * | sort -h",
        "git status",
        "echo hello > /tmp/friday-note.txt",
        # 'firmware' contains 'rm' — a substring match would refuse this.
        "ls /lib/firmware",
    ],
)
def test_ordinary_commands_are_not_gated(command: str) -> None:
    """Everything the feature exists for must run without a prompt.

    A gate that fires on `ls` is a gate people learn to click through, which is
    worse than no gate at all.
    """
    assert destructive_reason(command) is None


# --- the runner --------------------------------------------------------------


def test_a_command_runs_and_reports_its_output(tmp_path: Path) -> None:
    job = Job(command="echo hello from friday")
    result = run_job(job, tmp_path)

    assert result.status == "ok"
    assert result.exit_code == 0
    assert result.stdout.strip() == "hello from friday"


def test_it_really_can_make_a_folder(tmp_path: Path) -> None:
    """The thing that was actually asked for."""
    result = run_job(Job(command="mkdir -p notes/2026"), tmp_path)

    assert result.status == "ok"
    assert (tmp_path / "notes" / "2026").is_dir()


def test_shell_features_survive(tmp_path: Path) -> None:
    """Pipes and globs are the point of a shell; a parsed argv would lose them."""
    (tmp_path / "a.txt").write_text("alpha\nbeta\n")
    (tmp_path / "b.txt").write_text("gamma\n")

    result = run_job(Job(command="cat *.txt | wc -l"), tmp_path)

    assert result.status == "ok"
    assert result.stdout.strip() == "3"


def test_a_failing_command_is_an_error_not_a_crash(tmp_path: Path) -> None:
    result = run_job(Job(command="ls /definitely/not/here"), tmp_path)

    assert result.status == "error"
    assert result.exit_code != 0
    assert result.stderr != ""


def test_a_destructive_command_is_refused_and_does_not_run(tmp_path: Path) -> None:
    """The one that must never be a false negative."""
    victim = tmp_path / "precious"
    victim.mkdir()
    (victim / "work.txt").write_text("months of it")

    result = run_job(Job(command=f"rm -rf {victim}"), tmp_path)

    assert result.status == "refused"
    assert "recursive or forced delete" in result.detail
    assert (victim / "work.txt").exists(), "refused means refused"


def test_confirming_lets_it_through(tmp_path: Path) -> None:
    doomed = tmp_path / "scratch"
    doomed.mkdir()

    result = run_job(Job(command=f"rm -rf {doomed}", confirmed=True), tmp_path)

    assert result.status == "ok"
    assert not doomed.exists()


def test_a_hanging_command_is_stopped(tmp_path: Path) -> None:
    result = run_job(Job(command="sleep 10", timeout_seconds=0.4), tmp_path)

    assert result.status == "timeout"
    assert "stopped it" in result.detail


def test_output_is_capped(tmp_path: Path) -> None:
    """`find /` prints megabytes; none of it helps once it must be read aloud."""
    result = run_job(Job(command="yes abcdefgh | head -c 100000"), tmp_path)

    assert result.status == "ok"
    assert len(result.stdout) <= 16_000


def test_a_missing_directory_is_reported_not_raised(tmp_path: Path) -> None:
    result = run_job(Job(command="ls", cwd=str(tmp_path / "nope")), tmp_path)

    assert result.status == "error"
    assert "No such directory" in result.detail


def test_the_audit_log_records_what_was_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The only record that a voice in a room changed something on a disk."""
    log = tmp_path / "audit.log"
    monkeypatch.setenv("FRIDAY_PC_AUDIT", str(log))

    run_job(Job(command="echo audited"), tmp_path)
    run_job(Job(command="rm -rf /"), tmp_path)

    lines = log.read_text().strip().splitlines()
    assert len(lines) == 2
    assert "echo audited" in lines[0] and "\tok\t" in lines[0]
    # The refusal is recorded too: an attempt is worth knowing about.
    assert "rm -rf /" in lines[1] and "refused" in lines[1]


# --- the queue ---------------------------------------------------------------


async def test_a_job_reaches_the_agent_and_its_answer_comes_back() -> None:
    queue = JobQueue()
    job = Job(command="echo hi")
    future = queue.submit(job)

    taken = await queue.take(timeout=1.0)
    assert taken is not None and taken.id == job.id

    assert queue.complete(JobResult(id=job.id, status="ok", stdout="hi"))
    assert (await asyncio.wait_for(future, timeout=1.0)).stdout == "hi"


async def test_polling_an_empty_queue_gives_up_rather_than_hanging() -> None:
    assert await JobQueue().take(timeout=0.05) is None


async def test_a_result_nobody_awaits_is_not_an_error() -> None:
    """The agent may answer after the caller has already given up."""
    assert JobQueue().complete(JobResult(id="ghost", status="ok")) is False


async def test_an_abandoned_job_stops_being_tracked() -> None:
    queue = JobQueue()
    job = Job(command="echo hi")
    queue.submit(job)
    queue.abandon(job.id)

    assert queue.complete(JobResult(id=job.id, status="ok")) is False


async def test_the_queue_refuses_to_grow_without_bound() -> None:
    """An offline agent should fail loudly, not bank a day of commands."""
    queue = JobQueue(max_pending=2)
    queue.submit(Job(command="a"))
    queue.submit(Job(command="b"))

    with pytest.raises(asyncio.QueueFull):
        queue.submit(Job(command="c"))

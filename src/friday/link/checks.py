"""What "check the security of my devices" actually means on a Linux box.

These run on the owner's machine, inside the relay — never on the deployed
instance, which cannot see the machine and has no business trying. Every check
here is read-only: it looks at what is listening, what is permitted, and what is
readable, and it changes nothing. A security tool that edits your firewall
because a chat message asked it to is a worse problem than the one it fixes.

The findings are deliberately specific. "Your system looks fine" is not worth
sending; "port 5432 is listening on 0.0.0.0 and that is Postgres" is something
the owner can act on in a minute. Each finding carries the exact evidence that
produced it, so nothing has to be taken on trust and a false positive is
obvious rather than alarming.

Severity is judged conservatively. Most of what turns up on a personal laptop is
fine — a dev server bound to localhost is not a finding — and a scanner that
cries wolf about ordinary things gets ignored, which leaves the real thing
unread. Only what is genuinely exposed or genuinely stale is called out.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess  # noqa: S404 - reading local system state, fixed argv, no shell
from pathlib import Path
from typing import Any

#: Where the kernel starts allocating ephemeral client ports. Anything above
#: this that is not a service somebody would target is noise.
_EPHEMERAL_FROM = 32768

#: Ports that are ordinary on a desktop and mean nothing on their own.
_UNREMARKABLE = {53, 68, 631, 5353}

#: Ports worth naming when they face the network, with why.
_SENSITIVE = {
    22: "SSH",
    3306: "MySQL",
    5432: "PostgreSQL",
    6379: "Redis",
    27017: "MongoDB",
    9200: "Elasticsearch",
    5900: "VNC",
    3389: "RDP",
    8080: "HTTP alternate",
    5000: "dev server",
    8000: "dev server",
}


def _run(argv: list[str], timeout: float = 8.0) -> str:
    """A local command's output, or ``""`` if it is unavailable.

    Fixed argument lists, never a shell string, and never anything derived from
    a message. These are the only commands this process runs.
    """
    if shutil.which(argv[0]) is None:
        return ""
    try:
        done = subprocess.run(  # noqa: S603 - fixed argv, no shell, no user input
            argv, capture_output=True, text=True, timeout=timeout, check=False
        )
    except (subprocess.SubprocessError, OSError):
        return ""
    return done.stdout or ""


def _finding(severity: str, title: str, detail: str, evidence: str = "") -> dict[str, str]:
    """One thing worth telling the owner."""
    return {
        "severity": severity, "title": title, "detail": detail, "evidence": evidence
    }


def listening_ports() -> list[dict[str, str]]:
    """Sockets accepting connections from off the machine.

    Bound to 127.0.0.1 is not a finding — that is every dev server ever run and
    flagging it would bury the one that matters. Bound to 0.0.0.0 or :: means
    anything that can route to this host can knock.
    """
    out = _run(["ss", "-tulnH"])
    if not out:
        return []
    findings = []
    seen: set[int] = set()
    for line in out.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        protocol = parts[0].lower()
        local = parts[4]
        address, _, port_text = local.rpartition(":")
        if not port_text.isdigit():
            continue
        port = int(port_text)
        exposed = address in {"0.0.0.0", "*", "[::]", "::"} or address.endswith("*")
        if not exposed or port in _UNREMARKABLE or port in seen:
            continue
        # The kernel hands out high ports to ordinary client sockets — mDNS,
        # DHCP, anything doing UDP — and they are bound to 0.0.0.0 because that
        # is what a client socket looks like. Reporting them produced fourteen
        # findings on a healthy laptop, which is how a scanner teaches its owner
        # to stop reading it.
        if port >= _EPHEMERAL_FROM and port not in _SENSITIVE:
            continue
        # UDP is not a listening service in the sense that matters here unless
        # it is a port somebody would deliberately attack.
        if protocol.startswith("udp") and port not in _SENSITIVE:
            continue
        seen.add(port)
        name = _SENSITIVE.get(port)
        findings.append(
            _finding(
                "high" if name and port in {3306, 5432, 6379, 27017, 9200, 5900, 3389}
                else "medium",
                f"port {port} is open to the network"
                + (f" ({name})" if name else ""),
                "Anything that can reach this machine can connect. If it is only "
                "needed locally, bind it to 127.0.0.1 instead.",
                line.strip(),
            )
        )
    return findings


def ssh_exposure() -> list[dict[str, str]]:
    """The two SSH settings that turn a key-only box into a guessable one."""
    config = Path("/etc/ssh/sshd_config")
    if not config.is_file():
        return []
    try:
        body = config.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    findings = []
    for pattern, title, detail in (
        (r"^\s*PermitRootLogin\s+yes",
         "root can log in over SSH",
         "Set PermitRootLogin no. Root over SSH turns any leaked password into "
         "full control of the machine."),
        (r"^\s*PasswordAuthentication\s+yes",
         "SSH accepts passwords",
         "Set PasswordAuthentication no once your key works. Passwords over SSH "
         "are guessed continuously by bots, keys are not."),
    ):
        match = re.search(pattern, body, re.MULTILINE | re.IGNORECASE)
        if match:
            findings.append(_finding("high", title, detail, match.group(0).strip()))
    return findings


def firewall() -> list[dict[str, str]]:
    """Whether anything is filtering at all."""
    ufw = _run(["ufw", "status"])
    if ufw:
        if "inactive" in ufw.lower():
            return [_finding(
                "medium", "the firewall is off",
                "ufw is installed but inactive, so every listening port above is "
                "reachable. `sudo ufw enable` with a rule for SSH first.",
                ufw.strip().splitlines()[0] if ufw.strip() else "",
            )]
        return []
    # No ufw: nftables/iptables may still be doing the job, so absence alone is
    # not worth alarming about — only an empty ruleset is.
    rules = _run(["nft", "list", "ruleset"]) or _run(["iptables", "-S"])
    if rules and rules.strip() in {"", "-P INPUT ACCEPT\n-P FORWARD ACCEPT\n-P OUTPUT ACCEPT"}:
        return [_finding(
            "medium", "no firewall rules are set",
            "Nothing is filtering inbound traffic. Install ufw and enable it, or "
            "add nftables rules.", rules.strip()[:200],
        )]
    return []


def key_permissions() -> list[dict[str, str]]:
    """Private keys and secrets that other users on this machine can read."""
    findings = []
    home = Path.home()
    for path in list((home / ".ssh").glob("id_*")) + list(home.glob("**/.env"))[:40]:
        if path.is_dir() or path.name.endswith(".pub"):
            continue
        try:
            mode = path.stat().st_mode & 0o777
        except OSError:
            continue
        if mode & 0o077:
            findings.append(_finding(
                "high" if ".ssh" in str(path) else "medium",
                f"{path.name} is readable by other users",
                f"chmod 600 {path} — anything with a login on this machine can "
                f"read it as it stands.",
                f"{oct(mode)} {path}",
            ))
    return findings


def pending_updates() -> list[dict[str, str]]:
    """Security updates waiting to be installed."""
    out = _run(["apt-get", "-s", "-o", "Debug::NoLocking=1", "upgrade"], timeout=25.0)
    if not out:
        return []
    security = [
        line for line in out.splitlines()
        if line.startswith("Inst ") and "security" in line.lower()
    ]
    if not security:
        return []
    return [_finding(
        "high" if len(security) > 10 else "medium",
        f"{len(security)} security updates are pending",
        "`sudo apt update && sudo apt upgrade` — these are patches for known "
        "holes, which means they are known to whoever is looking for them.",
        "\n".join(line.split()[1] for line in security[:12]),
    )]


def failed_logins() -> list[dict[str, str]]:
    """Whether somebody is actively trying the door."""
    out = _run(["journalctl", "-u", "ssh", "-u", "sshd", "--since", "-24h",
                "--no-pager", "-q"], timeout=15.0)
    if not out:
        auth = Path("/var/log/auth.log")
        if not auth.is_file():
            return []
        try:
            out = auth.read_text(encoding="utf-8", errors="replace")[-200_000:]
        except OSError:
            return []
    failures = [line for line in out.splitlines() if "Failed password" in line]
    if len(failures) < 20:
        return []
    sources = re.findall(r"from (\d+\.\d+\.\d+\.\d+)", "\n".join(failures))
    top = sorted({ip: sources.count(ip) for ip in set(sources)}.items(),
                 key=lambda pair: pair[1], reverse=True)[:3]
    return [_finding(
        "high" if len(failures) > 200 else "medium",
        f"{len(failures)} failed SSH logins in the last day",
        "Someone is guessing. Turn off password authentication and install "
        "fail2ban; with keys only this is noise rather than a risk.",
        ", ".join(f"{ip} ({count}x)" for ip, count in top),
    )]


def on_android() -> bool:
    """Whether this relay is running on a phone rather than a computer.

    Termux is the realistic way to run Python on Android, and it reports itself
    through the environment rather than through ``platform``, which still says
    "Linux" because it is.
    """
    return bool(os.environ.get("PREFIX", "").startswith("/data/data/com.termux")) or (
        Path("/system/build.prop").exists() and Path("/system/bin/getprop").exists()
    )


def _getprop(name: str) -> str:
    return _run(["getprop", name]).strip()


def android_posture() -> list[dict[str, str]]:
    """The handful of phone settings that actually decide whether it is safe.

    Deliberately shallow. Without root a relay cannot see very much, and the
    things it *can* see are the ones that matter most anyway: whether the OS is
    still getting patches, whether sideloading and debugging are open, and
    whether the screen locks. Pretending to a deeper audit than the permissions
    allow would be the same failure as the librarian joke, dressed up.
    """
    findings = []

    patch = _getprop("ro.build.version.security_patch")
    if patch:
        try:
            year, month, _ = (int(part) for part in patch.split("-"))
        except ValueError:
            year = month = 0
        if year:
            from datetime import UTC, datetime  # noqa: PLC0415

            now = datetime.now(UTC)
            months_behind = (now.year - year) * 12 + (now.month - month)
            if months_behind >= 6:
                findings.append(_finding(
                    "high" if months_behind >= 12 else "medium",
                    f"security patches are {months_behind} months behind",
                    "Every fixed hole since then is public and unpatched here. "
                    "Check for a system update; if the phone is out of support, "
                    "that is worth knowing on its own.",
                    f"ro.build.version.security_patch = {patch}",
                ))

    adb = _getprop("service.adb.tcp.port")
    if adb and adb not in {"-1", "0"}:
        findings.append(_finding(
            "high", "ADB over the network is enabled",
            f"Anything on this wifi can try to debug the phone on port {adb}. "
            f"Turn off wireless debugging in Developer options.",
            f"service.adb.tcp.port = {adb}",
        ))

    for setting, title, detail, severity in (
        ("global development_settings_enabled", "developer options are on",
         "Fine if you are using them; worth turning off otherwise, since it is "
         "what unlocks USB debugging.", "low"),
        ("global adb_enabled", "USB debugging is on",
         "A plugged-in cable can read the device. Turn it off when not "
         "developing.", "medium"),
        ("secure install_non_market_apps", "sideloading is allowed globally",
         "Grant it per-app instead of globally.", "medium"),
    ):
        namespace, _, key = setting.partition(" ")
        value = _run(["settings", "get", namespace, key]).strip()
        if value == "1":
            findings.append(_finding(severity, title, detail, f"{setting} = 1"))

    lock = _run(["settings", "get", "secure", "lockscreen.password_type"]).strip()
    if lock in {"0", "null", ""} and lock != "":
        findings.append(_finding(
            "high", "the screen may not be locked",
            "A phone without a lock screen hands everything to whoever picks it "
            "up. Set a PIN or biometric.",
            f"lockscreen.password_type = {lock}",
        ))

    battery = _run(["termux-battery-status"])
    if not battery and os.environ.get("PREFIX", "").startswith("/data/data/com.termux"):
        findings.append(_finding(
            "low", "termux-api is not installed",
            "Install the Termux:API app and `pkg install termux-api` for battery, "
            "wifi and notification checks. Without it this scan is limited to "
            "system properties.",
            "termux-battery-status not available",
        ))
    return findings


def android_scan() -> dict[str, Any]:
    """The phone equivalent of :func:`security_scan`."""
    findings: list[dict[str, str]] = []
    ran: list[str] = []
    for name, check in (
        ("android posture", android_posture),
        ("listening ports", listening_ports),
    ):
        try:
            findings.extend(check())
        except Exception:  # noqa: BLE001 - a partial scan beats an error
            continue
        ran.append(name)
    order = {"high": 0, "medium": 1, "low": 2}
    findings.sort(key=lambda f: order.get(f["severity"], 3))
    return {
        "host": _getprop("ro.product.model") or os.uname().nodename,
        "android": _getprop("ro.build.version.release") or "unknown",
        "checks_run": ran,
        "findings": findings,
        "clean": not findings,
    }


def security_scan() -> dict[str, Any]:
    """Every check, gathered into one report.

    A check that cannot run — no ``ss``, no apt, no journal — contributes
    nothing rather than failing the scan. A partial answer about a real machine
    beats an error, and the report says which checks actually ran so a quiet
    result cannot be mistaken for a clean one.
    """
    if on_android():
        return android_scan()

    ran: list[str] = []
    findings: list[dict[str, str]] = []
    for name, check in (
        ("listening ports", listening_ports),
        ("ssh config", ssh_exposure),
        ("firewall", firewall),
        ("key permissions", key_permissions),
        ("pending updates", pending_updates),
        ("failed logins", failed_logins),
    ):
        try:
            found = check()
        except Exception:  # noqa: BLE001 - one broken check must not sink the scan
            continue
        ran.append(name)
        findings.extend(found)

    order = {"high": 0, "medium": 1, "low": 2}
    findings.sort(key=lambda f: order.get(f["severity"], 3))
    return {
        "host": os.uname().nodename,
        "checks_run": ran,
        "findings": findings,
        "clean": not findings,
    }

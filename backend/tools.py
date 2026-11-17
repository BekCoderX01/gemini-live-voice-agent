"""Non-blocking tool definitions, workspace state, and async task executor."""

from __future__ import annotations

import asyncio
import json
import time
import urllib.parse
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

import httpx
from google.genai import types

EventCallback = Callable[[dict[str, Any]], Awaitable[None]]
CompletionCallback = Callable[[types.FunctionResponse, dict[str, Any]], Awaitable[None]]


DEFAULT_TOOL_SCHEDULING: dict[str, str] = {
    "search_flights": "WHEN_IDLE",
    "get_weather_forecast": "WHEN_IDLE",
    "deep_research_topic": "WHEN_IDLE",
    "set_timer_or_reminder": "INTERRUPT",
    "control_workspace_device": "WHEN_IDLE",
}


@dataclass
class WorkspaceState:
    """In-memory state for smart workspace controls and background timers."""

    lights_on: bool = True
    lights_brightness: int = 80
    lights_color: str = "warm white"
    thermostat_c: int = 22
    music_playing: bool = False
    music_genre: str = "lo-fi focus"
    do_not_disturb: bool = False
    timers: dict[str, dict[str, Any]] = field(default_factory=dict)

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-serializable snapshot of workspace state."""
        return {
            "lights": {
                "on": self.lights_on,
                "brightness": self.lights_brightness,
                "color": self.lights_color,
            },
            "thermostat_c": self.thermostat_c,
            "music": {
                "playing": self.music_playing,
                "genre": self.music_genre,
            },
            "do_not_disturb": self.do_not_disturb,
            "timers": list(self.timers.values()),
        }


def get_function_declarations() -> list[types.FunctionDeclaration]:
    """Return all agent FunctionDeclarations configured with NON_BLOCKING behavior."""
    return [
        types.FunctionDeclaration(
            name="search_flights",
            description=(
                "Searches available flights between an origin and destination city. "
                "Runs asynchronously in the background. Invoke when the user asks about "
                "flight options, ticket prices, or airlines."
            ),
            behavior=types.Behavior.NON_BLOCKING,
            parameters={
                "type": "OBJECT",
                "properties": {
                    "origin": {
                        "type": "STRING",
                        "description": "Departure city or airport code (e.g., San Francisco or SFO).",
                    },
                    "destination": {
                        "type": "STRING",
                        "description": "Arrival city or airport code (e.g., Tokyo or NRT).",
                    },
                    "departure_date": {
                        "type": "STRING",
                        "description": "Optional travel date (e.g., 2026-12-01 or next Friday).",
                    },
                    "cabin_class": {
                        "type": "STRING",
                        "description": "Optional cabin class: economy, premium_economy, or business.",
                    },
                },
                "required": ["destination"],
            },
        ),
        types.FunctionDeclaration(
            name="get_weather_forecast",
            description=(
                "Fetches current weather and short-term forecast for a city. "
                "Runs asynchronously in the background. Invoke when the user asks "
                "about weather, temperature, rain, or forecast."
            ),
            behavior=types.Behavior.NON_BLOCKING,
            parameters={
                "type": "OBJECT",
                "properties": {
                    "city": {
                        "type": "STRING",
                        "description": "City name to look up weather for (e.g., Seattle, Tokyo, London).",
                    },
                    "units": {
                        "type": "STRING",
                        "description": "Temperature unit: 'celsius' or 'fahrenheit'.",
                    },
                },
                "required": ["city"],
            },
        ),
        types.FunctionDeclaration(
            name="deep_research_topic",
            description=(
                "Runs a multi-step background research briefing on a topic and returns "
                "key findings. Invoke when the user asks for research, background info, "
                "or a multi-angle summary."
            ),
            behavior=types.Behavior.NON_BLOCKING,
            parameters={
                "type": "OBJECT",
                "properties": {
                    "topic": {
                        "type": "STRING",
                        "description": "Topic, concept, or entity to research.",
                    },
                    "focus": {
                        "type": "STRING",
                        "description": "Optional focus angle (e.g., history, architecture, pros and cons).",
                    },
                },
                "required": ["topic"],
            },
        ),
        types.FunctionDeclaration(
            name="set_timer_or_reminder",
            description=(
                "Starts a live asynchronous countdown timer or reminder for a specified "
                "number of seconds. When the timer finishes, the result interrupts or "
                "notifies the conversation. Invoke when the user asks to set a timer or reminder."
            ),
            behavior=types.Behavior.NON_BLOCKING,
            parameters={
                "type": "OBJECT",
                "properties": {
                    "seconds": {
                        "type": "INTEGER",
                        "description": "Duration in seconds (between 1 and 3600).",
                    },
                    "label": {
                        "type": "STRING",
                        "description": "Short label describing what the timer is for (e.g., Tea, Stand up).",
                    },
                },
                "required": ["seconds"],
            },
        ),
        types.FunctionDeclaration(
            name="control_workspace_device",
            description=(
                "Reads or controls smart workspace devices: lights, thermostat, ambient music, "
                "or do-not-disturb mode. Invoke whenever the user wants to change room lighting, "
                "temperature, music, focus mode, or check workspace status."
            ),
            behavior=types.Behavior.NON_BLOCKING,
            parameters={
                "type": "OBJECT",
                "properties": {
                    "device": {
                        "type": "STRING",
                        "description": "Target device: 'lights', 'thermostat', 'music', 'do_not_disturb', or 'status'.",
                    },
                    "action": {
                        "type": "STRING",
                        "description": "Action: 'on', 'off', 'set', or 'status'.",
                    },
                    "value": {
                        "type": "STRING",
                        "description": "Optional value: brightness (0-100) or color for lights, temperature in C (16-30) for thermostat, or genre for music.",
                    },
                },
                "required": ["device"],
            },
        ),
    ]


def get_tools_metadata(scheduling_overrides: dict[str, str] | None = None) -> list[dict[str, Any]]:
    """Return UI-friendly metadata describing available non-blocking tools."""
    overrides = scheduling_overrides or {}
    meta = []
    for decl in get_function_declarations():
        name = decl.name or ""
        meta.append(
            {
                "name": name,
                "description": decl.description or "",
                "behavior": "NON_BLOCKING",
                "default_scheduling": overrides.get(
                    name, DEFAULT_TOOL_SCHEDULING.get(name, "WHEN_IDLE")
                ),
            }
        )
    return meta


class NonBlockingToolExecutor:
    """Executes Live API function calls concurrently without blocking audio streams."""

    def __init__(
        self,
        workspace: WorkspaceState | None = None,
        scheduling_overrides: dict[str, str] | None = None,
        simulate_latency: bool = True,
    ) -> None:
        self.workspace = workspace or WorkspaceState()
        self.scheduling_overrides = dict(scheduling_overrides or {})
        self.simulate_latency = simulate_latency
        self.active_tasks: dict[str, asyncio.Task[None]] = {}
        self.cancelled_ids: set[str] = set()
        self.records: dict[str, dict[str, Any]] = {}

    def update_scheduling(self, tool_name: str, scheduling: str) -> None:
        """Update per-tool scheduling mode dynamically during a session."""
        norm = scheduling.strip().upper()
        if norm in ("INTERRUPT", "WHEN_IDLE", "SILENT"):
            self.scheduling_overrides[tool_name] = norm

    def get_scheduling(self, tool_name: str) -> str:
        """Return effective scheduling mode for a tool."""
        return self.scheduling_overrides.get(
            tool_name, DEFAULT_TOOL_SCHEDULING.get(tool_name, "WHEN_IDLE")
        )

    def dispatch(
        self,
        function_calls: list[types.FunctionCall],
        *,
        allow_scheduling: bool,
        on_event: EventCallback,
        on_complete: CompletionCallback,
    ) -> list[asyncio.Task[None]]:
        """Spawn non-blocking background tasks for each incoming FunctionCall."""
        spawned: list[asyncio.Task[None]] = []
        for idx, fc in enumerate(function_calls):
            call_id = fc.id or f"call_{int(time.time() * 1000)}_{idx}"
            name = fc.name or "unknown_tool"
            args = dict(fc.args or {})
            scheduling = self.get_scheduling(name) if allow_scheduling else None

            record = {
                "id": call_id,
                "name": name,
                "args": args,
                "status": "running",
                "scheduling": scheduling or "DEFAULT (Extended Thinking)",
                "progress": "Starting...",
                "started_at": time.time(),
                "finished_at": None,
                "duration_ms": None,
                "result": None,
                "error": None,
            }
            self.records[call_id] = record

            task = asyncio.create_task(
                self._run_single(
                    call_id=call_id,
                    name=name,
                    args=args,
                    scheduling=scheduling,
                    on_event=on_event,
                    on_complete=on_complete,
                ),
                name=f"live-tool-{name}-{call_id}",
            )
            self.active_tasks[call_id] = task
            spawned.append(task)
        return spawned

    async def cancel(self, ids: list[str], on_event: EventCallback) -> list[str]:
        """Cancel running background tool tasks requested by tool_call_cancellation."""
        cancelled: list[str] = []
        for call_id in ids:
            self.cancelled_ids.add(call_id)
            task = self.active_tasks.pop(call_id, None)
            if task and not task.done():
                task.cancel()
            if call_id in self.workspace.timers:
                self.workspace.timers[call_id]["status"] = "cancelled"
            if call_id in self.records:
                rec = self.records[call_id]
                if rec["status"] == "running":
                    rec["status"] = "cancelled"
                    rec["progress"] = "Cancelled on user interruption"
                    rec["finished_at"] = time.time()
                    rec["duration_ms"] = int((rec["finished_at"] - rec["started_at"]) * 1000)
                    await on_event({"type": "tool_cancelled", "tool": dict(rec)})
            cancelled.append(call_id)

        if cancelled:
            await on_event(
                {"type": "workspace_state", "state": self.workspace.snapshot()}
            )
        return cancelled

    async def shutdown(self) -> None:
        """Cancel and await all active tasks on session teardown."""
        tasks = list(self.active_tasks.values())
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self.active_tasks.clear()

    async def _run_single(
        self,
        *,
        call_id: str,
        name: str,
        args: dict[str, Any],
        scheduling: str | None,
        on_event: EventCallback,
        on_complete: CompletionCallback,
    ) -> None:
        rec = self.records[call_id]
        await on_event({"type": "tool_started", "tool": dict(rec)})

        async def report_progress(message: str) -> None:
            if call_id in self.cancelled_ids:
                return
            rec["progress"] = message
            await on_event({"type": "tool_progress", "tool": dict(rec)})

        try:
            output = await self._execute_tool(call_id, name, args, report_progress, on_event)
            if call_id in self.cancelled_ids:
                return

            rec["status"] = "completed"
            rec["progress"] = "Completed"
            rec["finished_at"] = time.time()
            rec["duration_ms"] = int((rec["finished_at"] - rec["started_at"]) * 1000)
            rec["result"] = output

            fr_kwargs: dict[str, Any] = {
                "id": call_id,
                "name": name,
                "response": {"output": output},
            }
            if scheduling is not None:
                fr_kwargs["scheduling"] = types.FunctionResponseScheduling(scheduling)

            fn_response = types.FunctionResponse(**fr_kwargs)
            await on_event({"type": "tool_completed", "tool": dict(rec)})
            await on_complete(fn_response, dict(rec))

        except asyncio.CancelledError:
            if rec["status"] == "running":
                rec["status"] = "cancelled"
                rec["progress"] = "Cancelled"
                rec["finished_at"] = time.time()
                rec["duration_ms"] = int((rec["finished_at"] - rec["started_at"]) * 1000)
                await on_event({"type": "tool_cancelled", "tool": dict(rec)})
            raise
        except Exception as exc:
            if call_id in self.cancelled_ids:
                return
            rec["status"] = "error"
            rec["progress"] = f"Error: {exc}"
            rec["error"] = str(exc)
            rec["finished_at"] = time.time()
            rec["duration_ms"] = int((rec["finished_at"] - rec["started_at"]) * 1000)

            fr_kwargs = {
                "id": call_id,
                "name": name,
                "response": {"error": str(exc)},
            }
            if scheduling is not None:
                fr_kwargs["scheduling"] = types.FunctionResponseScheduling(scheduling)

            await on_event({"type": "tool_error", "tool": dict(rec)})
            await on_complete(types.FunctionResponse(**fr_kwargs), dict(rec))
        finally:
            self.active_tasks.pop(call_id, None)

    async def _execute_tool(
        self,
        call_id: str,
        name: str,
        args: dict[str, Any],
        report_progress: Callable[[str], Awaitable[None]],
        on_event: EventCallback,
    ) -> dict[str, Any]:
        if name == "search_flights":
            return await self._tool_search_flights(args, report_progress)
        if name == "get_weather_forecast":
            return await self._tool_get_weather(args, report_progress)
        if name == "deep_research_topic":
            return await self._tool_deep_research(args, report_progress)
        if name == "set_timer_or_reminder":
            return await self._tool_set_timer(call_id, args, report_progress, on_event)
        if name == "control_workspace_device":
            return await self._tool_control_workspace(args, report_progress, on_event)
        raise ValueError(f"Unknown function tool: {name}")

    async def _sleep(self, seconds: float) -> None:
        if self.simulate_latency and seconds > 0:
            await asyncio.sleep(seconds)

    async def _tool_search_flights(
        self,
        args: dict[str, Any],
        report_progress: Callable[[str], Awaitable[None]],
    ) -> dict[str, Any]:
        origin = str(args.get("origin") or "San Francisco (SFO)")
        destination = str(args.get("destination") or "Seattle (SEA)")
        departure_date = str(args.get("departure_date") or "upcoming weekend")
        cabin = str(args.get("cabin_class") or "economy")

        await report_progress(f"Querying airline schedules for {origin} -> {destination}...")
        await self._sleep(1.1)
        await report_progress("Comparing fares, nonstop options, and seat availability...")
        await self._sleep(1.2)

        multiplier = {"economy": 1.0, "premium_economy": 1.55, "business": 2.9}.get(
            cabin.lower(), 1.0
        )
        base_prices = [189, 245, 312]
        flights = [
            {
                "flight_number": "DL 418",
                "airline": "Delta Air Lines",
                "departure": "08:15",
                "arrival": "10:42",
                "stops": 0,
                "cabin_class": cabin,
                "price_usd": round(base_prices[0] * multiplier),
            },
            {
                "flight_number": "UA 902",
                "airline": "United Airlines",
                "departure": "13:30",
                "arrival": "16:05",
                "stops": 0,
                "cabin_class": cabin,
                "price_usd": round(base_prices[1] * multiplier),
            },
            {
                "flight_number": "AS 311",
                "airline": "Alaska Airlines",
                "departure": "18:45",
                "arrival": "21:18",
                "stops": 0,
                "cabin_class": cabin,
                "price_usd": round(base_prices[2] * multiplier),
            },
        ]
        return {
            "origin": origin,
            "destination": destination,
            "departure_date": departure_date,
            "flights": flights,
            "cheapest_option": flights[0],
        }

    async def _tool_get_weather(
        self,
        args: dict[str, Any],
        report_progress: Callable[[str], Awaitable[None]],
    ) -> dict[str, Any]:
        city = str(args.get("city") or "San Francisco").strip()
        units = str(args.get("units") or "celsius").strip().lower()
        temp_unit = "fahrenheit" if units.startswith("f") else "celsius"

        await report_progress(f"Geocoding {city}...")
        await self._sleep(0.5)

        try:
            async with httpx.AsyncClient(timeout=5.0) as http:
                geo = await http.get(
                    "https://geocoding-api.open-meteo.com/v1/search",
                    params={"name": city, "count": 1, "language": "en", "format": "json"},
                )
                geo.raise_for_status()
                geo_data = geo.json()
                results = geo_data.get("results") or []
                if results:
                    loc = results[0]
                    await report_progress(
                        f"Fetching live weather for {loc['name']}, {loc.get('country', '')}..."
                    )
                    wx = await http.get(
                        "https://api.open-meteo.com/v1/forecast",
                        params={
                            "latitude": loc["latitude"],
                            "longitude": loc["longitude"],
                            "current": "temperature_2m,relative_humidity_2m,apparent_temperature,wind_speed_10m,weather_code",
                            "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max",
                            "temperature_unit": temp_unit,
                            "forecast_days": 2,
                        },
                    )
                    wx.raise_for_status()
                    wx_data = wx.json()
                    cur = wx_data.get("current", {})
                    daily = wx_data.get("daily", {})
                    code = int(cur.get("weather_code", 0))
                    return {
                        "source": "open-meteo-live",
                        "city": loc["name"],
                        "country": loc.get("country", ""),
                        "units": temp_unit,
                        "temperature": cur.get("temperature_2m"),
                        "feels_like": cur.get("apparent_temperature"),
                        "humidity_percent": cur.get("relative_humidity_2m"),
                        "wind_speed_kmh": cur.get("wind_speed_10m"),
                        "condition": _describe_wmo_code(code),
                        "today_high": (daily.get("temperature_2m_max") or [None])[0],
                        "today_low": (daily.get("temperature_2m_min") or [None])[0],
                        "precipitation_chance_percent": (
                            daily.get("precipitation_probability_max") or [None]
                        )[0],
                    }
        except Exception:
            pass

        await report_progress(f"Using cached regional weather estimate for {city}...")
        await self._sleep(0.5)
        base_c = 18
        temp = round(base_c * 9 / 5 + 32) if temp_unit == "fahrenheit" else base_c
        return {
            "source": "simulated-fallback",
            "city": city,
            "units": temp_unit,
            "temperature": temp,
            "condition": "Partly cloudy with light breeze",
            "humidity_percent": 62,
            "precipitation_chance_percent": 15,
        }

    async def _tool_deep_research(
        self,
        args: dict[str, Any],
        report_progress: Callable[[str], Awaitable[None]],
    ) -> dict[str, Any]:
        topic = str(args.get("topic") or "Gemini Live API").strip()
        focus = str(args.get("focus") or "key overview and practical takeaways").strip()

        await report_progress(f"Searching reference summaries on '{topic}'...")
        await self._sleep(0.9)

        summary_text: str | None = None
        source = "wikipedia-rest"
        try:
            slug = urllib.parse.quote(topic.replace(" ", "_"))
            async with httpx.AsyncClient(timeout=5.0) as http:
                resp = await http.get(
                    f"https://en.wikipedia.org/api/rest_v1/page/summary/{slug}",
                    headers={"User-Agent": "GeminiLiveVoiceAgent/0.1"},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("extract"):
                        summary_text = data["extract"]
        except Exception:
            summary_text = None

        await report_progress(f"Synthesizing briefing focused on {focus}...")
        await self._sleep(1.1)

        if not summary_text:
            source = "local-synthesis"
            summary_text = (
                f"Background synthesis completed for '{topic}' (focus: {focus}). "
                "The topic involves core architectural tradeoffs around latency, reliability, "
                "and real-world operational constraints."
            )

        return {
            "topic": topic,
            "focus": focus,
            "source": source,
            "summary": summary_text,
            "key_takeaways": [
                f"Primary definition and context for {topic}.",
                f"Specific focus evaluated: {focus}.",
                "Ready for follow-up questions in conversation.",
            ],
        }

    async def _tool_set_timer(
        self,
        call_id: str,
        args: dict[str, Any],
        report_progress: Callable[[str], Awaitable[None]],
        on_event: EventCallback,
    ) -> dict[str, Any]:
        raw_seconds = args.get("seconds", 5)
        try:
            seconds = max(1, min(3600, int(raw_seconds)))
        except (TypeError, ValueError):
            seconds = 5
        label = str(args.get("label") or "Timer").strip()

        self.workspace.timers[call_id] = {
            "id": call_id,
            "label": label,
            "total_seconds": seconds,
            "remaining_seconds": seconds,
            "status": "running",
        }
        await on_event({"type": "workspace_state", "state": self.workspace.snapshot()})

        if not self.simulate_latency:
            self.workspace.timers[call_id]["remaining_seconds"] = 0
            self.workspace.timers[call_id]["status"] = "finished"
            await on_event({"type": "workspace_state", "state": self.workspace.snapshot()})
            return {
                "status": "timer_expired",
                "label": label,
                "duration_seconds": seconds,
                "message": f"The {seconds}-second timer for '{label}' has finished.",
            }

        for remaining in range(seconds, 0, -1):
            self.workspace.timers[call_id]["remaining_seconds"] = remaining
            await report_progress(f"'{label}' — {remaining}s remaining")
            await on_event({"type": "workspace_state", "state": self.workspace.snapshot()})
            await asyncio.sleep(1.0)

        self.workspace.timers[call_id]["remaining_seconds"] = 0
        self.workspace.timers[call_id]["status"] = "finished"
        await on_event({"type": "workspace_state", "state": self.workspace.snapshot()})

        return {
            "status": "timer_expired",
            "label": label,
            "duration_seconds": seconds,
            "message": f"The {seconds}-second timer for '{label}' has finished.",
        }

    async def _tool_control_workspace(
        self,
        args: dict[str, Any],
        report_progress: Callable[[str], Awaitable[None]],
        on_event: EventCallback,
    ) -> dict[str, Any]:
        device = str(args.get("device") or "status").strip().lower()
        action = str(args.get("action") or "status").strip().lower()
        value = args.get("value")

        await report_progress(f"Updating workspace device '{device}'...")
        await self._sleep(0.15)

        if device in ("lights", "light", "lamp"):
            if action == "on":
                self.workspace.lights_on = True
            elif action == "off":
                self.workspace.lights_on = False
            if value is not None:
                val_str = str(value).strip()
                if val_str.rstrip("%").isdigit():
                    self.workspace.lights_brightness = max(
                        0, min(100, int(val_str.rstrip("%")))
                    )
                    self.workspace.lights_on = self.workspace.lights_brightness > 0
                else:
                    self.workspace.lights_color = val_str
                    self.workspace.lights_on = True
        elif device in ("thermostat", "temperature", "ac", "heat"):
            if value is not None:
                try:
                    temp_val = int(float(str(value).replace("C", "").replace("°", "").strip()))
                    self.workspace.thermostat_c = max(16, min(30, temp_val))
                except ValueError:
                    pass
        elif device in ("music", "audio", "speaker"):
            if action in ("on", "play", "start"):
                self.workspace.music_playing = True
            elif action in ("off", "stop", "pause"):
                self.workspace.music_playing = False
            if value is not None and str(value).strip():
                self.workspace.music_genre = str(value).strip()
                if action != "off":
                    self.workspace.music_playing = True
        elif device in ("do_not_disturb", "dnd", "focus"):
            if action in ("on", "enable"):
                self.workspace.do_not_disturb = True
            elif action in ("off", "disable"):
                self.workspace.do_not_disturb = False

        snapshot = self.workspace.snapshot()
        await on_event({"type": "workspace_state", "state": snapshot})
        return {
            "device": device,
            "action": action,
            "workspace_state": snapshot,
        }


def _describe_wmo_code(code: int) -> str:
    mapping = {
        0: "Clear sky",
        1: "Mainly clear",
        2: "Partly cloudy",
        3: "Overcast",
        45: "Foggy",
        48: "Depositing rime fog",
        51: "Light drizzle",
        53: "Moderate drizzle",
        55: "Dense drizzle",
        61: "Slight rain",
        63: "Moderate rain",
        65: "Heavy rain",
        71: "Slight snow",
        73: "Moderate snow",
        75: "Heavy snow",
        80: "Slight rain showers",
        81: "Moderate rain showers",
        82: "Violent rain showers",
        95: "Thunderstorm",
    }
    return mapping.get(code, f"Weather code {code}")


def safe_json_dumps(data: Any) -> str:
    """Serialize arbitrary payload cleanly for debug logging."""
    return json.dumps(data, default=str)

"""Classroom Pulse: a local, heuristic classroom analytics dashboard."""

from datetime import datetime, timezone
from html import escape
import os
import tempfile

import cv2
import pandas as pd
import plotly.express as px
import streamlit as st

from engine import Config, advisories, fuse, process_audio, process_photo, process_video


st.set_page_config(
    page_title="Classroom Pulse",
    page_icon="C",
    layout="wide",
    initial_sidebar_state="expanded",
)

css = """
<style>
:root {
    color-scheme: dark;
    --page: #0B1120;
    --sidebar: #111827;
    --card: #1E293B;
    --accent: #6366F1;
    --cyan: #22D3EE;
    --text: #F8FAFC;
    --muted: #94A3B8;
    --success: #10B981;
    --warning: #F59E0B;
    --danger: #EF4444;
}
.stApp, [data-testid="stAppViewContainer"] {
    background: var(--page);
    color: var(--text);
}
[data-testid="stHeader"] { background: rgba(11, 17, 32, .9); }
[data-testid="stSidebar"] { background: var(--sidebar); }
[data-testid="stSidebar"] * { color: var(--text); }
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1440px; }
h1, h2, h3, p, label, [data-testid="stMarkdownContainer"] {
    color: var(--text);
}
[data-testid="stCaptionContainer"], .muted-text { color: var(--muted) !important; }
button[data-baseweb="tab"] {
    color: var(--muted) !important;
    font-weight: 600;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: var(--text) !important;
    border-bottom-color: var(--accent) !important;
}
[data-testid="stFileUploader"], [data-testid="stCameraInput"] {
    background: rgba(30, 41, 59, .55);
    border: 1px solid #334155;
    border-radius: 14px;
}
[data-testid="stFileUploaderDropzone"] { background: transparent; }
[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] span,
[data-testid="stCameraInput"] button { color: var(--muted) !important; }
[data-testid="stAlert"] { border-radius: 12px; }
div.stButton > button, div.stDownloadButton > button {
    min-height: 2.7rem;
    border-radius: 10px;
    border: 1px solid #475569;
    color: var(--text);
    background: #263449;
    font-weight: 600;
}
div.stButton > button[kind="primary"] {
    color: white;
    background: var(--accent);
    border-color: var(--accent);
}
div.stButton > button:hover, div.stDownloadButton > button:hover {
    border-color: var(--cyan);
    color: white;
}
input, textarea, [data-baseweb="select"] > div {
    color: var(--text) !important;
    background-color: #1E293B !important;
}
[data-testid="stDataFrame"], [data-testid="stTable"] {
    border: 1px solid #334155;
    border-radius: 12px;
    overflow: hidden;
}
.dashboard-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 1rem;
    padding: .2rem 0 1.4rem;
}
.dashboard-title { margin: 0; font-size: 2.15rem; font-weight: 750; letter-spacing: -.04em; }
.dashboard-subtitle { margin: .35rem 0 0; color: var(--muted); font-size: 1rem; }
.system-status {
    display: inline-flex;
    align-items: center;
    gap: .55rem;
    white-space: nowrap;
    padding: .55rem .8rem;
    border-radius: 999px;
    color: #D1FAE5;
    background: rgba(16, 185, 129, .12);
    border: 1px solid rgba(16, 185, 129, .3);
    font-size: .82rem;
    font-weight: 650;
}
.status-dot { width: .5rem; height: .5rem; border-radius: 50%; background: var(--success); }
.metric-card {
    min-height: 112px;
    padding: 1rem 1.1rem;
    margin-bottom: .8rem;
    border: 1px solid #334155;
    border-radius: 15px;
    background: var(--card);
}
.metric-label { color: var(--muted); font-size: .85rem; font-weight: 550; }
.metric-value { color: var(--text); margin-top: .35rem; font-size: 1.65rem; font-weight: 720; }
.metric-detail { color: var(--muted); margin-top: .2rem; font-size: .76rem; }
.section-heading { margin: .4rem 0 1rem; }
.section-heading h2 { margin: 0; font-size: 1.45rem; font-weight: 700; }
.section-heading p { margin: .35rem 0 0; color: var(--muted); }
.status-pill {
    display: inline-block;
    padding: .23rem .62rem;
    border-radius: 999px;
    font-size: .78rem;
    font-weight: 700;
}
.status-present { color: #A7F3D0; background: rgba(16, 185, 129, .16); }
.status-absent { color: #FECACA; background: rgba(239, 68, 68, .16); }
.status-neutral { color: #CBD5E1; background: rgba(148, 163, 184, .15); }
.status-warning { color: #FDE68A; background: rgba(245, 158, 11, .16); }
.attendance-table { width: 100%; border-collapse: collapse; overflow: hidden; border-radius: 12px; }
.attendance-table th {
    padding: .8rem .9rem;
    text-align: left;
    color: var(--muted);
    background: #172235;
    border-bottom: 1px solid #334155;
    font-size: .82rem;
}
.attendance-table td {
    padding: .8rem .9rem;
    color: var(--text);
    background: var(--card);
    border-bottom: 1px solid #334155;
    font-size: .9rem;
}
.empty-state {
    padding: 1.2rem 1.3rem;
    border: 1px dashed #475569;
    border-radius: 14px;
    color: var(--muted);
    background: rgba(30, 41, 59, .45);
}
.advisory-card {
    padding: 1rem 1.1rem;
    margin: .5rem 0;
    border: 1px solid #334155;
    border-left: 3px solid var(--accent);
    border-radius: 12px;
    background: var(--card);
}
@media (max-width: 700px) {
    .dashboard-header { align-items: flex-start; flex-direction: column; }
    .dashboard-title { font-size: 1.8rem; }
}
</style>
"""
st.markdown(css, unsafe_allow_html=True)


def render_header() -> None:
    st.markdown(
        """
        <div class="dashboard-header">
          <div>
            <h1 class="dashboard-title">Classroom Pulse</h1>
            <p class="dashboard-subtitle">AI-Powered Classroom Attendance &amp; Engagement Analytics</p>
          </div>
          <div class="system-status"><span class="status-dot"></span>System Ready&nbsp; | &nbsp;Local Processing</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(
    label: str,
    value: str,
    detail: str | None = None,
) -> None:
    detail_html = f'<div class="metric-detail">{escape(detail)}</div>' if detail else ""
    st.markdown(
        f"""
        <div class="metric-card">
          <div class="metric-label">{escape(label)}</div>
          <div class="metric-value">{escape(value)}</div>
          {detail_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_header(title: str, description: str | None = None) -> None:
    description_html = f"<p>{escape(description)}</p>" if description else ""
    st.markdown(
        f'<div class="section-heading"><h2>{escape(title)}</h2>{description_html}</div>',
        unsafe_allow_html=True,
    )


def render_status_badge(text: str, status: str = "neutral") -> None:
    allowed_statuses = {"present", "absent", "neutral", "warning"}
    badge_class = status if status in allowed_statuses else "neutral"
    st.markdown(
        f'<span class="status-pill status-{badge_class}">{escape(text)}</span>',
        unsafe_allow_html=True,
    )


def render_empty_state(message: str) -> None:
    st.markdown(
        f'<div class="empty-state">{escape(message)}</div>',
        unsafe_allow_html=True,
    )


def initialize_session() -> None:
    defaults = {
        "visual": pd.DataFrame(),
        "audio": pd.DataFrame(),
        "register": pd.DataFrame(),
        "duration": 0.0,
        "video_truncated": False,
        "video_name": "",
        "photo_result": None,
        "photo_count": None,
        "photo_source": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_session() -> None:
    st.session_state.visual = pd.DataFrame()
    st.session_state.audio = pd.DataFrame()
    st.session_state.register = pd.DataFrame()
    st.session_state.duration = 0.0
    st.session_state.video_truncated = False
    st.session_state.video_name = ""
    st.session_state.photo_result = None
    st.session_state.photo_count = None
    st.session_state.photo_source = ""


def format_duration(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    minutes, remainder = divmod(total_seconds, 60)
    if minutes:
        return f"{minutes}m {remainder}s"
    return f"{remainder}s"


def trend_summary(fused: pd.DataFrame) -> tuple[str, float | None]:
    if fused.empty or len(fused) < 2:
        return "Insufficient data", None
    ordered = fused.sort_values("time_s")
    segment_size = max(1, int(len(ordered) * 0.2))
    start_mean = float(ordered["engagement_index"].iloc[:segment_size].mean())
    end_mean = float(ordered["engagement_index"].iloc[-segment_size:].mean())
    change = end_mean - start_mean
    if change > 5:
        return "Increasing", change
    if change < -5:
        return "Decreasing", change
    return "Stable", change


def severity_for(score: float, threshold: float, trend: str) -> tuple[str, str]:
    if score < threshold:
        return "Elevated", "warning"
    if trend == "Decreasing":
        return "Watch", "warning"
    return "Low", "present"


def render_attendance_table(register: pd.DataFrame) -> None:
    headers = ["Student/Track ID", "First Seen", "Last Seen", "Duration", "Attendance Status"]
    rows: list[str] = []
    for _, item in register.iterrows():
        status = str(item["Status"])
        badge_status = "present" if status == "Present" else "absent"
        cells = [
            escape(str(item["Student ID"])),
            f'{float(item["First Seen (s)"]):.1f}s',
            f'{float(item["Last Seen (s)"]):.1f}s',
            f'{float(item["Total Duration (s)"]):.1f}s',
            f'<span class="status-pill status-{badge_status}">{escape(status)}</span>',
        ]
        rows.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>")
    header_html = "".join(f"<th>{escape(item)}</th>" for item in headers)
    st.markdown(
        f'<table class="attendance-table"><thead><tr>{header_html}</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table>',
        unsafe_allow_html=True,
    )


def configure_chart(figure: object) -> object:
    figure.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(30,41,59,.35)",
        font_color="#F8FAFC",
        margin=dict(l=12, r=12, t=48, b=12),
        legend_title_text="",
    )
    return figure


initialize_session()
render_header()

with st.sidebar:
    st.markdown("### Session Settings")
    threshold = st.slider("Attendance threshold (%)", 50, 95, 70) / 100
    window = st.select_slider(
        "Analysis window (seconds)",
        options=[1, 2, 5, 10],
        value=2,
    )
    low = st.slider("Low engagement threshold", 20, 80, 45)
    alert_duration = st.number_input(
        "Alert duration (seconds)",
        min_value=10,
        max_value=1800,
        value=300,
        step=10,
    )
    config = Config(
        presence_threshold=threshold,
        window_seconds=float(window),
        low_engagement=float(low),
        low_duration_seconds=float(alert_duration),
    )

    st.markdown("### Session Controls")
    if st.button("Reset session", key="sidebar_reset", use_container_width=True):
        reset_session()
        st.rerun()

    with st.expander("View session status"):
        if st.session_state.video_name:
            st.write(f"Video analyzed: {st.session_state.video_name}")
            st.write(f"Duration: {format_duration(st.session_state.duration)}")
        else:
            st.write("No video analysis in this session.")
        st.write(f"Audio windows: {len(st.session_state.audio)}")
        st.write(f"Anonymous tracks: {len(st.session_state.register)}")

    with st.expander("Privacy & Information"):
        st.write(
            "Video, image, and WAV processing use local OpenCV and signal heuristics. "
            "No face recognition, identity matching, speech transcription, or external "
            "AI service is used."
        )
        st.write(
            "Face tracks are temporary anonymous IDs, not verified student identities. "
            "Presence and engagement values are exploratory proxies and should not be "
            "used for grading or disciplinary decisions."
        )
        st.write(
            "When hosted remotely, uploaded media is processed on that host. Run the app "
            "on the classroom device to keep processing on that device."
        )


tabs = st.tabs(
    ["ATTENDANCE", "AUDIO INSIGHTS", "ENGAGEMENT", "AI SUGGESTIONS", "REPORTS"]
)
attendance_tab, audio_tab, engagement_tab, suggestions_tab, reports_tab = tabs


with attendance_tab:
    render_section_header(
        "Smart Attendance Monitoring",
        "Review sustained presence estimates from anonymous face tracks.",
    )
    st.caption(
        "Track IDs are not verified identities. A single image or webcam snapshot "
        "cannot establish attendance duration."
    )
    source = st.radio(
        "Choose an input",
        ["Classroom video", "Classroom image", "Webcam snapshot"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if source == "Classroom video":
        upload = st.file_uploader(
            "Upload a classroom video (MP4, AVI, or MOV)",
            type=["mp4", "avi", "mov"],
            key="classroom_video",
        )
        if upload is not None:
            video_bytes = upload.getvalue()
            if not video_bytes:
                st.error("This video file is empty. Choose a valid video and try again.")
            elif upload.size > 200 * 1024 * 1024:
                st.error("Video files must be smaller than 200 MB.")
            else:
                st.video(video_bytes)
                if st.button("Process attendance", type="primary", key="process_video"):
                    suffix = os.path.splitext(upload.name)[1].lower()
                    if suffix not in {".mp4", ".avi", ".mov"}:
                        st.error("Unsupported video format. Upload MP4, AVI, or MOV.")
                    else:
                        temporary_path = None
                        try:
                            with tempfile.NamedTemporaryFile(
                                delete=False,
                                suffix=suffix,
                            ) as temporary_file:
                                temporary_path = temporary_file.name
                                temporary_file.write(video_bytes)
                            with st.spinner("Analyzing video frames..."):
                                register, visual, duration, truncated = process_video(
                                    temporary_path,
                                    config,
                                )
                            st.session_state.register = register
                            st.session_state.visual = visual
                            st.session_state.duration = duration
                            st.session_state.video_truncated = truncated
                            st.session_state.video_name = upload.name
                            st.success("Attendance analysis is ready.")
                            if truncated:
                                st.warning(
                                    "The processing limit was reached; the video "
                                    "analysis covers only its first portion."
                                )
                        except (ValueError, RuntimeError, OSError, cv2.error) as error:
                            st.error(
                                f"Could not process this video. Check that it is a "
                                f"valid, supported video file. Details: {error}"
                            )
                        finally:
                            if temporary_path and os.path.exists(temporary_path):
                                try:
                                    os.remove(temporary_path)
                                except OSError:
                                    st.warning(
                                        "The temporary video file could not be removed."
                                    )

    elif source == "Classroom image":
        upload = st.file_uploader(
            "Upload a classroom image (JPG or PNG)",
            type=["jpg", "jpeg", "png"],
            key="classroom_image",
        )
        if upload is not None:
            image_bytes = upload.getvalue()
            if not image_bytes:
                st.error("This image file is empty. Choose a valid image and try again.")
            else:
                st.image(image_bytes, caption="Uploaded classroom image", use_container_width=True)
                if st.button("Detect participants", type="primary", key="process_image"):
                    try:
                        result, count = process_photo(image_bytes)
                        st.session_state.photo_result = result
                        st.session_state.photo_count = count
                        st.session_state.photo_source = upload.name
                    except (ValueError, RuntimeError, OSError, cv2.error) as error:
                        st.error(f"Could not read this image. Choose a valid JPG or PNG. Details: {error}")
                if (
                    st.session_state.photo_result is not None
                    and st.session_state.photo_source == upload.name
                ):
                    st.image(
                        st.session_state.photo_result,
                        caption=f'{st.session_state.photo_count} anonymous face detections',
                        use_container_width=True,
                    )
                    st.info("Snapshot only: presence duration is not available.")

    else:
        st.caption(
            "If the camera does not open, allow camera access in your browser and try again."
        )
        camera_image = st.camera_input(
            "Capture a webcam snapshot",
            help="Allow camera access in your browser to take a single snapshot.",
        )
        if camera_image is not None:
            image_bytes = camera_image.getvalue()
            if not image_bytes:
                st.error("The camera returned an empty image. Check camera permission and try again.")
            elif st.button("Detect participants", type="primary", key="process_camera"):
                try:
                    result, count = process_photo(image_bytes)
                    st.session_state.photo_result = result
                    st.session_state.photo_count = count
                    st.session_state.photo_source = "webcam"
                except (ValueError, RuntimeError, OSError, cv2.error) as error:
                    st.error(
                        "Could not process the webcam snapshot. Check browser camera "
                        f"permission and try again. Details: {error}"
                    )
            if (
                st.session_state.photo_result is not None
                and st.session_state.photo_source == "webcam"
            ):
                st.image(
                    st.session_state.photo_result,
                    caption=f'{st.session_state.photo_count} anonymous face detections',
                    use_container_width=True,
                )
                st.info("Snapshot only: presence duration is not available.")

    register = st.session_state.register
    if not register.empty:
        st.markdown("---")
        render_section_header("Attendance results")
        present_count = int((register["Status"] == "Present").sum())
        metric_columns = st.columns(3)
        with metric_columns[0]:
            render_metric_card("Detected participants", str(len(register)), "Anonymous tracks")
        with metric_columns[1]:
            render_metric_card("Present tracks", str(present_count), "Sustained presence")
        with metric_columns[2]:
            render_metric_card("Video duration", format_duration(st.session_state.duration))
        render_attendance_table(register)
        st.download_button(
            "Download Attendance CSV",
            data=register.to_csv(index=False).encode("utf-8"),
            file_name="attendance_register.csv",
            mime="text/csv",
        )
    else:
        render_empty_state("Upload and process a video to create an attendance register.")


with audio_tab:
    render_section_header(
        "Classroom Acoustic Intelligence",
        "Explore acoustic energy and heuristic activity categories from a WAV recording.",
    )
    audio_upload = st.file_uploader(
        "Upload a PCM WAV audio file",
        type=["wav"],
        key="classroom_audio",
    )
    if audio_upload is not None:
        audio_bytes = audio_upload.getvalue()
        if not audio_bytes:
            st.error("This audio file is empty. Choose a valid WAV recording.")
        else:
            st.audio(audio_bytes)
            if st.button("Analyze acoustic activity", type="primary", key="process_audio"):
                try:
                    audio_result = process_audio(audio_bytes, float(window))
                    if audio_result.empty:
                        st.error(
                            "No usable audio data was found. Choose a PCM WAV file "
                            "containing a recording."
                        )
                    else:
                        st.session_state.audio = audio_result
                        st.success("Acoustic analysis is ready.")
                except (ValueError, RuntimeError, OSError, EOFError) as error:
                    st.error(
                        "Could not analyze this WAV file. Check that it contains "
                        f"supported audio data. Details: {error}"
                    )

    audio = st.session_state.audio
    if audio.empty:
        render_empty_state("Upload a WAV recording to view acoustic activity insights.")
    else:
        category_counts = audio["audio_state"].value_counts()
        dominant_category = str(category_counts.index[0])
        columns = st.columns(3)
        with columns[0]:
            render_metric_card("Average acoustic energy", f'{audio["rms"].mean():.3f}', "RMS")
        with columns[1]:
            render_metric_card("Peak acoustic energy", f'{audio["rms"].max():.3f}', "RMS")
        with columns[2]:
            render_metric_card("Most frequent category", dominant_category)
        st.caption(
            "Categories are signal heuristics; they do not identify instructor speech "
            "or distinguish speakers."
        )
        energy_chart = px.line(
            audio,
            x="time_s",
            y="rms",
            markers=True,
            labels={"time_s": "Time (seconds)", "rms": "RMS energy"},
            title="Audio activity timeline",
        )
        st.plotly_chart(configure_chart(energy_chart), use_container_width=True)
        chart_columns = st.columns(2)
        with chart_columns[0]:
            category_chart = px.bar(
                category_counts.rename_axis("Category").reset_index(name="Windows"),
                x="Category",
                y="Windows",
                title="Detected acoustic categories",
                color="Category",
                color_discrete_sequence=["#6366F1", "#22D3EE", "#10B981", "#F59E0B"],
            )
            st.plotly_chart(configure_chart(category_chart), use_container_width=True)
        with chart_columns[1]:
            timeline_chart = px.scatter(
                audio,
                x="time_s",
                y="audio_state",
                color="audio_state",
                labels={"time_s": "Time (seconds)", "audio_state": "Acoustic category"},
                title="Acoustic category timeline",
                color_discrete_sequence=["#6366F1", "#22D3EE", "#10B981", "#F59E0B"],
            )
            st.plotly_chart(configure_chart(timeline_chart), use_container_width=True)


fused = fuse(st.session_state.visual, st.session_state.audio, float(window))


with engagement_tab:
    render_section_header(
        "Classroom Engagement Analytics",
        "Session-level exploratory proxies based only on analyzed video and available audio.",
    )
    if fused.empty:
        render_empty_state("Process a classroom video to generate the engagement timeline.")
    else:
        average_score = float(fused["engagement_index"].mean())
        visual_score = float(fused["visual_proxy"].mean() * 100)
        aligned_audio = fused["interaction_proxy"].dropna()
        audio_score = (
            f"{aligned_audio.mean() * 100:.1f}/100"
            if not aligned_audio.empty
            else "Not analyzed"
        )
        metric_columns = st.columns(4)
        with metric_columns[0]:
            render_metric_card(
                "Overall engagement score",
                f"{average_score:.1f}/100",
                "Heuristic proxy",
            )
        with metric_columns[1]:
            render_metric_card(
                "Visual presence indicator",
                f"{visual_score:.1f}%",
                "Detected-track proxy",
            )
        with metric_columns[2]:
            render_metric_card(
                "Audio interaction score",
                audio_score,
                "Heuristic; only when audio is aligned",
            )
        with metric_columns[3]:
            render_metric_card("Session duration", format_duration(st.session_state.duration))

        trend, trend_change = trend_summary(fused)
        trend_detail = (
            f"{trend_change:+.1f} points across session"
            if trend_change is not None
            else "More data needed"
        )
        trend_column, peak_column, dip_column = st.columns(3)
        with trend_column:
            render_metric_card("Engagement trend", trend, trend_detail)
        peak = fused.loc[fused["engagement_index"].idxmax()]
        dip = fused.loc[fused["engagement_index"].idxmin()]
        with peak_column:
            render_metric_card(
                "Signal peak",
                f'{peak["engagement_index"]:.1f}/100',
                f'At {peak["time_s"]:.1f}s · proxy',
            )
        with dip_column:
            render_metric_card(
                "Signal dip",
                f'{dip["engagement_index"]:.1f}/100',
                f'At {dip["time_s"]:.1f}s · proxy',
            )

        timeline = px.line(
            fused,
            x="time_s",
            y="engagement_index",
            markers=True,
            labels={"time_s": "Time (seconds)", "engagement_index": "Proxy score"},
            title="Engagement proxy timeline",
        )
        timeline.add_hline(
            y=low,
            line_dash="dash",
            line_color="#F59E0B",
            annotation_text="Low-signal threshold",
        )
        st.plotly_chart(configure_chart(timeline), use_container_width=True)

        contributions = fused[["time_s", "visual_proxy"]].copy()
        audio_available = fused["interaction_proxy"].notna()
        contributions["Visual contribution"] = (
            fused["visual_proxy"].where(~audio_available, fused["visual_proxy"] * 0.7) * 100
        )
        if audio_available.any():
            contributions["Audio contribution"] = (
                fused["interaction_proxy"].fillna(0).where(audio_available, 0) * 30
            )
            contribution_chart = px.area(
                contributions,
                x="time_s",
                y=["Visual contribution", "Audio contribution"],
                labels={"time_s": "Time (seconds)", "value": "Contribution"},
                title="Visual and audio contribution",
                color_discrete_sequence=["#6366F1", "#22D3EE"],
            )
        else:
            contribution_chart = px.area(
                contributions,
                x="time_s",
                y="Visual contribution",
                labels={"time_s": "Time (seconds)", "value": "Contribution"},
                title="Visual contribution (audio not analyzed)",
                color_discrete_sequence=["#6366F1"],
            )
        distribution_chart = px.histogram(
            fused,
            x="engagement_index",
            nbins=12,
            labels={"engagement_index": "Proxy score"},
            title="Engagement proxy distribution",
            color_discrete_sequence=["#6366F1"],
        )
        chart_columns = st.columns(2)
        with chart_columns[0]:
            st.plotly_chart(configure_chart(contribution_chart), use_container_width=True)
        with chart_columns[1]:
            st.plotly_chart(configure_chart(distribution_chart), use_container_width=True)
        st.download_button(
            "Download engagement data CSV",
            data=fused.to_csv(index=False).encode("utf-8"),
            file_name="engagement_timeline.csv",
            mime="text/csv",
        )


with suggestions_tab:
    render_section_header(
        "Smart Teaching Assistant",
        "Rules-based teaching prompts grounded in this session's available signals.",
    )
    if fused.empty:
        render_empty_state("Process a classroom video to generate session-specific suggestions.")
    else:
        average_score = float(fused["engagement_index"].mean())
        trend, trend_change = trend_summary(fused)
        if average_score < low:
            engagement_level = "Low collective signal"
        elif average_score < 65:
            engagement_level = "Moderate collective signal"
        else:
            engagement_level = "Elevated collective signal"
        severity, severity_kind = severity_for(average_score, float(low), trend)

        metric_columns = st.columns(3)
        with metric_columns[0]:
            render_metric_card(
                "Current classroom engagement level",
                engagement_level,
                f"{average_score:.1f}/100 proxy",
            )
        with metric_columns[1]:
            trend_detail = (
                f"{trend_change:+.1f} points across session"
                if trend_change is not None
                else "More data needed"
            )
            render_metric_card("Engagement trend", trend, trend_detail)
        with metric_columns[2]:
            st.markdown(
                '<div class="metric-label">Alert severity</div>',
                unsafe_allow_html=True,
            )
            render_status_badge(severity, severity_kind)

        st.markdown("### Teaching recommendations")
        suggestions = advisories(fused, config)
        for suggestion in suggestions:
            st.markdown(
                f'<div class="advisory-card">{escape(suggestion)}</div>',
                unsafe_allow_html=True,
            )
        if trend == "Decreasing":
            st.markdown(
                '<div class="advisory-card">The session signal is trending downward. '
                "Consider an interactive question or a brief comprehension check.</div>",
                unsafe_allow_html=True,
            )
        st.caption(
            "These are heuristic prompts, not predictions from an external AI model. "
            "Use classroom context and educator judgment."
        )


with reports_tab:
    render_section_header(
        "Session Analytics & Reports",
        "A concise summary of the analyses available in this session.",
    )
    register = st.session_state.register
    total_tracks = len(register)
    present_tracks = (
        int((register["Status"] == "Present").sum())
        if total_tracks
        else 0
    )
    average_score = (
        float(fused["engagement_index"].mean())
        if not fused.empty
        else None
    )
    audio = st.session_state.audio
    summary_columns = st.columns(4)
    with summary_columns[0]:
        render_metric_card(
            "Session duration",
            format_duration(st.session_state.duration)
            if st.session_state.duration > 0
            else "Not analyzed",
        )
    with summary_columns[1]:
        render_metric_card(
            "Participants detected",
            str(total_tracks) if total_tracks else "Not analyzed",
            "Anonymous tracks",
        )
    with summary_columns[2]:
        attendance_rate = (
            f"{100 * present_tracks / total_tracks:.1f}%"
            if total_tracks
            else "Not analyzed"
        )
        render_metric_card("Present tracks", str(present_tracks) if total_tracks else "—", attendance_rate)
    with summary_columns[3]:
        render_metric_card(
            "Average engagement",
            f"{average_score:.1f}/100" if average_score is not None else "Not analyzed",
            "Heuristic proxy",
        )

    if not audio.empty:
        category_counts = audio["audio_state"].value_counts()
        acoustic_summary = (
            f'{category_counts.index[0]} ({int(category_counts.iloc[0])} windows); '
            f'mean RMS {audio["rms"].mean():.3f}, peak RMS {audio["rms"].max():.3f}.'
        )
    else:
        acoustic_summary = "No audio analysis available."
    st.markdown("#### Acoustic activity")
    st.write(acoustic_summary)

    if not fused.empty:
        report_timeline = px.line(
            fused,
            x="time_s",
            y="engagement_index",
            markers=True,
            labels={"time_s": "Time (seconds)", "engagement_index": "Proxy score"},
            title="Session engagement timeline",
        )
        st.plotly_chart(configure_chart(report_timeline), use_container_width=True)
        if not register.empty:
            render_attendance_table(register)
        suggestions = advisories(fused, config)
    else:
        render_empty_state("Process a classroom video to include an engagement timeline.")
        suggestions = []

    if st.button("Reset session", key="reports_reset"):
        reset_session()
        st.rerun()

    report_lines = [
        "# Classroom Pulse — Session Report",
        f"Generated (UTC): {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Summary",
        f"Session duration: {format_duration(st.session_state.duration) if st.session_state.duration > 0 else 'Not analyzed'}",
        f"Participants detected: {total_tracks if total_tracks else 'Not analyzed'}",
        f"Present tracks: {present_tracks if total_tracks else 'Not analyzed'}",
        f"Attendance threshold: {threshold * 100:.0f}%",
        f"Average engagement proxy: {f'{average_score:.1f}/100' if average_score is not None else 'Not analyzed'}",
        f"Acoustic activity: {acoustic_summary}",
        "",
        "## Engagement timeline",
    ]
    if fused.empty:
        report_lines.append("Not available; no video analysis was performed.")
    else:
        report_lines.append("| Time (seconds) | Engagement proxy |")
        report_lines.append("| ---: | ---: |")
        sample_step = max(1, len(fused) // 50)
        sample_indices = list(range(0, len(fused), sample_step))
        if len(fused) - 1 not in sample_indices:
            sample_indices.append(len(fused) - 1)
        for _, point in fused.iloc[sample_indices].iterrows():
            report_lines.append(
                f'| {point["time_s"]:.1f} | {point["engagement_index"]:.1f} |'
            )
    report_lines.extend(["", "## Teaching suggestions"])
    report_lines.extend(f"- {item}" for item in suggestions)
    if not suggestions:
        report_lines.append("Not available; no video analysis was performed.")
    if not register.empty:
        report_lines.extend(
            [
                "",
                "## Anonymous attendance register",
                "| Student/Track ID | First Seen (s) | Last Seen (s) | Duration (s) | Attendance Status |",
                "| --- | ---: | ---: | ---: | --- |",
            ]
        )
        for _, item in register.iterrows():
            report_lines.append(
                f'| {item["Student ID"]} | {item["First Seen (s)"]:.1f} | '
                f'{item["Last Seen (s)"]:.1f} | {item["Total Duration (s)"]:.1f} | '
                f'{item["Status"]} |'
            )
    report_lines.extend(
        [
            "",
            "Note: Track IDs are anonymous and are not verified identities. Presence, "
            "acoustic categories, and engagement values are heuristic proxies.",
        ]
    )
    st.download_button(
        "Download Session Report",
        data="\n".join(report_lines).encode("utf-8"),
        file_name="classroom_pulse_session_report.md",
        mime="text/markdown",
    )

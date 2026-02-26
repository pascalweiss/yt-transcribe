from concurrent.futures import ThreadPoolExecutor, as_completed

from yt_transcribe.downloader import fetch_channel_video_ids, fetch_video_info
from yt_transcribe.log import log
from yt_transcribe.output import sanitize, get_transcribed_ids
from yt_transcribe.transcriber import TranscribeConfig, TranscribeResult, transcribe_single
from yt_transcribe.whisper import WhisperBackend


def run_channel_mode(
    channel_url: str,
    config: TranscribeConfig,
    backend: WhisperBackend,
    min_seconds: int = 60,
    amount: int = 0,
    workers: int = 1,
    after_ts: int | None = None,
    before_ts: int | None = None,
    tab: str = "videos",
) -> list[TranscribeResult]:
    """Transcribe videos from a YouTube channel."""
    filter_parts = []
    if tab != "videos":
        filter_parts.append(f"tab: {tab}")
    if tab != "shorts":
        filter_parts.append(f"min duration: {min_seconds}s")
    if after_ts is not None or before_ts is not None:
        filter_parts.append("date filter active")
    log(f"Fetching video list ({', '.join(filter_parts)})...")

    all_ids = fetch_channel_video_ids(channel_url, min_seconds, after_ts, before_ts, tab=tab)

    if not all_ids:
        log("No videos found.")
        return []

    total_count = len(all_ids)

    # Get channel name from first available video
    channel_name = ""
    for vid in all_ids[:5]:
        try:
            first_info = fetch_video_info(f"https://www.youtube.com/watch?v={vid}")
            channel_name = first_info.channel
            break
        except Exception:
            continue
    ch_dir = sanitize(channel_name) or "unknown_channel"
    log(f"Channel: {channel_name or ch_dir}")

    # Find already-transcribed video IDs
    done_ids = get_transcribed_ids(config.output_dir / ch_dir)

    # Filter to pending
    pending_ids = [vid for vid in all_ids if vid not in done_ids]
    done_count = total_count - len(pending_ids)
    log(f"Found {total_count} videos, {done_count} already transcribed, {len(pending_ids)} remaining")

    if not pending_ids:
        log("Nothing to transcribe.")
        return []

    # Apply amount limit
    process_count = len(pending_ids)
    if amount > 0 and process_count > amount:
        process_count = amount

    amount_msg = f" (--amount={amount})" if amount > 0 else ""
    log(f"Will transcribe {process_count} videos{amount_msg} with {workers} worker(s)")

    # Print plan
    print("\n  Videos to transcribe:")
    for i in range(process_count):
        print(f"    {i + 1}. https://www.youtube.com/watch?v={pending_ids[i]}")
    print()

    to_process = pending_ids[:process_count]

    if workers <= 1:
        results = _run_sequential(to_process, process_count, config, backend)
    else:
        results = _run_parallel(to_process, process_count, config, backend, workers)

    succeeded = sum(1 for r in results if r.success)
    failed = sum(1 for r in results if not r.success)
    log(f"Batch complete: {succeeded} succeeded, {failed} failed")

    return results


def _run_sequential(
    video_ids: list[str],
    total: int,
    config: TranscribeConfig,
    backend: WhisperBackend,
) -> list[TranscribeResult]:
    results = []
    for i, vid in enumerate(video_ids):
        url = f"https://www.youtube.com/watch?v={vid}"
        label = f"{i + 1}/{total}"
        result = transcribe_single(url, config, backend, worker_id=0, progress_label=label)
        results.append(result)
    return results


def _run_parallel(
    video_ids: list[str],
    total: int,
    config: TranscribeConfig,
    backend: WhisperBackend,
    workers: int,
) -> list[TranscribeResult]:
    results = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for i, vid in enumerate(video_ids):
            url = f"https://www.youtube.com/watch?v={vid}"
            label = f"{i + 1}/{total}"
            worker_id = (i % workers) + 1
            future = executor.submit(transcribe_single, url, config, backend, worker_id, label)
            futures[future] = vid

        for future in as_completed(futures):
            results.append(future.result())

    return results

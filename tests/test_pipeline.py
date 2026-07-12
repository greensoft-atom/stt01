import json

from conftest import FakeEngine, make_wav

from courtstt.pipeline import file_key, run_batch


def test_batch_produces_all_outputs(cfg, fake_engine, tmp_path):
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    make_wav(in_dir / "a.wav")
    out_dir = tmp_path / "out"

    summary = run_batch(cfg, fake_engine, in_dir, out_dir)

    assert [r.name for r in summary.done] == ["a.wav"]
    assert (out_dir / "a.json").exists()
    assert (out_dir / "a.txt").exists()
    assert (out_dir / "a.srt").exists()
    assert (out_dir / "manifest.json").exists()
    assert (out_dir / "stt01.log").exists()

    data = json.loads((out_dir / "a.json").read_text(encoding="utf-8"))
    assert len(data["segments"]) == 2
    assert data["info"]["duration"] == 6.0
    # second segment is low confidence -> flagged count recorded
    assert summary.done[0].flagged == 1


def test_resume_skips_done_files(cfg, fake_engine, tmp_path):
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    make_wav(in_dir / "a.wav")
    out_dir = tmp_path / "out"

    run_batch(cfg, fake_engine, in_dir, out_dir)
    summary2 = run_batch(cfg, fake_engine, in_dir, out_dir)

    assert summary2.skipped == ["a.wav"]
    assert summary2.done == []
    assert len(fake_engine.calls) == 1  # engine not called again


def test_modified_file_is_reprocessed(cfg, fake_engine, tmp_path):
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    wav = make_wav(in_dir / "a.wav")
    out_dir = tmp_path / "out"

    run_batch(cfg, fake_engine, in_dir, out_dir)
    wav.write_bytes(b"RIFF1111WAVE-different")  # size change -> new key
    summary2 = run_batch(cfg, fake_engine, in_dir, out_dir)

    assert [r.name for r in summary2.done] == ["a.wav"]


def test_one_bad_file_does_not_kill_batch(cfg, fake_engine, tmp_path):
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    make_wav(in_dir / "fail.wav")
    make_wav(in_dir / "good.wav")
    out_dir = tmp_path / "out"

    summary = run_batch(cfg, fake_engine, in_dir, out_dir)

    assert [r.name for r in summary.failed] == ["fail.wav"]
    assert [r.name for r in summary.done] == ["good.wav"]
    # failure is recorded so a fixed file (new mtime/size) reprocesses, same file skips? no:
    # failed files are NOT marked done, so they retry on the next run
    summary2 = run_batch(cfg, fake_engine, in_dir, out_dir)
    assert [r.name for r in summary2.failed] == ["fail.wav"]
    assert summary2.skipped == ["good.wav"]


def test_cancellation_stops_between_files(cfg, fake_engine, tmp_path):
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    make_wav(in_dir / "a.wav")
    make_wav(in_dir / "b.wav")
    out_dir = tmp_path / "out"

    calls = {"n": 0}

    def stop_after_first():
        calls["n"] += 1
        return calls["n"] > 1  # allow first file, stop before second

    summary = run_batch(cfg, fake_engine, in_dir, out_dir, should_stop=stop_after_first)

    assert summary.cancelled
    assert [r.name for r in summary.done] == ["a.wav"]


def test_non_audio_files_ignored(cfg, fake_engine, tmp_path):
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    (in_dir / "notes.txt").write_text("not audio")
    summary = run_batch(cfg, fake_engine, in_dir, tmp_path / "out")
    assert summary.done == [] and summary.failed == []


def test_file_key_changes_with_content(tmp_path):
    wav = make_wav(tmp_path / "a.wav")
    k1 = file_key(wav)
    wav.write_bytes(b"RIFF0000WAVE plus more bytes")
    assert file_key(wav) != k1

"""Regression coverage for speech rendering scope, using an isolated fake synthesizer."""
import tempfile
import threading
import unittest
import wave
from pathlib import Path

from text_director import DirectorError, render_directed_audio


def write_wav(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as audio:
        audio.setparams((1, 2, 22050, 0, "NONE", "not compressed"))
        audio.writeframes(bytes(4410))


class Synthesizer:
    def __init__(self):
        self.calls = []

    def infer(self, **kwargs):
        self.calls.append(kwargs)
        write_wav(Path(kwargs["output_path"]))
        return kwargs["output_path"]


def role(role_id, voice, condition="自然"):
    kind = "narrator" if role_id == "narrator" else "character"
    return [role_id, role_id, kind, "资料", condition, voice, "自然叙述", "否"]


def sentence(order, speaker):
    return [order, "第一章", speaker, speaker, "ZH", "测试。", "测试。", "中性叙述", "平静", 0.5, "自然", 0]


class RenderVoiceScopeTest(unittest.TestCase):
    def render(self, root, model, roles, sentences, orders=None):
        voice = root / "voice-current.wav"
        write_wav(voice)
        return render_directed_audio(
            document={"title": "测试"}, role_table=roles, segment_table=sentences,
            model=model, model_lock=threading.Lock(), output_root=root / "renders",
            demo_dir=root, demo_voices={"unused-demo.wav": "无关演示音色"},
            uploaded_files=[str(voice), str(root / "missing.wav")],
            generated_files=[str(root / "voice-missing.wav")],
            fragment_only_orders=orders, project_process_dir=root / "process",
        )

    def test_single_sentence_ignores_other_speaker_and_visual_roles(self):
        for unused_voice in ("", None, "voice-missing", "missing.wav"):
            with self.subTest(voice=unused_voice), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                model = Synthesizer()
                audio, *_ = self.render(
                    root, model,
                    [role("narrator", "voice-current"), role("other", unused_voice), role("visual", "", "")],
                    [sentence(1, "narrator"), sentence(2, "other")], [1],
                )
                self.assertTrue(Path(audio).is_file())
                self.assertEqual(len(model.calls), 1)
                self.assertEqual(Path(model.calls[0]["spk_audio_prompt"]), (root / "voice-current.wav").resolve())

    def test_full_render_allows_silent_visual_role(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Synthesizer()
            self.render(Path(directory), model, [role("narrator", "voice-current"), role("visual", "", "")], [sentence(1, "narrator")])
            self.assertEqual(len(model.calls), 1)

    def test_missing_selected_voice_blocks_before_synthesis(self):
        for orders in (None, [2]):
            for voice in ("", "voice-absent"):
                with self.subTest(orders=orders, voice=voice), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    model = Synthesizer()
                    with self.assertRaisesRegex(DirectorError, "角色 other.*音色"):
                        self.render(root, model, [role("narrator", "voice-current"), role("other", voice)],
                                    [sentence(1, "narrator"), sentence(2, "other")], orders)
                    self.assertEqual(model.calls, [])
                    self.assertFalse((root / "renders").exists())

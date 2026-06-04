"""IEMOCAP dataset loader (stub — requires license from USC SAIL).

IEMOCAP is a gated dataset. Request access at https://sail.usc.edu/iemocap/

Once you have the dataset under ``data_dir`` with the standard layout
(Session1..Session5 with sentences/wav/ and dialog/EmoEvaluation/), implement
the ``_index_sessions`` method below to populate ``self.records``.

The standard SER benchmark on IEMOCAP uses 4 classes:
    happy (merging happy + excited), sad, angry, neutral.
"""
from __future__ import annotations

from pathlib import Path

from src.data.base import BaseSERDataset

# Standard 4-class IEMOCAP setup (happy includes excited per common practice)
IEMOCAP_EMOTIONS = ["neutral", "happy", "sad", "angry"]
IEMOCAP_LABEL2ID = {emo: i for i, emo in enumerate(IEMOCAP_EMOTIONS)}
IEMOCAP_ID2LABEL = {i: emo for emo, i in IEMOCAP_LABEL2ID.items()}


class IEMOCAPDataset(BaseSERDataset):
    """IEMOCAP dataset (4-class).

    Args:
        data_dir: Root IEMOCAP directory.
        sessions: Which session numbers (1-5) to include. Use for leave-one-session-out CV.
    """

    def __init__(
        self,
        data_dir: str | Path,
        sessions: list[int] | None = None,
        sample_rate: int = 16000,
        max_audio_seconds: float = 8.0,
    ):
        super().__init__(sample_rate=sample_rate, max_audio_seconds=max_audio_seconds)
        self.data_dir = Path(data_dir)
        self.sessions = sessions or [1, 2, 3, 4, 5]
        self.records: list[dict] = self._index_sessions()

    def _index_sessions(self) -> list[dict]:
        """Walk the IEMOCAP directory and build a list of
        {audio_path, text, label} records.

        TODO: implement once IEMOCAP access is granted. Standard recipe:
          1. Read each Session<N>/dialog/EmoEvaluation/*.txt file.
          2. Parse [start_time - end_time] utterance_id emotion_code dims lines.
          3. Map ``exc`` → happy and keep only neu/hap/exc/sad/ang.
          4. Match utterance_id to Session<N>/sentences/wav/<dialog>/<utt>.wav.
          5. Get transcript from Session<N>/dialog/transcriptions/<dialog>.txt.
        """
        raise NotImplementedError(
            "IEMOCAP loader not yet implemented — requires dataset access. "
            "See docstring for the standard parsing recipe."
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        audio = self.load_audio(rec["audio_path"], target_sr=self.sample_rate)
        audio = self.pad_or_truncate(audio)
        return {
            "audio": audio,
            "text": rec["text"],
            "label": rec["label"],
            "utterance_id": rec["utterance_id"],
        }

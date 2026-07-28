# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

import pytest

from s2s_service.segmentizer import length_segmentizer
from s2s_service.segmentizer import sentence_segmentizer

pytestmark = pytest.mark.unit


class TestSentenceSegmentizer(unittest.TestCase):
    def test_sengmentizer_content(self):
        """Test that the segmentizer correctly splits sentences."""
        # Test data
        test_text = """
        But I must explain to you how all this mistaken idea of denouncing pleasure and praising pain was 
        born and I will give you a complete account of the system, and expound the actual teachings of 
        the great explorer of the truth, the master-builder of human happiness.
        No one rejects, dislikes, or avoids pleasure itself, because it is pleasure, but because those 
        who do not know how to pursue pleasure rationally encounter consequences that are extremely painful.
        Nor again is there anyone who loves or pursues or desires to obtain pain of itself, because 
        it is pain, but because occasionally circumstances occur in which toil and pain can procure him 
        some great pleasure.
        To take a trivial example, which of us ever undertakes laborious physical exercise, except 
        to obtain some advantage from it? But who has any right to find fault with a man who chooses 
        to enjoy a pleasure that has no annoying consequences, or one who avoids a pain that produces 
        no resultant pleasure?
        On the other hand, we denounce with righteous indignation and dislike men who are so 
        beguiled and demoralized by the charms of pleasure of the moment, so blinded by desire, 
        that they cannot foresee the pain and trouble that are bound to ensue; and equal blame 
        belongs to those who fail in their duty through weakness of will, which is the same as saying 
        through shrinking from toil and pain.
        These cases are perfectly simple and easy to distinguish. In a free hour, when our power of 
        choice is untrammelled and when nothing prevents our being able to do what we like best, 
        every pleasure is to be welcomed and every pain avoided.
        But in certain circumstances and owing to the claims of duty or the obligations of business 
        it will frequently occur that pleasures have to be repudiated and annoyances accepted.
        """

        # Expected first few sentences for validation
        expected_iterations = [
            "But I must explain to you how all this mistaken idea of denouncing pleasure and praising pain was born and I will give you a complete account of the system, and expound the actual teachings of the great explorer of the truth, the master-builder of human happiness.",
            "No one rejects, dislikes, or avoids pleasure itself, because it is pleasure, but because those who do not know how to pursue pleasure rationally encounter consequences that are extremely painful.",
            "Nor again is there anyone who loves or pursues or desires to obtain pain of itself, because it is pain, but because occasionally circumstances occur in which toil and pain can procure him some great pleasure.",
        ]

        chunks = list(sentence_segmentizer(test_text))

        # Test first few sentences
        for i, expected in enumerate(expected_iterations):
            # Normalize whitespace in both expected and actual output
            normalized_expected = " ".join(expected.split())
            normalized_actual = " ".join(chunks[i].split())
            assert normalized_actual == normalized_expected, (
                f"Sentence {i} mismatch:\nExpected: {normalized_expected}\nGot: {normalized_actual}"
            )

    def test_segmentizer_empty_input(self):
        """Test segmentizer behavior with empty input."""
        chunks = list(sentence_segmentizer(""))
        self.assertEqual(
            len(chunks),
            0,
            "Empty input should produce no chunks",
        )

    def test_segmentizer_single_sentence(self):
        """Test segmentizer with a single sentence."""
        test_text = "This is a single sentence."
        chunks = list(sentence_segmentizer(test_text))
        self.assertEqual(
            len(chunks),
            1,
            "Single sentence should produce one chunk",
        )
        self.assertEqual(chunks[0].strip(), test_text, "Chunk content should match input")

    def test_segmentizer_sentence_boundaries(self):
        """Test that sentences are properly split at boundaries."""
        test_text = "First sentence. Second sentence! Third sentence?"
        chunks = list(sentence_segmentizer(test_text))
        self.assertEqual(
            len(chunks),
            3,
            "Should split into three sentences",
        )
        self.assertEqual(chunks[0].strip(), "First sentence.")
        self.assertEqual(chunks[1].strip(), "Second sentence!")
        self.assertEqual(chunks[2].strip(), "Third sentence?")

    def test_sentence_segmentizer_basic(self):
        text = "Hello world. This is a test! Is it working? Yes."
        expected = ["Hello world. ", " This is a test! ", " Is it working? ", " Yes. "]
        result = list(sentence_segmentizer(iter(text)))
        self.assertEqual(result, expected)

    def test_sentence_segmentizer_empty(self):
        self.assertEqual(list(sentence_segmentizer(iter([]))), [])

    def test_sentence_segmentizer_no_punctuation(self):
        text = ["This is a sentence without punctuation"]
        expected = ["This is a sentence without punctuation "]
        result = list(sentence_segmentizer(iter(text)))
        self.assertEqual(result, expected)


class TestLengthSegmentizer(unittest.TestCase):
    def test_length_segmentizer_basic(self):
        text = ["a" * 50, "b" * 100, "c" * 75]
        chunk_size = 60
        result = list(length_segmentizer(iter(text), chunk_size=chunk_size))
        # Should split into chunks of 60, then 60, then 50+40+25
        expected = ["a" * 50 + "b" * 10, "b" * 60, "b" * 30 + "c" * 30, "c" * 45 + " "]
        self.assertEqual(result, expected)

    def test_length_segmentizer_exact_chunk(self):
        text = ["x" * 100]
        chunk_size = 100
        result = list(length_segmentizer(iter(text), chunk_size=chunk_size))
        self.assertEqual(result, ["x" * 100])

    def test_length_segmentizer_empty(self):
        self.assertEqual(list(length_segmentizer(iter([]))), [])

    def test_length_segmentizer_shorter_than_chunk(self):
        text = ["short"]
        chunk_size = 10
        result = list(length_segmentizer(iter(text), chunk_size=chunk_size))
        self.assertEqual(result, ["short "])


if __name__ == "__main__":
    unittest.main()

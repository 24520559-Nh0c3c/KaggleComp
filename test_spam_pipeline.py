import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, call

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import spam_pipeline as pipeline


class PipelineTests(unittest.TestCase):
    def test_scheduler_steps_only_for_successful_optimizer_updates(self):
        scaler = Mock()
        scaler.get_scale.side_effect = [8, 8, 8, 4, 4, 8]
        optimizer, scheduler = Mock(), Mock()
        for _ in range(3):
            pipeline.optimizer_step(scaler, optimizer, scheduler)
        self.assertEqual(scaler.step.call_args_list, [call(optimizer)] * 3)
        self.assertEqual(scaler.update.call_count, 3)
        self.assertEqual(scheduler.step.call_count, 2)

    def test_stratification_and_training_only_weights(self):
        train = pd.DataFrame({pipeline.LABEL: [0] * 5 + [1] * 10 + [2] * 15 + [3] * 20})
        classes, labels = pipeline.encode_labels(train)
        self.assertEqual(classes, [0, 1, 2, 3])
        seen = []
        for tr, val in StratifiedKFold(5, shuffle=True, random_state=42).split(labels, labels):
            seen.extend(val)
            self.assertFalse(set(tr) & set(val))
            counts = np.bincount(labels[tr], minlength=4)
            np.testing.assert_allclose(pipeline.class_weights(labels[tr]), len(tr) / (4 * counts))
        self.assertEqual(sorted(seen), list(range(len(labels))))

    def test_label_validation_and_dev_unseen(self):
        train = pd.DataFrame({pipeline.LABEL: [0, 1, 2, 3] * 5})
        with self.assertRaisesRegex(ValueError, "Dev contains"):
            pipeline.encode_labels(train, pd.DataFrame({pipeline.LABEL: [4]}))
        with self.assertRaisesRegex(ValueError, "five"):
            pipeline.encode_labels(train.iloc[:-1])
        with self.assertRaisesRegex(ValueError, "four classes"):
            pipeline.encode_labels(train[train[pipeline.LABEL] != 3])
        with self.assertRaisesRegex(ValueError, "integer labels"):
            pipeline.encode_labels(train.assign(spam_label=train.spam_label + 1))

    def test_submission_id_order_and_mismatches(self):
        test = pd.DataFrame({"id": [12, 10, 11]})
        sample = pd.DataFrame({"ID": [10, 11, 12], "spam_label": [0, 0, 0]})
        result = pipeline.make_submission(test, sample, [2, 3, 1])
        self.assertEqual(result.spam_label.tolist(), [3, 1, 2])
        with self.assertRaisesRegex(ValueError, "do not match"):
            pipeline.make_submission(test, sample.assign(ID=[10, 11, 13]), [2, 3, 1])
        with self.assertRaisesRegex(ValueError, "unique"):
            pipeline.make_submission(test, sample.assign(ID=[10, 10, 12]), [2, 3, 1])
        with self.assertRaisesRegex(ValueError, "integer labels"):
            pipeline.make_submission(test, sample, [0, 1, 4])
        self.assertEqual(pipeline.make_submission(test.rename(columns={"id": "ID"}), sample, [2, 3, 1]).spam_label.tolist(), [3, 1, 2])

    def test_validation_text_and_metrics(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "train.csv"
            pd.DataFrame({"comment": ["text"]}).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "missing columns"):
                pipeline.load_frame(path, True)
            pd.DataFrame([{"comment": "hello\x00! 😀", "spam_label": 0}]).to_csv(path, index=False)
            frame = pipeline.load_frame(path, True)
            self.assertEqual(pipeline.make_texts(pd.DataFrame({"review": ["hello\x00! 😀"]})), ["hello ! 😀"])
            self.assertEqual(pipeline.make_texts(frame)[0], "hello ! 😀")
            pd.DataFrame({"text": ["hi"], "label": [0]}).to_csv(path, index=False)
            self.assertEqual(pipeline.load_frame(path, True).spam_label.tolist(), [0])
        result = pipeline.metrics(np.array([0]), np.array([[1, 0, 0, 0]]))
        self.assertEqual(result["accuracy"], 1)
        self.assertEqual(result["macro_f1"], 0.25)

    def test_ensemble_accumulation(self):
        self.assertEqual(len(pipeline.Config().models), 3)
        self.assertEqual(pipeline.Config().patience, 3)
        oof, test = np.zeros((2, 4)), np.zeros((1, 4))
        for model in range(3):
            probabilities = np.eye(4)[[model]]
            for fold in range(5):
                pipeline.accumulate_probabilities(oof, test, np.array([fold % 2]), probabilities, probabilities, 3)
        np.testing.assert_allclose(test, [[1/3, 1/3, 1/3, 0]])
        np.testing.assert_allclose(oof[0], [1, 1, 1, 0])
        np.testing.assert_allclose(oof[1], [2/3, 2/3, 2/3, 0])


if __name__ == "__main__":
    unittest.main()
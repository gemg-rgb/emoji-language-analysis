```python
import unittest
import numpy as np
from collections import namedtuple
from emoji_language_analysis import load_data, preprocess_data, train_naive_bayes, predict_language


class TestEmojiLanguageAnalysis(unittest.TestCase):
    def setUp(self):
        self.test_data = np.array([
            ['', '😄', '🎉'],
            ['Python', '10', '20'],
            ['Java', '5', '15']
        ])
        self.rownames = np.array(['Python', 'Java'])
        self.colnames = np.array(['😄', '🎉'])
        self.x, self.cidx, self.fidx = preprocess_data(self.test_data, self.rownames, self.colnames)
        self.model = train_naive_bayes(self.x, self.rownames, self.colnames)

    def test_load_data(self):
        with self.assertRaises(FileNotFoundError):
            load_data('nonexistent.txt')
        # Note: Cannot test with actual file since it's not provided
        self.assertIsInstance(self.test_data, np.ndarray)
        self.assertEqual(self.colnames.shape[0], 2)
        self.assertEqual(self.rownames.shape[0], 2)

    def test_preprocess_data(self):
        x, cidx, fidx = preprocess_data(self.test_data, self.rownames, self.colnames)
        self.assertEqual(x.shape, (2, 2))
        self.assertEqual(cidx['Python'], 0)
        self.assertEqual(fidx['😄'], 0)
        with self.assertRaises(ValueError):
            invalid_data = np.array([['', '😄'], ['Python', 'invalid']])
            preprocess_data(invalid_data, np.array(['Python']), np.array(['😄']))

    def test_train_naive_bayes(self):
        model = train_naive_bayes(self.x, self.rownames, self.colnames)
        self.assertIsInstance(model, namedtuple)
        self.assertEqual(len(model.classes), 2)
        self.assertEqual(model.features.size, 2)
        self.assertTrue((model.pxc >= 0).all())

    def test_predict_language(self):
        patterns = ['😄']
        predictions = predict_language(self.model, patterns, self.fidx)
        self.assertEqual(len(predictions), 2)
        self.assertIsInstance(predictions[0], tuple)
        self.assertIn(predictions[0][0], ['Python', 'Java'])

    def test_predict_language_invalid_pattern(self):
        with self.assertRaises(KeyError):
            predict_language(self.model, ['🚀'], self.fidx)


if __name__ == '__main__':
    unittest.main()
```
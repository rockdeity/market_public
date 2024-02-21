import unittest
from unittest.mock import patch
import os
import json
import pickle
import pandas as pd
import requests

from modelling_prep import fetch_and_process_data

class TestFetchAndProcessData_grade(unittest.TestCase):

    def setUp(self):
        self.stock = 'AAPL'
        self.endpoint = 'balance-sheet-statement'
        self.data_path = 'test\modeling_prep'

    @patch('modelling_prep.os.path.exists')
    @patch('modelling_prep.pickle.load')
    def test_load_data_from_pickle_file(self, mock_pickle_load, mock_path_exists):
        mock_path_exists.return_value = True
        mock_pickle_load.return_value = pd.DataFrame({'symbol': ['AAPL'], 'date': ['2022-01-01']})

        result = fetch_and_process_data(self.stock, self.endpoint, self.data_path)

        self.assertTrue(mock_path_exists.called)
        self.assertTrue(mock_pickle_load.called)
        self.assertEqual(result.shape, (1, 2))

    # @patch('modelling_prep.os.path.exists')
    # @patch('modelling_prep.json.load')
    # @patch('modelling_prep.requests.get')
    # def test_load_data_from_json_file(self, mock_requests_get, mock_json_load, mock_path_exists):
    #     mock_path_exists.side_effect = [False, True]
    #     mock_json_load.return_value = [{'symbol': 'META', 'date': '2022-01-01'}]
    #     mock_requests_get.return_value.status_code = 200
        
    #     result = fetch_and_process_data('META', self.endpoint, self.data_path)

    #     self.assertTrue(mock_path_exists.called)
    #     self.assertTrue(mock_requests_get.called)
    #     self.assertTrue(mock_json_load.called)
    #     self.assertEqual(result.shape, (1, 2))

    @patch('modelling_prep.os.path.exists')
    @patch('modelling_prep.requests.get')
    def test_fetch_data_from_endpoint(self, mock_requests_get, mock_path_exists):
        mock_path_exists.return_value = False
        mock_requests_get.return_value.status_code = 200
        mock_requests_get.return_value.json.return_value = [{'symbol': 'AAPL', 'date': '2022-01-01'}]

        result = fetch_and_process_data(self.stock, self.endpoint, self.data_path)

        self.assertTrue(mock_path_exists.called)
        self.assertTrue(mock_requests_get.called)
        self.assertEqual(result.shape, (1, 2))

    @patch('modelling_prep.os.path.exists')
    @patch('modelling_prep.requests.get')
    def test_fetch_data_from_endpoint_error(self, mock_requests_get, mock_path_exists):
        mock_path_exists.side_effect = [False, False]
        mock_requests_get.return_value.status_code = 404

        result = fetch_and_process_data(self.stock, self.endpoint, self.data_path)

        self.assertTrue(mock_path_exists.called)
        self.assertTrue(mock_requests_get.called)
        self.assertIsNone(result)

    @patch('modelling_prep.os.path.exists')
    @patch('modelling_prep.pickle.dump')
    def test_save_data_to_pickle_file(self, mock_pickle_dump, mock_path_exists):
        mock_path_exists.side_effect = [False, False]

        result = fetch_and_process_data(self.stock, self.endpoint, self.data_path)

        self.assertTrue(mock_path_exists.called)
        self.assertTrue(mock_pickle_dump.called)
        self.assertEqual(result.shape, (1, 1))

if __name__ == '__main__':
    unittest.main()
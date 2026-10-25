"""Scientific invariants: units, chemical identity and evaluation isolation."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
from rdkit import Chem
from affinity_eda import pki_from_nm,structure_features,grouped_split,fingerprint_generator,metrics


class ScientificInvariants(unittest.TestCase):
    def test_affinity_units_and_order(self):
        np.testing.assert_allclose(pki_from_nm([100,1000]),[7,6])
        with self.assertRaises(ValueError):pki_from_nm([0])
        with self.assertRaises(ValueError):pki_from_nm([-1])
        with self.assertRaises(ValueError):pki_from_nm([np.nan])

    def test_salt_removal_preserves_charge(self):
        data=structure_features('C[NH3+].[Cl-]')
        self.assertEqual(data['smiles'],'C[NH3+]')
        self.assertTrue(data['fragment_removed'])

    def test_do_not_guess_components_of_mixture(self):
        self.assertEqual(structure_features('c1ccccc1.c1ccccc1')['structure_status'],'multiple_organic_components')

    def test_chirality_is_a_model_feature(self):
        gen=fingerprint_generator()
        first=gen.GetFingerprint(Chem.MolFromSmiles('C[C@H](O)F'))
        second=gen.GetFingerprint(Chem.MolFromSmiles('C[C@@H](O)F'))
        self.assertNotEqual(first,second)

    def test_grouped_split_never_shares_a_scaffold(self):
        data=pd.DataFrame({'scaffold':[f'S{i//3}' for i in range(90)]})
        labels=grouped_split(data,'scaffold',42)
        groups={name:set(data.loc[labels.eq(name),'scaffold']) for name in ['train','validation','test']}
        self.assertFalse(groups['train']&groups['test'])
        self.assertFalse(groups['train']&groups['validation'])
        self.assertFalse(groups['validation']&groups['test'])
        self.assertEqual(set(labels),{'train','validation','test'})

    def test_constant_control_has_no_rank_correlation(self):
        result=metrics(np.array([4.,5.,6.,7.,8.]),np.full(5,5.21))
        self.assertIsNone(result['Spearman'])


if __name__=='__main__':unittest.main()
